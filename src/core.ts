/**
 * Amazon Customer Support Agent — Pipeline Core
 *
 * Architecture (from diagram):
 *   User Input
 *     └─► Intent Classification (ML model — first turn only)
 *     └─► Guardrail
 *           ├─ BLOCKED / SCOPE_NOTICE ──► LR:intent (predefined prompt) ──► Response
 *           └─ PASS / GREETING
 *                 └─► Retrieve (vector store from customer_query.json)
 *                       └─► LLM as Judge (reliable?)
 *                             ├─ Reliable ──────────────► Response (action: resolve)
 *                             └─ Not reliable ──────────► Escalate to human
 *   Response ──► Chat History (per session)
 *
 * POST /end-chat  →  returns structured session summary
 */

import { Document } from "@langchain/core/documents";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";

import { classifyUserInput } from "./guardrail.js";
import { VectorStore } from "./vectorstore.js";
import { ONNXEmbed } from "./embed.js";
import { SYSTEM_PROMPT, LR_INTENT_PROMPT, JUDGE_PROMPT } from "./prompt.js";

import customerQueryDataset from "./dataset/customer_query.json";

// ===========================================================================
// Config
// ===========================================================================

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_API_BASE_URL = process.env.OPENAI_API_BASE_URL;
const OPENAI_MODEL = process.env.OPENAI_MODEL;

// ===========================================================================
// LLM client (shared, stateless)
// ===========================================================================

const llm = new ChatOpenAI({
    openAIApiKey: OPENAI_API_KEY,
    configuration: { baseURL: OPENAI_API_BASE_URL },
    modelName: OPENAI_MODEL,
    temperature: 0.3,
});

// ===========================================================================
// In-memory vector store (loaded once at startup)
// ===========================================================================

type QueryEntry = { query: string; response: string };

export const vectorStore = new VectorStore(new ONNXEmbed());

export async function loadDatasetIntoVectorStore(): Promise<void> {
    const entries: QueryEntry[] = (customerQueryDataset as { AmazonHelp: QueryEntry[] }).AmazonHelp;
    console.log(`[startup] Loading ${entries.length} Q&A pairs into vector store…`);
    const documents = entries.map(
        (entry) =>
            new Document({
                pageContent: entry.query,
                metadata: { response: entry.response },
            })
    );

    await vectorStore.addDocuments(documents);
    console.log(`[startup] Vector store ready — ${vectorStore.memoryVectors.length} vectors indexed.`);
}

// ===========================================================================
// Session state
// ===========================================================================

export type Intent = "complaint" | "question" | "positive" | "other" | "unknown";
export type Action = "resolve" | "escalate_to_human";

export interface SessionState {
    /** Full chat history (system + human + AI messages) */
    history: BaseMessage[];
    /** ML-classified intent from the first user message */
    intent: Intent;
    /** Final action taken in the last pipeline run */
    lastAction: Action;
}

export const sessions = new Map<string, SessionState>();

function getSession(sessionId: string): SessionState {
    if (!sessions.has(sessionId)) {
        sessions.set(sessionId, {
            history: [new SystemMessage(SYSTEM_PROMPT)],
            intent: "unknown",
            lastAction: "resolve",
        });
    }
    return sessions.get(sessionId)!;
}

// ===========================================================================
// Intent classification (ML model via Python subprocess)
// ===========================================================================

/**
 * Runs the sklearn TF-IDF + LogisticRegression model (tweet_classify.joblib)
 * as a Python3.11 subprocess and returns the predicted intent label.
 * Only called on the FIRST message of a new session.
 */
async function classifyIntent(text: string): Promise<Intent> {
    try {
        const proc = Bun.spawn(
            ["python3.11", "src/intent_classify.py", text],
            { cwd: process.cwd(), stdout: "pipe", stderr: "pipe" }
        );
        const output = await new Response(proc.stdout).text();
        await proc.exited;
        const label = output.trim().toLowerCase() as Intent;
        const valid: Intent[] = ["complaint", "question", "positive", "other"];
        return valid.includes(label) ? label : "other";
    } catch (err) {
        console.warn("[intent] ML classification failed, defaulting to 'other':", err);
        return "other";
    }
}

// ===========================================================================
// Pipeline steps
// ===========================================================================

/** Step: Retrieve top-k similar Q&A pairs from the vector store */
async function retrieve(userMessage: string, k = 3): Promise<Array<[Document, number]>> {
    return vectorStore.similaritySearchWithScore(userMessage, k);
}

/** Step: LLM as Judge — decide if retrieved docs are reliable */
async function judgeReliability(
    userMessage: string,
    retrievedDocs: Array<[Document, number]>
): Promise<boolean> {
    const docsText = retrievedDocs
        .map(([doc, score], i) => {
            const resp = String(doc.metadata.response ?? "");
            return `[${i + 1}] (score=${score.toFixed(3)})\nQ: ${doc.pageContent}\nA: ${resp}`;
        })
        .join("\n\n");

    const judgeMessages = [
        new SystemMessage(JUDGE_PROMPT),
        new HumanMessage(
            `Customer query: "${userMessage}"\n\nRetrieved examples:\n${docsText}`
        ),
    ];

    const judgeResponse = await llm.invoke(judgeMessages);
    const verdict = String(judgeResponse.content).trim().toUpperCase();
    return verdict.startsWith("RELIABLE");
}

/** Step: Generate response using retrieved context */
async function generateWithContext(
    userMessage: string,
    retrievedDocs: Array<[Document, number]>,
    history: BaseMessage[]
): Promise<string> {
    const context = retrievedDocs
        .map(([doc], i) => {
            const resp = String(doc.metadata.response ?? "");
            return `Example ${i + 1}:\nCustomer: ${doc.pageContent}\nAgent: ${resp}`;
        })
        .join("\n\n");

    const messages = [
        ...history,
        new HumanMessage(
            `Context from knowledge base:\n${context}\n\nCustomer message: ${userMessage}\n\nPlease provide a helpful, empathetic response tailored to this customer.`
        ),
    ];

    const response = await llm.invoke(messages);
    return String(response.content).trim();
}

/** Step: Off-topic / blocked — uses LR:intent predefined prompt */
async function generateLRIntentResponse(
    userMessage: string,
    reason: string,
    history: BaseMessage[]
): Promise<string> {
    // Hard security blocks never go to the LLM
    if (reason === "jailbreak" || reason === "injection") {
        return "I cannot process requests that attempt to override my instructions or inject content into this conversation. If you have an Amazon-related question, I'm happy to help!";
    }

    const messages = [
        new SystemMessage(LR_INTENT_PROMPT),
        ...history.filter((m) => !(m instanceof SystemMessage)),
        new HumanMessage(userMessage),
    ];

    const response = await llm.invoke(messages);
    return String(response.content).trim();
}

// ===========================================================================
// Pipeline result types
// ===========================================================================

export interface ChatResult {
    sessionId: string;
    intent: Intent;
    action: Action;
}

export interface SessionSummary {
    action: Action;
    intent: Intent;
    conversation: Array<{ role: "human" | "ai"; content: string }>;
}

// ===========================================================================
// Main pipeline
// ===========================================================================

export async function runPipeline(userMessage: string, sessionId: string): Promise<ChatResult> {
    const session = getSession(sessionId);
    const { history } = session;

    // --- Step 0: Intent classification (ML model, first turn only) ---
    const isFirstTurn = session.intent === "unknown";
    if (isFirstTurn) {
        session.intent = await classifyIntent(userMessage);
        console.log(`[${sessionId}] intent: ${session.intent}`);
    }

    // --- Step 1: Guardrail ---
    const verdict = await classifyUserInput(userMessage);
    console.log(`[${sessionId}] guardrail: ${verdict.kind}`);

    let action: Action;

    if (verdict.kind === "BLOCKED" || verdict.kind === "SCOPE_NOTICE") {
        // LR:intent branch
        const reason = verdict.kind === "BLOCKED" ? verdict.reason : "scope_notice";
        // Blocked queries are handled inline by the agent — not escalated
        action = "resolve";
    } else {
        // PASS or GREETING — go through RAG pipeline
        const retrievedDocs = await retrieve(userMessage);
        console.log(`[${sessionId}] retrieved ${retrievedDocs.length} docs`);

        // --- LLM as Judge ---
        const reliable = await judgeReliability(userMessage, retrievedDocs);
        console.log(`[${sessionId}] judge: ${reliable ? "RELIABLE" : "NOT_RELIABLE"}`);

        if (reliable) {
            action = "resolve";
        } else {
            // Escalate to human
            action = "escalate_to_human";
        }
    }

    // --- Append to chat history ---
    history.push(new HumanMessage(userMessage));
    session.lastAction = action;

    return { sessionId, intent: session.intent, action };
}

// ===========================================================================
// Session summary (end-of-conversation structured output)
// ===========================================================================

export function buildSessionSummary(sessionId: string): SessionSummary | null {
    const session = sessions.get(sessionId);
    if (!session) return null;

    const conversation = session.history
        .filter((m) => m instanceof HumanMessage || m instanceof AIMessage)
        .map((m) => ({
            role: (m instanceof HumanMessage ? "human" : "ai") as "human" | "ai",
            content: String(m.content),
        }));

    return {
        action: session.lastAction,
        intent: session.intent,
        conversation,
    };
}