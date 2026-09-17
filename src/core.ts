import { Document } from "@langchain/core/documents";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";

import { classifyUserInput } from "./guardrail.js";
import { VectorStore } from "./vectorstore.js";
import { ONNXEmbed } from "./embed.js";
import { classifyIntent } from "./intent_classify.js";
import { SYSTEM_PROMPT, PREDEFINED_PROMPT, JUDGE_PROMPT, JUDGE_KNOWLEDGE_PROMPT } from "./support/prompt.js";

import customerQueryDataset from "./dataset/customer_query.json";



const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_API_BASE_URL = process.env.OPENAI_API_BASE_URL;
const OPENAI_MODEL = process.env.OPENAI_MODEL;

try {
    if (!OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY");
    if (!OPENAI_API_BASE_URL) throw new Error("Missing OPENAI_API_BASE_URL");
    if (!OPENAI_MODEL) throw new Error("Missing OPENAI_MODEL");
} catch (err) {
    console.error("[config] LLM client configuration error:", err);
    process.exit(1);
}

export const openai = new ChatOpenAI({
    openAIApiKey: OPENAI_API_KEY,
    configuration: {
        baseURL: OPENAI_API_BASE_URL,
    },
    modelName: OPENAI_MODEL,
    temperature: 0.3,
    maxRetries: 2,
});

// In-memory vector store (loaded once at startup)
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

// Session state
export type Intent = "complaint" | "question" | "positive" | "other" | "unknown";
export type Action = "resolve" | "escalated_to_human";

export interface SessionState {
    /** Full chat history (system + human + AI messages) */
    history: BaseMessage[];
    /** ML-classified intent from LR model (set after guardrail passes) */
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

// Intent classification via Logistic Regression (ML model — Python subprocess)
// async function classifyIntent(text: string): Promise<Intent> {
//     try {
//         const proc = Bun.spawn(
//             ["python3.11", "src/intent_classify.py", text],
//             { cwd: process.cwd(), stdout: "pipe", stderr: "pipe" }
//         );
//         const output = await new Response(proc.stdout).text();
//         await proc.exited;
//         const label = output.trim().toLowerCase() as Intent;
//         const valid: Intent[] = ["complaint", "question", "positive", "other"];
//         return valid.includes(label) ? label : "other";
//     } catch (err) {
//         console.warn("[intent] ML classification failed, defaulting to 'other':", err);
//         return "other";
//     }
// }

/**
 * Judge 1 — decide whether the query can be answered directly or
 * requires retrieval from the knowledge base.
 * Returns true  → direct answer (no RAG needed)
 * Returns false → require knowledge (proceed to retrieval + Judge 2)
 */
async function judgeNeedsKnowledge(
    userMessage: string,
    intent: Intent,
    history: BaseMessage[]
): Promise<boolean> {
    const judgeMessages = [
        new SystemMessage(JUDGE_KNOWLEDGE_PROMPT),
        ...history.filter((m) => !(m instanceof SystemMessage)),
        new HumanMessage(
            `Customer intent: ${intent}\nCustomer message: "${userMessage}"\n\nDoes answering this query require looking up specific knowledge (order details, policies, product info, etc.)? Reply ONLY with "REQUIRES_KNOWLEDGE" or "DIRECT_ANSWER".`
        ),
    ];

    const judgeResponse = await openai.invoke(judgeMessages);
    const verdict = String(judgeResponse.content).trim().toUpperCase();
    return verdict.startsWith("REQUIRES_KNOWLEDGE");
}

// Generate a direct answer using only chat history (no RAG)
async function generateDirectAnswer(
    userMessage: string,
    intent: Intent,
    history: BaseMessage[]
): Promise<string> {
    const messages = [
        ...history,
        new HumanMessage(
            `Customer intent: ${intent}\nCustomer message: ${userMessage}\n\nPlease provide a concise, helpful response.`
        ),
    ];

    const response = await openai.invoke(messages);
    return String(response.content).trim();
}

// Retrieve top-k similar Q&A pairs from the vector store
async function retrieve(userMessage: string, k = 3): Promise<Array<[Document, number]>> {
    return vectorStore.similaritySearchWithScore(userMessage, k);
}

/**
 * Judge 2 — LLM as Judge — decide if retrieved docs are reliable enough
 * to generate a grounded answer.
 * Returns true  → reliable (generate output)
 * Returns false → not reliable (escalate to human)
 */
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

    const judgeResponse = await openai.invoke(judgeMessages);
    const verdict = String(judgeResponse.content).trim().toUpperCase();
    return verdict.startsWith("RELIABLE");
}

// Generate response using retrieved context (RAG output)
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

    const response = await openai.invoke(messages);
    return String(response.content).trim();
}

// Blocked / out-of-scope — uses pre-defined prompt (no LLM for hard blocks)
async function generatePredefinedResponse(
    userMessage: string,
    reason: string,
    history: BaseMessage[]
): Promise<string> {
    // Hard security blocks never go to the LLM
    if (reason === "jailbreak" || reason === "injection") {
        return "I cannot process requests that attempt to override my instructions or inject content into this conversation. If you have an Amazon-related question, I'm happy to help!";
    }

    const messages = [
        new SystemMessage(PREDEFINED_PROMPT),
        ...history.filter((m) => !(m instanceof SystemMessage)),
        new HumanMessage(userMessage),
    ];

    const response = await openai.invoke(messages);
    return String(response.content).trim();
}


export interface ChatResult {
    sessionId: string;
    intent: Intent;
    action: Action;
    response: string;
}

export interface SessionSummary {
    action: Action;
    intent: Intent;
    conversation: Array<{ role: "human" | "ai"; content: string }>;
}

export async function runPipeline(userMessage: string, sessionId: string): Promise<ChatResult> {
    const session = getSession(sessionId);
    const { history } = session;

    let responseText: string;
    let action: Action;

    // Guardrail
    const verdict = await classifyUserInput(userMessage);
    console.log(`[${sessionId}] guardrail: ${verdict.kind}`);

    if (verdict.kind === "BLOCKED" || verdict.kind === "SCOPE_NOTICE") {
        // Pre-defined prompt (blocked / out-of-scope)
        const reason = verdict.kind === "BLOCKED" ? verdict.reason : "scope_notice";
        responseText = await generatePredefinedResponse(userMessage, reason, history);
        action = "resolve";
        console.log(`[${sessionId}] branch: predefined-prompt (${reason})`);
    } else {
        // LR intent classification
        session.intent = await classifyIntent(userMessage);
        console.log(`[${sessionId}] intent (LR): ${session.intent}`);

        // Judge 1 — direct answer vs. require knowledge
        const requiresKnowledge = await judgeNeedsKnowledge(userMessage, session.intent, history);
        console.log(`[${sessionId}] judge-1: ${requiresKnowledge ? "REQUIRES_KNOWLEDGE" : "DIRECT_ANSWER"}`);

        if (!requiresKnowledge) {
            // Direct answer (no RAG)
            responseText = await generateDirectAnswer(userMessage, session.intent, history);
            action = "resolve";
            console.log(`[${sessionId}] branch: direct-answer`);
        } else {
            // Require knowledge (RAG)
            const retrievedDocs = await retrieve(userMessage);
            console.log(`[${sessionId}] retrieved ${retrievedDocs.length} docs`);

            // Judge 2 — RAG reliability
            const reliable = await judgeReliability(userMessage, retrievedDocs);
            console.log(`[${sessionId}] judge-2: ${reliable ? "RELIABLE" : "NOT_RELIABLE"}`);

            if (reliable) {
                // Branch B2a: Generated output (RAG)
                responseText = await generateWithContext(userMessage, retrievedDocs, history);
                action = "resolve";
                console.log(`[${sessionId}] branch: generated-output`);
            } else {
                // Branch B2b: Escalate to human
                responseText =
                    "I'm sorry, I don't have enough information to help with this. Let me connect you with a human agent who can assist you further.";
                action = "escalated_to_human";
                console.log(`[${sessionId}] branch: escalated-to-human`);
            }
        }
    }

    history.push(new HumanMessage(userMessage));
    history.push(new AIMessage(responseText));
    session.lastAction = action;

    return { sessionId, intent: session.intent, action, response: responseText };
}

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