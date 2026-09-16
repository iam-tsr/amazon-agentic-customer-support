/**
 * Evaluation test — loads the local JSONL dataset (src/dataset/test_data.jsonl),
 * feeds every query through the pipeline, then compares intent / action / response
 * against the ground truth.
 *
 * For entries where the dataset response is null (escalated to human) we use an
 * LLM-as-judge to evaluate the bot's generated escalation response.
 * For entries with a non-null dataset response we also use the LLM-as-judge to
 * assess semantic similarity between the bot's output and the expected response.
 */

import { runPipeline } from "../src/core.js";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// Types

interface TestEntry {
    action: string;
    intent: string;
    query: string;
    response: string | null;
}

interface ResponseEvaluation {
    passed: boolean;
    score: number;
    feedback: string;
}

interface EvaluationResult {
    index: number;
    query: string;
    expectedIntent: string;
    actualIntent: string;
    intentMatch: boolean;
    expectedAction: string;
    actualAction: string;
    actionMatch: boolean;
    responseEval: ResponseEvaluation;
}

export interface EvaluationReport {
    total: number;
    resolved: number;
    escalated: number;
    intentMatches: number;
    actionMatches: number;
    responsePassed: number;
    intentAccuracy: number;
    actionAccuracy: number;
    responseAccuracy: number;
    results: EvaluationResult[];
    summary: string;
}

// LLM-as-Judge client

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_API_BASE_URL = process.env.OPENAI_API_BASE_URL;
const OPENAI_MODEL = process.env.OPENAI_MODEL;

function createJudge(): ChatOpenAI {
    if (!OPENAI_API_KEY || !OPENAI_API_BASE_URL || !OPENAI_MODEL) {
        throw new Error("Missing OPENAI_API_KEY / OPENAI_API_BASE_URL / OPENAI_MODEL env vars");
    }
    return new ChatOpenAI({
        openAIApiKey: OPENAI_API_KEY,
        configuration: { baseURL: OPENAI_API_BASE_URL },
        modelName: OPENAI_MODEL,
        temperature: 0.1,
    });
}

// Judge Prompts

const JUDGE_RESPONSE_PROMPT = `You are an evaluation judge for an Amazon customer support chatbot.

You will receive three things:
1. The customer's original query
2. The chatbot's generated response
3. The expected ground-truth response (from a labelled dataset)

Your task is to score how well the chatbot's response matches the expected response on a scale of 0 to 100.

Rules:
- Score 90-100: The response is essentially equivalent to the expected response.
- Score 70-89: The response addresses the core issue but differs in wording or approach.
- Score 50-69: The response partially addresses the query but misses key points.
- Score 30-49: The response is off-topic or provides only minimal help.
- Score 0-29: The response is completely wrong or unhelpful.

Return ONLY a JSON object: {"score": <0-100>, "feedback": "<explanation>"}`;

const JUDGE_ESCALATION_PROMPT = `You are an evaluation judge for an Amazon customer support chatbot.

You will receive the customer's original query. The dataset labels this query as "escalated_to_human" (expected response is null).

Evaluate the chatbot's generated response for this escalation scenario.

A good escalation response should:
- Acknowledge the customer's frustration empathetically
- Indicate the bot cannot fully resolve the issue
- Offer or initiate escalation to a human agent
- Be polite and professional

Score 0-100:
- 90-100: Perfectly handles escalation
- 70-89: Good escalation
- 50-69: Adequate
- 30-49: Poor
- 0-29: Fails at escalation

Return ONLY a JSON object: {"score": <0-100>, "feedback": "<explanation>"}`;

// JSONL Loader

interface RawTestEntry {
    action: string;
    intent: string;
    query: string;
    response: string | null;
}

async function loadTestDataset(limit?: number): Promise<TestEntry[]> {
    const filePath = new URL("../src/dataset/test_data.jsonl", import.meta.url);
    const file = Bun.file(filePath);
    const text = await file.text();
    const lines = text.trim().split("\n").filter((l) => l.trim().length > 0);
    const entries: TestEntry[] = [];
    for (const line of lines) {
        try {
            const parsed: RawTestEntry = JSON.parse(line);
            entries.push({
                action: parsed.action,
                intent: parsed.intent,
                query: parsed.query,
                response: parsed.response,
            });
        } catch {
            // skip malformed lines
        }
    }
    if (limit !== undefined) {
        return entries.slice(1, 2);
    }
    return entries;
}

// LLM-as-Judge Functions

async function judgeResponseQuality(
    query: string,
    botResponse: string,
    expectedResponse: string,
): Promise<ResponseEvaluation> {
    const judge = createJudge();
    const messages = [
        new SystemMessage(JUDGE_RESPONSE_PROMPT),
        new HumanMessage(
            `Customer query: "${query}"\n\n` +
                `Chatbot response: "${botResponse}"\n\n` +
                `Expected response: "${expectedResponse}"\n\n` +
                `Score the chatbot's response.`,
        ),
    ];

    try {
        const judgeResponse = await judge.invoke(messages);
        const raw = String(judgeResponse.content).trim();
        const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] ?? raw;
        const result = JSON.parse(jsonStr);
        return {
            passed: result.score >= 70,
            score: typeof result.score === "number" ? result.score : 0,
            feedback: result.feedback ?? "No feedback provided.",
        };
    } catch (err) {
        console.warn("[judge-response] Error:", err);
        return { passed: false, score: 0, feedback: "Judge evaluation failed." };
    }
}

async function judgeEscalationResponse(query: string, botResponse: string): Promise<ResponseEvaluation> {
    const judge = createJudge();
    const messages = [
        new SystemMessage(JUDGE_ESCALATION_PROMPT),
        new HumanMessage(
            `Customer query: "${query}"\n\n` +
                `Chatbot response: "${botResponse}"\n\n` +
                `Score the chatbot's escalation response.`,
        ),
    ];

    try {
        const judgeResponse = await judge.invoke(messages);
        const raw = String(judgeResponse.content).trim();
        const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0] ?? raw;
        const result = JSON.parse(jsonStr);
        return {
            passed: result.score >= 70,
            score: typeof result.score === "number" ? result.score : 0,
            feedback: result.feedback ?? "No feedback provided.",
        };
    } catch (err) {
        console.warn("[judge-escalation] Error:", err);
        return { passed: false, score: 0, feedback: "Judge evaluation failed." };
    }
}

// Main Evaluation Function

export interface TestConfig {
    concurrency?: number;
    responsePassThreshold?: number;
}

export async function runEvaluation(config: TestConfig = {}, limit?: number): Promise<EvaluationReport> {
    const { concurrency = 5 } = config;
    const entries = await loadTestDataset(limit);

    const report: EvaluationReport = {
        total: entries.length,
        resolved: 0,
        escalated: 0,
        intentMatches: 0,
        actionMatches: 0,
        responsePassed: 0,
        intentAccuracy: 0,
        actionAccuracy: 0,
        responseAccuracy: 0,
        results: [],
        summary: "",
    };

    for (let i = 0; i < entries.length; i += concurrency) {
        const batch = entries.slice(i, i + concurrency);
        const batchResults = await Promise.all(
            batch.map(async (entry, batchIndex) => {
                const index = i + batchIndex;
                const sessionId = `eval_${index}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

                let actualIntent = "unknown";
                let actualAction: "resolve" | "escalated_to_human" = "resolve";
                let botResponse = "";

                try {
                    const result = await runPipeline(entry.query, sessionId);
                    actualIntent = result.intent;
                    actualAction = result.action;
                    botResponse = result.response;
                } catch (err) {
                    console.error(`[eval] Pipeline failed for entry ${index}:`, err);
                }

                const intentMatch = actualIntent === entry.intent;
                if (intentMatch) report.intentMatches++;

                const actionMatch = actualAction === entry.action;
                if (actionMatch) report.actionMatches++;

                let responseEval: ResponseEvaluation;
                if (entry.response === null) {
                    report.escalated++;
                    responseEval = await judgeEscalationResponse(entry.query, botResponse);
                } else {
                    report.resolved++;
                    responseEval = await judgeResponseQuality(entry.query, botResponse, entry.response);
                }

                if (responseEval.passed) report.responsePassed++;

                return {
                    index,
                    query: entry.query,
                    expectedIntent: entry.intent,
                    actualIntent,
                    intentMatch,
                    expectedAction: entry.action,
                    actualAction,
                    actionMatch,
                    responseEval,
                };
            }),
        );

        report.results.push(...batchResults);

        const done = Math.min(i + concurrency, entries.length);
        console.log(`[eval] Progress: ${done}/${entries.length} evaluated`);
    }

    report.intentAccuracy = report.total > 0 ? (report.intentMatches / report.total) * 100 : 0;
    report.actionAccuracy = report.total > 0 ? (report.actionMatches / report.total) * 100 : 0;
    report.responseAccuracy = report.total > 0 ? (report.responsePassed / report.total) * 100 : 0;

    const passed = report.results.filter(
        (r) => r.intentMatch && r.actionMatch && r.responseEval.passed,
    ).length;
    report.summary =
        `\n` +
        `═══ Evaluation Report ═══\n` +
        `Total queries   : ${report.total}\n` +
        `  Resolved      : ${report.resolved}\n` +
        `  Escalated     : ${report.escalated}\n\n` +
        `Intent accuracy  : ${report.intentAccuracy.toFixed(1)}% (${report.intentMatches}/${report.total})\n` +
        `Action accuracy  : ${report.actionAccuracy.toFixed(1)}% (${report.actionMatches}/${report.total})\n` +
        `Response accuracy: ${report.responseAccuracy.toFixed(1)}% (${report.responsePassed}/${report.total})\n\n` +
        `Overall passed   : ${passed}/${report.total}\n` +
        `═══════════════════════════`;

    console.log(report.summary);
    return report;
}