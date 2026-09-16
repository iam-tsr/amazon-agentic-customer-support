import { runPipeline } from "../src/core.js";
import { cosineSim } from "../src/guardrail.js";
import { ONNXEmbed } from "../src/embed.js";

const embedding = new ONNXEmbed();

interface EvaluationResult {
    index: number;
    query: string;
    response: string;
    intentMatch: boolean;
    actionMatch: boolean;
}

export interface EvaluationReport {
    total: number;
    intentMatches: number;
    actionMatches: number;
    intentAccuracy: number;
    actionAccuracy: number;
    responsePassed: number;
    responseAccuracy: number;
    results: EvaluationResult[];
}

interface TestEntry {
    action: string;
    intent: string;
    query: string;
    response: string | null;
}

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
        return entries.slice(0, limit);
    }
    return entries;
}

async function evaluateResponse(expected: string | null, actual: string): Promise<boolean> {
    if (expected === null || expected.trim() === "") {
        return true; // No expected response, consider it passed
    }

    if (!actual.trim()) {
        return false;
    }

    const [expectedEmbedding, actualEmbedding] = await Promise.all([
        embedding.embedQuery(expected),
        embedding.embedQuery(actual),
    ]);

    if (expectedEmbedding.length === 0 || actualEmbedding.length === 0) {
        return false;
    }

    const similarity = cosineSim(expectedEmbedding, actualEmbedding);
    return similarity >= 0.8; // Consider it passed if similarity is above threshold
}

export async function runEvaluation(concurrency: number, limit?: number): Promise<EvaluationReport> {
    const entries = await loadTestDataset(limit);

    const report: EvaluationReport = {
        total: entries.length,
        intentMatches: 0,
        actionMatches: 0,
        intentAccuracy: 0,
        actionAccuracy: 0,
        responsePassed: 0,
        responseAccuracy: 0,
        results: [],
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

                if (await evaluateResponse(entry.response, botResponse)) report.responsePassed++;

                return {
                    index,
                    query: entry.query,
                    response: botResponse,
                    intentMatch,
                    actionMatch
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

    const summary =
        `\n` +
        `═══ Evaluation Report ═══\n` +
        `Total queries   : ${report.total}\n` +
        `Intent accuracy  : ${report.intentAccuracy.toFixed(1)}% (${report.intentMatches}/${report.total})\n` +
        `Action accuracy  : ${report.actionAccuracy.toFixed(1)}% (${report.actionMatches}/${report.total})\n` +
        `Response accuracy: ${report.responseAccuracy.toFixed(1)}% (${report.responsePassed}/${report.total})\n\n` +
        `═══════════════════════════`;

    console.log(summary);
    return report;
}

if (import.meta.main) {
    const actual = "You're very welcome! We're thrilled to hear that you're enjoying your new item. Please don't hesitate to reach out if you need anything else in the future. Have a wonderful day!"
    const expected = "Thank you for your kind words about our packaging. We're glad you enjoyed the experience. If you have any other questions, feel free to reach out."

    const passed = await evaluateResponse(expected, actual);
    console.log(`Response evaluation passed: ${passed}`);
}