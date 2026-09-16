import {
    loadDatasetIntoVectorStore,
    runPipeline,
    buildSessionSummary,
    sessions,
    vectorStore,
} from "./src/core.js";
import { runEvaluation, EvaluationReport } from "./tests/test_eval.js";

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "0.0.0.0";


// Response helpers

function jsonResponse(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data, null, 2), {
        status,
        headers: { "Content-Type": "application/json" },
    });
}

function errorResponse(message: string, status = 400): Response {
    return jsonResponse({ error: message }, status);
}

// Request router

async function handleRequest(req: Request): Promise<Response> {
    const url = new URL(req.url);

    // GET /health
    if (req.method === "GET" && url.pathname === "/health") {
        return jsonResponse({
            status: "ok",
            vectorStoreSize: vectorStore.memoryVectors.length,
            activeSessions: sessions.size,
        });
    }

    // GET /sessions
    if (req.method === "GET" && url.pathname === "/sessions") {
        return jsonResponse({ sessions: [...sessions.keys()] });
    }

    // POST /chat
    if (req.method === "POST" && url.pathname === "/chat") {
        let body: { sessionId?: string; message?: string };
        try {
            body = (await req.json()) as { sessionId?: string; message?: string };
        } catch {
            return errorResponse("Invalid JSON body");
        }

        const { message, sessionId: rawSessionId } = body;
        if (!message || typeof message !== "string" || message.trim() === "") {
            return errorResponse("'message' field is required and must be a non-empty string");
        }

        const sessionId =
            rawSessionId ??
            `session_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

        try {
            const result = await runPipeline(message.trim(), sessionId);
            return jsonResponse(result);
        } catch (err) {
            console.error(`[${sessionId}] pipeline error:`, err);
            return errorResponse("Internal server error — pipeline failed", 500);
        }
    }

    // POST /end-chat
    if (req.method === "POST" && url.pathname === "/end-chat") {
        let body: { sessionId?: string };
        try {
            body = (await req.json()) as { sessionId?: string };
        } catch {
            return errorResponse("Invalid JSON body");
        }

        const { sessionId } = body;
        if (!sessionId || typeof sessionId !== "string") {
            return errorResponse("'sessionId' field is required");
        }

        const summary = buildSessionSummary(sessionId);
        if (!summary) {
            return errorResponse(`Session '${sessionId}' not found`, 404);
        }

        // Clean up the session after summarizing
        sessions.delete(sessionId);
        console.log(`[${sessionId}] session ended — action=${summary.action} intent=${summary.intent}`);

        return jsonResponse(summary);
    }

    // DELETE /sessions/:id
    if (req.method === "DELETE" && url.pathname.startsWith("/sessions/")) {
        const id = url.pathname.slice("/sessions/".length);
        const deleted = sessions.delete(id);
        return jsonResponse({ deleted, sessionId: id });
    }

    // POST /test — trigger evaluation test
    if (req.method === "POST" && url.pathname === "/test") {
        let body: { concurrency?: number; limit?: number };
        try {
            body = await req.json() as { concurrency?: number; limit?: number };
            const concurrency = body.concurrency ?? 3;
            const limit = body.limit;
            const report: EvaluationReport = await runEvaluation(concurrency, limit);
            return jsonResponse(report, 200);
        } catch (err) {
            console.error("[test] Evaluation failed:", err);
            return errorResponse("Evaluation test failed", 500);
        }
    }

    return errorResponse("Not found", 404);
}


// Startup
console.log("=".repeat(60));
console.log("  Amazon Customer Support Agent");
console.log("=".repeat(60));
console.log(`  LLM endpoint : ${process.env.OPENAI_API_BASE_URL}`);
console.log(`  Model        : ${process.env.OPENAI_MODEL}`);
console.log(`  Port         : ${PORT}`);
console.log("=".repeat(60));

// Load all Q&A pairs into vector store BEFORE accepting requests
await loadDatasetIntoVectorStore();

const server = Bun.serve({
    hostname: HOST,
    port: PORT,
    fetch: handleRequest,
});

console.log(`\n✅ Server listening on http://${HOST}:${PORT}`);
console.log("\nEndpoints:");
console.log("  POST   /chat         { sessionId?: string, message: string }");
console.log("  POST   /end-chat     { sessionId: string }");
console.log("  POST   /test         { limit?: number of test entries to evaluate }");
console.log("  GET    /health");
console.log("  GET    /sessions");
console.log("  DELETE /sessions/:id\n");