import { loadDatasetIntoVectorStore, runPipeline, sessions, vectorStore } from "./src/app.js";


const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "0.0.0.0";





function jsonResponse(data: unknown, status = 200): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { "Content-Type": "application/json" },
    });
}

function errorResponse(message: string, status = 400): Response {
    return jsonResponse({ error: message }, status);
}

async function handleRequest(req: Request): Promise<Response> {
    const url = new URL(req.url);

    // --- GET /health ---
    if (req.method === "GET" && url.pathname === "/health") {
        return jsonResponse({
            status: "ok",
            vectorStoreSize: vectorStore.memoryVectors.length,
            activeSessions: sessions.size,
        });
    }

    // --- GET /sessions ---
    if (req.method === "GET" && url.pathname === "/sessions") {
        return jsonResponse({ sessions: [...sessions.keys()] });
    }

    // --- POST /chat ---
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

        const sessionId = rawSessionId ?? `session_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

        try {
            const result = await runPipeline(message.trim(), sessionId);
            return jsonResponse(result);
        } catch (err) {
            console.error(`[${sessionId}] pipeline error:`, err);
            return errorResponse("Internal server error — pipeline failed", 500);
        }
    }

    // --- DELETE /sessions/:id ---
    if (req.method === "DELETE" && url.pathname.startsWith("/sessions/")) {
        const id = url.pathname.slice("/sessions/".length);
        const deleted = sessions.delete(id);
        return jsonResponse({ deleted, sessionId: id });
    }

    return errorResponse("Not found", 404);
}

// ===========================================================================
// Startup
// ===========================================================================

console.log("=".repeat(60));
console.log("  Amazon Customer Support Agent");
console.log("=".repeat(60));

// Load dataset into vector store BEFORE accepting requests (like FastAPI's startup event)
await loadDatasetIntoVectorStore();

const server = Bun.serve({
    hostname: HOST,
    port: PORT,
    fetch: handleRequest,
});

console.log(`\n✅ Server listening on http://${HOST}:${PORT}`);
console.log("\nEndpoints:");
console.log("  POST /chat         { sessionId?: string, message: string }");
console.log("  GET  /health");
console.log("  GET  /sessions");
console.log("  DELETE /sessions/:id\n");