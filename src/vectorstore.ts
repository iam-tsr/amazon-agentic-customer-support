/**
 * Vector store built on LangChain's in-memory vector store, following the
 * documented pattern:
 * https://docs.langchain.com/oss/javascript/integrations/vectorstores#adding-documents
 *
 * The two assignment operations, per the LangChain unified interface:
 *
 *   Document add:              await vectorStore.addDocuments([document]);
 *   Cosine similarity search:  const results = await vectorStore
 *                                  .similaritySearch("Hello world", 10);
 *
 * MemoryVectorStore's default similarity metric is cosine similarity, and it
 * accepts any embeddings model implementing EmbeddingsInterface
 * (embedDocuments + embedQuery) — here defaulting to the local ONNX
 * BGESmallENV15 embedder so the store works offline out of the box.
 */
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { ONNXEmbed } from "./embed.js";

export class VectorStore extends MemoryVectorStore {
    /**
     * Follows the documented initialization pattern: pass an embeddings
     * model when constructing the vector store.
     */
    constructor(embeddings: EmbeddingsInterface = new ONNXEmbed()) {
        // super() wires the embeddings model; MemoryVectorStore defaults
        // its similarity function to cosine similarity.
        super(embeddings);
    }
}

// ===========================================================================
// Self-test: `bun src/vectorstore.ts`
// ===========================================================================

if (import.meta.main) {
    // Documented usage: new VectorStore(embeddings) -> addDocuments ->
    // similaritySearch(query, k).
    const vectorStore = new VectorStore();

    const document = new Document({
        pageContent: "Hello world",
    });
    await vectorStore.addDocuments([document]);

    const documents = [
        new Document({
            pageContent: "Where is my Amazon order and when will it be delivered?",
            metadata: { topic: "orders" },
        }),
        new Document({
            pageContent: "Prime Video movies and TV shows are not playing.",
            metadata: { topic: "video" },
        }),
        new Document({
            pageContent: "How do I reset my account password?",
            metadata: { topic: "account" },
        }),
    ];
    await vectorStore.addDocuments(documents);
    console.log(`stored ${vectorStore.memoryVectors.length} documents`);

    const results = await vectorStore.similaritySearch("Hello world", 10);
    for (const doc of results) console.log(`  ${doc.pageContent}`);

    const scored = await vectorStore.similaritySearchWithScore(
        "query: track my package",
        2,
    );
    for (const [doc, score] of scored) {
        console.log(
            `  score=${score.toFixed(3)}  topic=${String(doc.metadata.topic)}  "${doc.pageContent}"`,
        );
    }
}
