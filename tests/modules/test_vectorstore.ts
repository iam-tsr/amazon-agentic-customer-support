import { Document } from "@langchain/core/documents";

import { VectorStore } from "../../src/vectorstore";


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
