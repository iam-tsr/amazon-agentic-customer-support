import { ONNXEmbed } from "../../src/embed";


const embedder = new ONNXEmbed();
const documents = [
    "Where is my Amazon order and when will it be delivered?",
    "Prime Video movies and TV shows are not playing.",
];

const docVectors = await embedder.embedDocuments(documents);
console.log(`embedDocuments: ${docVectors.length} vectors, dim=${docVectors[0]?.length}`);

const queryVector = await embedder.embedQuery("query: track my package");
console.log(`embedQuery:    dim=${queryVector.length}`);

// A delivery question should sit closer to the order reference than to
// the Prime Video reference — quick sanity check of the query/document
// embedding pairing.
const dot = (a: number[], b: number[]): number =>
    a.reduce((sum, v, i) => sum + v * (b[i] ?? 0), 0);
const norm = (v: number[]): number => Math.sqrt(dot(v, v));
const cos = (a: number[], b: number[]): number => dot(a, b) / (norm(a) * norm(b) || 1);
console.log(`cos(query, order-ref) = ${cos(queryVector, docVectors[0]!).toFixed(3)}`);
console.log(`cos(query, video-ref) = ${cos(queryVector, docVectors[1]!).toFixed(3)}`);