/**
 * Embedding utilities built on fastembed (BGESmallENV15, 384 dims).
 *
 * Exposes two functions matching the model's asymmetric embedding scheme:
 *   - embedQuery(text)    for short user queries (query embedder)
 *   - embedDocuments(texts) for reference passages/documents (passage embedder)
 *
 * The underlying model is a lazily-initialized singleton: nothing is loaded
 * until the first embedding call, so importing this module stays cheap.
 */
import { FlagEmbedding, EmbeddingModel } from "fastembed";

let modelPromise: Promise<FlagEmbedding> | null = null;

/** Lazily initialize the (shared) embedding model singleton. */
function getEmbeddingModel(): Promise<FlagEmbedding> {
    if (!modelPromise) {
        modelPromise = FlagEmbedding.init({
            model: EmbeddingModel.BGESmallENV15,
            cacheDir: "local_cache",
            showDownloadProgress: false,
        });
    }
    return modelPromise;
}

/** Embed a single user query using the query-side embedder. */
export async function embedQuery(text: string): Promise<number[]> {
    const model = await getEmbeddingModel();
    return model.queryEmbed(text);
}

/** Embed one or more documents/passages using the passage-side embedder. */
export async function embedDocuments(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const model = await getEmbeddingModel();
    const vectors: number[][] = [];
    // Batches of 4 keep memory flat for large document sets.
    for await (const batch of model.passageEmbed(texts, 4)) {
        vectors.push(...batch);
    }
    return vectors;
}

// ===========================================================================
// Self-test: `bun src/embed.ts`
// ===========================================================================

if (import.meta.main) {
    const documents = [
        "Where is my Amazon order and when will it be delivered?",
        "Prime Video movies and TV shows are not playing.",
    ];

    const docVectors = await embedDocuments(documents);
    console.log(`embedDocuments: ${docVectors.length} vectors, dim=${docVectors[0]?.length}`);

    const queryVector = await embedQuery("query: track my package");
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
}
