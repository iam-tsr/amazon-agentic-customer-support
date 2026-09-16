import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
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