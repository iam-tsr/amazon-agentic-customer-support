import { FlagEmbedding, EmbeddingModel, ExecutionProvider } from "fastembed";

export class ONNXEmbed {
    private modelPromise: Promise<FlagEmbedding> | null = null;

    // Lazily initialize the embedding model.
    private getEmbeddingModel(): Promise<FlagEmbedding> {
        if (!this.modelPromise) {
            this.modelPromise = FlagEmbedding.init({
                model: EmbeddingModel.BGESmallENV15,
                cacheDir: "local_cache",
                showDownloadProgress: false,
                executionProviders: [ExecutionProvider.CPU],
            });
        }
        return this.modelPromise;
    }

    // Embed a single user query using the query-side embedder.
    async embedQuery(text: string): Promise<number[]> {
        const model = await this.getEmbeddingModel();
        return model.queryEmbed(text);
    }

    // Embed one or more documents/passages using the passage-side embedder.
    async embedDocuments(texts: string[]): Promise<number[][]> {
        if (texts.length === 0) return [];
        const model = await this.getEmbeddingModel();
        const vectors: number[][] = [];
        // Batches of 4 keep memory flat for large document sets.
        for await (const batch of model.passageEmbed(texts, 4)) {
            vectors.push(...batch);
        }
        return vectors;
    }
}