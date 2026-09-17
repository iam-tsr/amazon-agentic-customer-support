import * as ort from 'onnxruntime-node';

const modelPath = new URL('./model/tweet_classify.onnx', import.meta.url).pathname;

export type IntentLabel = 'complaint' | 'question' | 'positive' | 'other';

const intentLabels = new Set<IntentLabel>(['complaint', 'question', 'positive', 'other']);

export async function classifyIntent(text: string): Promise<IntentLabel> {
    const session = await ort.InferenceSession.create(modelPath);
    const tensor = new ort.Tensor('string', [text], [1, 1]);
    const outputs = await session.run({ text_input: tensor }, ['output_label']);
    const labelOutput = outputs.output_label;

    if (!labelOutput || !('data' in labelOutput)) {
        throw new Error('ONNX model did not return output_label');
    }

    const label = String(labelOutput.data[0]).toLowerCase();
    return intentLabels.has(label as IntentLabel) ? (label as IntentLabel) : 'other';
}