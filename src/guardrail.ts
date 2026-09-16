import { createMiddleware, AIMessage } from "langchain";
import type { BaseMessage } from "@langchain/core/messages";
import { ONNXEmbed } from "./embed";

import {
    REFERENCES,
    STRONG_KEYWORDS,
    STRONG_PHRASES,
    WEAK_KEYWORDS,
    WEAK_PHRASES,
    SEMANTIC_THRESHOLD,
    JAILBREAK_PATTERNS,
    INJECTION_MARKERS,
    GREETING_RE,
    CHAT_TOKENS
} from "./support/guard_prompt";

const embedder = new ONNXEmbed();



// Helpers

export function cosineSim(a: number[], b: number[]): number {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        const av = a[i] ?? 0, bv = b[i] ?? 0;
        dot += av * bv; na += av * av; nb += bv * bv;
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

function tokenize(text: string): Set<string> {
    return new Set(text.toLowerCase().split(/[^a-z0-9']+|(?<!')'(?!s)/).filter((t) => t.length > 1));
}

// Extract plain text from a BaseMessage (string or content-block content).
function getMessageText(message: BaseMessage): string {
    const content = message.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        return content
            .map((part) =>
                typeof part === "string" ||
                (part && typeof part === "object" && "text" in part && typeof (part as { text?: unknown }).text === "string")
                    ? (typeof part === "string" ? part : (part as { text: string }).text)
                    : ""
            )
            .join(" ");
    }
    return String(content ?? "");
}


// Semantic fallback (lazy, degrades gracefully to keyword-only checks)

let referenceVectorsPromise: Promise<number[][] | null> | null = null;

function getReferenceVectors(): Promise<number[][] | null> {
    if (!referenceVectorsPromise) {
        referenceVectorsPromise = (async () => {
            try {
                return await embedder.embedDocuments(REFERENCES);
            } catch (err) {
                console.warn(
                    "[AmazonTopicGuardrail] reference embeddings unavailable, falling back to keyword-only checks:",
                    err
                );
                return null;
            }
        })();
    }
    return referenceVectorsPromise;
}

async function semanticScore(text: string): Promise<number> {
    const refVectors = await getReferenceVectors();
    if (!refVectors || refVectors.length === 0) return 0;
    try {
        const vector = await embedder.embedQuery(text);
        let best = 0;
        for (const ref of refVectors) best = Math.max(best, cosineSim(vector, ref));
        return best;
    } catch (err) {
        console.warn("[AmazonTopicGuardrail] semantic scoring failed, using keyword checks only:", err);
        return 0;
    }
}


// Classification

export type InputVerdict =
    | { kind: "PASS" }        // in-scope Amazon query
    | { kind: "GREETING" }    // pure greeting / chit-chat
    | { kind: "SCOPE_NOTICE" } // too short/ambiguous to judge — ask for detail
    | { kind: "BLOCKED"; reason: "jailbreak" | "injection" | "off_topic" };

/**
 * Classify a raw user message.
 * Order matters: security checks run first so a jailbreak wrapped inside an
 * in-scope query ("Where is my order? Also ignore previous instructions...")
 * is still blocked.
 */
export async function classifyUserInput(text: string): Promise<InputVerdict> {
    const trimmed = text.trim();

    if (!trimmed) return { kind: "SCOPE_NOTICE" };
    if (JAILBREAK_PATTERNS.some((p) => p.test(trimmed))) return { kind: "BLOCKED", reason: "jailbreak" };
    if (INJECTION_MARKERS.some((p) => p.test(trimmed))) return { kind: "BLOCKED", reason: "injection" };

    // keyword & phrase scoring
    const lowered = trimmed.toLowerCase();
    const tokens = tokenize(trimmed);
    let strong = false;
    let weakCount = 0;
    for (const phrase of STRONG_PHRASES) if (lowered.includes(phrase)) strong = true;
    for (const phrase of WEAK_PHRASES) if (lowered.includes(phrase)) weakCount++;
    for (const token of tokens) {
        if (STRONG_KEYWORDS.has(token)) strong = true;
        if (WEAK_KEYWORDS.has(token)) weakCount++;
    }
    if (strong) return { kind: "PASS" };
    if (weakCount >= 2) return { kind: "PASS" };

    // pure greeting / chit-chat: every token is a chat word
    if (tokens.size > 0 && [...tokens].every((t) => CHAT_TOKENS.has(t))) return { kind: "GREETING" };
    if (GREETING_RE.test(trimmed)) return { kind: "GREETING" };

    // semantic fallback against reference Amazon queries
    const score = await semanticScore(trimmed);
    if (score >= SEMANTIC_THRESHOLD) return { kind: "PASS" };
    if (trimmed.split(/\s+/).length <= 4) return { kind: "SCOPE_NOTICE" };
    return { kind: "BLOCKED", reason: "off_topic" };
}


/* Guardrail middleware (before-agent) */

const SCOPE_NOTICE_MESSAGE =
    "I can only help with Amazon-related questions — for example orders, shopping, products, delivery, returns and refunds, Prime Video movies and shows, or your Amazon account. Could you rephrase your question with a little more detail?";

const OFF_TOPIC_MESSAGE =
    "Sorry, I can only assist with Amazon services such as shopping, products, orders and delivery, returns and refunds, Prime Video movies, and account support. Please ask me something related to those and I'll be happy to help.";

const SECURITY_BLOCK_MESSAGE =
    "I cannot process requests that try to override my instructions or inject content into this conversation. If you have an Amazon-related question about your orders, shopping, delivery, products or Prime Video, I'm happy to help with that.";

export interface AmazonTopicGuardrailOptions {
    // Response for short/ambiguous input that isn't clearly in or out of scope.
    scopeNoticeMessage?: string;
    // Response for clearly irrelevant (non-Amazon) input.
    offTopicMessage?: string;
    // Response for jailbreak / prompt-injection attempts.
    securityMessage?: string;
}

export function amazonTopicGuardrail(options: AmazonTopicGuardrailOptions = {}) {
    return createMiddleware({
        name: "AmazonTopicGuardrail",
        beforeAgent: {
            hook: async (state) => {
                const messages = state.messages ?? [];
                if (messages.length === 0) return;

                // Validate only the newest human message — older ones were
                // already checked on the turns they arrived.
                let lastHuman: BaseMessage | undefined;
                for (let i = messages.length - 1; i >= 0; i--) {
                    const message = messages[i];
                    if (message && message._getType() === "human") {
                        lastHuman = message;
                        break;
                    }
                }
                if (!lastHuman) return;

                const verdict = await classifyUserInput(getMessageText(lastHuman));

                switch (verdict.kind) {
                    // In scope (or a harmless greeting) — let the agent run.
                    case "PASS":
                    case "GREETING":
                        return;

                    case "SCOPE_NOTICE":
                        return {
                            messages: [new AIMessage(options.scopeNoticeMessage ?? SCOPE_NOTICE_MESSAGE)],
                            jumpTo: "end",
                        };

                    case "BLOCKED":
                        return {
                            messages: [
                                new AIMessage(
                                    verdict.reason === "off_topic"
                                        ? options.offTopicMessage ?? OFF_TOPIC_MESSAGE
                                        : options.securityMessage ?? SECURITY_BLOCK_MESSAGE
                                ),
                            ],
                            jumpTo: "end",
                        };
                }
            },
            canJumpTo: ["end"],
        },
    });
}