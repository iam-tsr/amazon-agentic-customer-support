import { createMiddleware, AIMessage } from "langchain";

const contentFilterMiddleware = (bannedKeywords: string[]) => {
    const keywords = bannedKeywords.map(kw => kw.toLowerCase());

    return createMiddleware({
        name: "ContentFilterMiddleware",
        beforeAgent: {
            hook: (state) => {
                // Get the first user message
                if (!state.messages || state.messages.length === 0) {
                    return;
                }

                const firstMessage = state.messages[0];
                if (firstMessage._getType() !== "human") {
                    return;
                }

                const content = firstMessage.content.toString().toLowerCase();

                // Check for banned keywords
                for (const keyword of keywords) {
                    if (content.includes(keyword)) {
                        // Block execution before any processing
                        return {
                            messages: [
                                new AIMessage(
                                    "I cannot process requests containing inappropriate content. Please rephrase your request."
                                )
                            ],
                            jumpTo: "end",
                        };
                    }
                }

                return;
            },
            canJumpTo: ['end']
        }
    });
};

// Use the custom guardrail
import { createAgent } from "langchain";

const agent = createAgent({
    model: "gpt-5.5",
    tools: [searchTool, calculatorTool],
    middleware: [
        contentFilterMiddleware(["hack", "exploit", "malware"]),
    ],
});

// This request will be blocked before any processing
const result = await agent.invoke({
    messages: [{ role: "user", content: "How do I hack into a database?" }]
});