// Representative in-scope queries used as anchors for semantic similarity.
export const REFERENCES = [
    "Where is my order and when will it be delivered?",
    "How do I track my Amazon package or shipment?",
    "My Amazon parcel is late, delayed, or lost in shipping.",
    "I want to change the delivery address for my Amazon order.",
    "How do I cancel my Amazon order before it ships?",
    "I want to return a product I bought on Amazon and get a refund.",
    "The item I received from Amazon is damaged, defective, or the wrong product.",
    "I need a replacement for a product I purchased on Amazon.",
    "The Amazon seller is not responding to my issue with my purchase.",
    "My Amazon product stopped working or broke, quality complaint.",
    "I want to buy this product on Amazon, is it in stock?",
    "Add the item to my Amazon cart and proceed to checkout.",
    "Is this product available on Amazon in my size and preferred colour?",
    "Where can I find Amazon product details, specifications, and reviews?",
    "Prime Video movies and TV shows are not playing or keep buffering.",
    "Subtitles are not working for certain movies I am streaming on Prime Video.",
    "How do I rent, buy, download, or watch a movie on Amazon?",
    "Can you recommend good movies or TV shows to stream on Prime Video?",
    "I have an issue with my Amazon account, login, or password.",
    "My Amazon Prime membership, subscription, or payment was charged incorrectly.",
    "Amazon gift card, payment method, or refund not credited.",
    "How do I contact Amazon customer service about my order problem?",
    "How do I sign out of my Amazon account on all devices?",
    "I want a refund for the order I paid for on my card.",
];

// Single tokens that alone mark a query as Amazon-related.
export const STRONG_KEYWORDS = new Set([
    "amazon", "order", "orders", "ordered", "ordering", "delivery", "deliver",
    "delivered", "delivery", "shipped", "shipment", "shipping", "ship",
    "package", "parcel", "courier", "refund", "return", "returned",
    "replacement", "exchange", "cancel", "cancelled", "canceled",
    "cancellation", "track", "tracking", "purchase", "purchased",
    "product", "products", "item", "items", "seller", "cart", "wishlist",
    "prime", "subscription", "subscribed", "subscribe", "movie", "movies",
    "film", "films", "series", "episode", "episodes", "season", "seasons",
    "subtitles", "subtitle", "captions", "caption", "streaming", "stream",
    "watchlist", "trailer", "invoice", "payment", "echo", "alexa",
    "kindle", "aws", "signout", "signin", "login", "dispatch", "dispatched",
    "despatch", "despatched",
]);

// Multi-word phrases that alone mark a query as Amazon-related.
export const STRONG_PHRASES = [
    "prime video", "prime membership", "gift card", "customer service",
    "customer care", "fire tv", "fire stick", "sign out", "log out",
    "log in", "sign in", "add to cart", "out of stock", "in stock",
    "money back", "cash on delivery",
    // shopping intent
    "do you have", "do you sell", "need a new", "looking for a",
    "looking to buy", "want to buy", "want to purchase", "want to order",
    "where can i buy", "where can i get", "what options do you have",
    "need to buy", "i'd like to buy", "i would like to buy",
];

// Tokens that only weakly hint at Amazon context (need >= 2 to pass).
export const WEAK_KEYWORDS = new Set([
    "help", "support", "issue", "problem", "complaint", "escalate",
    "escalation", "query", "account", "charge", "charged", "discount",
    "price", "offer", "deal", "review", "rating", "warranty", "defective",
    "damaged", "broken", "buy", "bought", "buying", "download",
    "downloaded", "watch", "recommend", "suggest", "money", "delayed",
    "late", "address", "shoe", "gift", "size", "colour", "color",
    "recommendation", "app", "apps", "email", "customer", "refund",
    "voucher", "coupon", "membership", "subscribe", "driver",
]);

export const WEAK_PHRASES = ["not working", "money back", "customer care"];

/**
 * Minimum cosine similarity to the closest reference query for a
 * keyword-less query to count as in scope (calibrated in calibrate_threshold.ts).
 */
export const SEMANTIC_THRESHOLD = 0.74;


// Prompt injection & jailbreak detection (deterministic)

export const JAILBREAK_PATTERNS: RegExp[] = [
    // direct instruction override
    /ignore\s+(all|any|the|your|previous|prior|above|earlier|initial|original)\s.{0,40}(instruction|prompt|rule|guideline|direction|constraint|polic)/i,
    /disregard\s+(all|any|the|your|previous|prior|above|earlier|initial|original)\s.{0,40}(instruction|prompt|rule|guideline|direction|constraint|polic)/i,
    /forget\s+(all|any|the|your|previous|prior|above|earlier)\s.{0,40}(instruction|prompt|rule|guideline|context)/i,
    /disobey\s+(the\s+)?(rules|instructions|guidelines)/i,
    // persona swaps
    /\bDAN\b(?=[\s,.:;])/,
    /do\s+anything\s+now/i,
    /act\s+as\s+(a|an|my)?\s?(different|another|unfiltered|unrestricted|jailbroken|evil|hacker|anonymous)/i,
    /pretend\s+(to\s+be|you\s+are|you're)\s+(a|an|my)?\s?(different|another|unfiltered|unrestricted|jailbroken|evil|hacker|developer|freedom)/i,
    /you\s+are\s+(now|no\s+longer|free)\s+(an?\s)?(amazon|assistant|bot|ai|model|agent|unrestricted)/i,
    /you\s+have\s+been\s+(released|freed|liberated|unshackled)/i,
    // mode toggles
    /developer\s+mode|god\s+mode|opposite\s+mode|chaos\s+mode|unfiltered\s+mode/i,
    /(enter|activate|enable|switch\s+to|begin|start)\s.{0,20}(developer|unfiltered|unrestricted|jailbreak|dan)\s?mode/i,
    // system prompt extraction
    /system\s+prompt|initial\s+prompt|hidden\s+(instructions|prompt)|secret\s+(instructions|prompt)/i,
    /(show|reveal|print|display|give|tell|leak|expose|output|repeat)\s.{0,30}(your|the)\s.{0,10}(system|initial|original|developer|full|hidden|secret)\s?(prompt|instructions?|message|rules?|config)/i,
    /repeat\s.{0,40}(words?|instructions?|prompt|text|message|rules?|request).{0,30}(above|before|prior|verbatim|word\s+for\s+word|starting\s+with)/i,
    /print\s.{0,30}(verbatim|word\s+for\s+word|all\s+text|everything)\s.{0,20}(above|before|prior)/i,
    /what\s+(are|is)\s+your\s+(instructions|rules|guardrails|restrictions|guidelines|configuration)/i,
    /tell\s+me\s+your\s+(rules|instructions|guardrails|prompt|configuration)/i,
    // restriction removal
    /\bunrestrict(ed)?\b|\bunfiltered\b|\bguardrails?\s+off\b|\bno\s+(rules|restrictions|guardrails|filters|censorship|limits)\b/i,
    /pretend.{0,30}(there\s+are|there's)\s+no\s+(rules|restrictions|guardrails|filters|policies)/i,
    /(bypass|override|disable|turn\s+off|deactivate|circumvent|remove)\s.{0,30}(filter|guardrail|restriction|safety|policy|policies|censorship|security|limit)/i,
    /ignore\s+(all\s+)?safety/i,
    /act\s+(outside|above|beyond)\s+(the\s+)?rules/i,
    /no\s+longer\s+bound\s+by/i,
    /free\s+to\s+(do|say|answer|respond|discuss)\s+(anything|everything|any)\b/i,
    // compliance pressure
    /you\s+(must|have\s+to|are\s+required\s+to|will)\s+comply/i,
    /mandatory\s+compliance/i,
    // known leak-probe artifacts
    /(krishna.*giraffe|giraffe.*krishna)/i,
    /\bjailbreak(ing)?\s?(mode)?\b/i,
];

export const INJECTION_MARKERS: RegExp[] = [
    /```/,                        // fenced code block payload
    /^~~~[a-z]*\s*$/m,            // alternate fenced block
    /^\s*(system|assistant|developer)\s*:\s*/im, // chat role spoofing
    /<\|?(im_start|im_end|system|endoftext|bos|eos)\|?>/i, // special tokens
    /\{\{[^}]*\}\}|\{%[^%]*%\}/,  // template placeholders
    /<script\b/i,                 // html/script payload
];


// Greeting / chit-chat allowance (harmless — passed through to the agent)

export const GREETING_RE = /^\s*(hi|hii+|hello+|hey+|yo|good\s+(morning|afternoon|evening|day)|greetings|thanks?|thank\s+you|ok|okay|great|cool|nice)\s*[\s!.,]*$/i;

// Tokens allowed in a pure greeting/chit-chat message (each token of the
// message must be one of these for the message to count as a greeting)
export const CHAT_TOKENS = new Set([
    "hi", "hii", "hello", "hey", "yo", "good", "morning", "afternoon",
    "evening", "day", "greetings", "thanks", "thank", "you", "ok",
    "okay", "great", "cool", "nice", "there", "whats", "what's", "up",
    "how", "how's", "is", "it", "going", "a", "the", "mrng",
]);