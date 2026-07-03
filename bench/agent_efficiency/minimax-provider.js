// pi extension: register MiniMax as an Anthropic-compatible provider.
// API key is read from $MINIMAX_API_KEY at runtime — NEVER stored here.
export default function (pi) {
  pi.registerProvider("minimax", {
    name: "MiniMax", baseUrl: "https://api.minimax.io/anthropic",
    apiKey: "$MINIMAX_API_KEY", api: "anthropic-messages",
    models: [{ id: "MiniMax-M3", name: "MiniMax M3", reasoning: false,
      input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 1000000, maxTokens: 8192 }],
  });
}
