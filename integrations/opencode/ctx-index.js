/**
 * ctx-index: when an OpenCode session goes idle, ask the agents-context daemon to re-index
 * OpenCode sessions so they become searchable through the `ctx` MCP server.
 *
 * Installed by copying into ~/.config/opencode/plugins/. The daemon debounces and coalesces
 * requests and runs a periodic sync anyway, so a missed or failed request only delays indexing.
 *
 * Only the plugin itself is exported: OpenCode invokes every exported function as a plugin.
 */
const DAEMON = process.env.CTX_EMBED_URL ?? "http://127.0.0.1:7997"

export const CtxIndex = async (_input, injectedFetch) => {
  // Tests inject a fetch; if OpenCode ever passes a second argument, it must not replace fetch.
  const fetchImpl = typeof injectedFetch === "function" ? injectedFetch : fetch
  let last = 0
  const request = async () => {
    // Both hook styles below can fire for the same idle event.
    const now = Date.now()
    if (now - last < 5000) return false
    last = now
    try {
      await fetchImpl(`${DAEMON}/index`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ source: "opencode" }),
        signal: AbortSignal.timeout(2000),
      })
      return true
    } catch {
      // Daemon down or restarting — never break the session over indexing.
      return false
    }
  }
  return {
    "session.idle": request,
    event: async ({ event }) => {
      if (event?.type === "session.idle") return request()
    },
  }
}
