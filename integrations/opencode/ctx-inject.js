/**
 * ctx-inject: before OpenCode sends a user prompt to the model, ask the agents-context daemon
 * for relevant past-session context and attach it as a synthetic message part.
 *
 * The daemon decides whether OpenCode injection is on (CTX_INJECT_OPENCODE / CTX_INJECT), so
 * this plugin never reads config itself. The part is `synthetic: true`, which the ctx OpenCode
 * reader drops, so injected context is never re-indexed.
 *
 * Installed by copying into ~/.config/opencode/plugins/. Separate from ctx-index.js so either
 * can be removed on its own. Tested against @opencode-ai/plugin 1.15.x (the `chat.message` hook).
 */
const DAEMON = process.env.CTX_EMBED_URL ?? "http://127.0.0.1:7997"

export const CtxInject = async (input, injectedFetch) => {
  const fetchImpl = typeof injectedFetch === "function" ? injectedFetch : fetch
  // PluginInput carries the working directory; its project scopes the search server-side.
  const cwd = input?.worktree || input?.directory || ""

  return {
    "chat.message": async (hookInput, output) => {
      try {
        const parts = output?.parts
        if (!Array.isArray(parts)) return
        // The prompt is the user's own text parts; skip OpenCode's own synthetic nudges.
        const prompt = parts
          .filter((p) => p?.type === "text" && !p.synthetic && typeof p.text === "string")
          .map((p) => p.text)
          .join("\n")
          .trim()
        if (!prompt || prompt.startsWith("/")) return // slash commands and empty prompts: skip

        const sessionID = hookInput?.sessionID ?? output?.message?.sessionID ?? ""
        const res = await fetchImpl(`${DAEMON}/context`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ prompt, cwd, session_id: sessionID, client: "opencode" }),
          signal: AbortSignal.timeout(3000),
        })
        if (!res.ok) return
        const data = await res.json()
        if (!data?.context) return

        parts.push({
          id: `prt_ctxinject_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`,
          sessionID,
          messageID: hookInput?.messageID ?? output?.message?.id ?? "",
          type: "text",
          text: data.context,
          synthetic: true,
        })
        return true
      } catch {
        // A slow/absent daemon or an API shape change must never break the user's prompt.
        return false
      }
    },
  }
}
