# How `ctx` is invoked from Claude Code

`ctx` is touched on almost every turn, but it "finds through past sessions" on two
very different schedules. This doc explains exactly when each path runs, so you
know what is happening while you work.

There are **three mechanisms**, wired independently:

| Mechanism | What it does | When it runs | Visible to you? |
|---|---|---|---|
| `--inject` hook | **Retrieval** — pulls relevant past sessions into your prompt | **Every prompt** (injects only if something scores high enough) | No — silent prefix |
| index hooks | **Indexing** — writes the new conversation into Qdrant | **Every turn** + session events | No |
| MCP search tools | **Explicit retrieval** — `search_sessions`, `get_session`, … | **Only when the model decides, or you ask** | Yes — a tool call |

---

## 1. Auto-inject — retrieval on *every* prompt

Registered as a `UserPromptSubmit` hook in `~/.claude/settings.json`:

```json
"UserPromptSubmit": [
  { "hooks": [ { "type": "command",
      "command": "…\\agents-context\\bin\\ctx.exe",
      "args": ["hook", "--inject"], "timeout": 5 } ] }
]
```

On **every message you send**, `ctx hook --inject`:

1. sends your prompt to the daemon's `/context` endpoint,
2. embeds it (bge-m3) and searches Qdrant within the **current project**,
3. keeps hits scoring above `CTX_INJECT_MIN_SCORE`, up to `CTX_INJECT_MAX_TOKENS`,
4. prepends them to your prompt as a `<past-sessions>` block **before the model reads it**.

So retrieval is *attempted* every turn, but only *injects* when something is
actually relevant. It is silent — you never see the block, and it is stripped
before re-indexing so it cannot feed back on itself.

**Gate:** this only happens when injection is enabled. It is **off by default**;
turn it on with `CTX_INJECT=on` (or per-client `CTX_INJECT_CLAUDE` /
`CTX_INJECT_OPENCODE`).

### Example

> **You:** "how did we handle the Jira delta watermark again?"
> → high-scoring chunk from a past DEV-39 session is injected; the model answers
> with that context already in hand, without calling any tool.

> **You:** "fix this typo"
> → nothing scores above the threshold; nothing is injected. The hook still ran,
> it was just a no-op.

---

## 2. Indexing — every turn (this is *writing*, not finding)

Three hooks keep the index current:

```json
"Stop":       ["hook", "--source", "claude"],   // after each assistant reply
"SessionEnd": ["hook", "--source", "claude"],   // when a session closes
"PreCompact": ["hook", "--source", "claude"]     // before context is compacted
```

These fire `ctx hook`, which asks the daemon to index new/grown sessions into
Qdrant. This is what *makes* a conversation searchable later — it is ingestion,
not retrieval. Indexing is incremental (fingerprint-based), so unchanged sessions
are skipped.

---

## 3. MCP search tools — only *sometimes*

The `ctx` MCP server exposes tools the model calls **agentically**:

- `search_sessions` — semantic search over past sessions
- `get_session` — pull one full session
- `list_sessions`, `list_projects` — browse what is indexed
- `deep_search_sessions` — reranked, higher-quality search (opt-in, ~seconds)

These do **not** run every turn. The model calls one only when it judges that
past sessions would help, or when **you explicitly ask**. Each call is a visible
tool use.

### Example

> **You:** "search my past sessions for why we picked dense over hybrid search"
> → the model calls `search_sessions("dense vs hybrid ranking decision")`,
> reads the hits, and cites the session id + date.

> **You (CLI):** `ctx search --rerank "reranker latency decision"`
> → runs the deep, reranked search from the terminal.

---

## Configuration reference

All in `%APPDATA%\agents-context\.env` (quote Windows paths with single quotes).

| Variable | Default | Effect |
|---|---|---|
| `CTX_INJECT` | `off` | Master switch for auto-inject |
| `CTX_INJECT_CLAUDE` | falls back to `CTX_INJECT` | Inject for Claude Code |
| `CTX_INJECT_OPENCODE` | falls back to `CTX_INJECT` | Inject for OpenCode |
| `CTX_INJECT_TEAMS` | `off` | Also inject relevant Teams threads |
| `CTX_INJECT_CHATS` | `off` | Let those Teams threads include your private chats (needs `CTX_INJECT_TEAMS`) |
| `CTX_INJECT_MIN_SCORE` | code default | Minimum score for a chunk to be injected |
| `CTX_INJECT_MAX_TOKENS` | code default | Token budget for the injected block |

The `.env` is re-read by the daemon each run, so `ctx redact add` and inject
setting changes apply without a restart. Injection scope stays within the current
project unless a query explicitly asks for `all` or another project.

---

## How to see it working

```sh
# Watch retrieval + indexing as you work:
tail -f "%LOCALAPPDATA%\agents-context\embed-server.log"

# What the daemon would inject for a query, and index state:
curl -s http://127.0.0.1:7997/index/status
curl -s http://127.0.0.1:7997/health
```

A quick mental model: **inject and indexing are automatic and silent every turn;
the MCP search tools are deliberate and visible only when needed.**
