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
      "args": ["hook", "--inject"], "timeout": 10 } ] }
]
```

The hook gives up on its own about 3 s after `main` starts (`INJECT_BUDGET`). It
skips the lookup when less than 2 s of that is left, because the daemon may wait
1.5 s for the embedder, and a hook that leaves early would still have those
sessions marked as already shown. The `timeout: 10` is a safety margin
for a machine that stalls before the hook even runs, for example under memory
pressure or on the first launch of a freshly installed `ctx.exe`. With `5` such a
stall gets the hook killed, which means a warning and no context.

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
- `deep_search_sessions` — reranked search; offered only with `CTX_RERANK=on`, see below

A query that names something by its exact spelling (a file name like `eval.rs`, a
Jira key like `SEDN26-1910`, a function like `boundObjectId`) is also matched
word for word (BM25) and returns turns only, without session summaries in front.
Other queries are searched by meaning, as before (`CTX_LEXICAL`, DEV-61/62).

These do **not** run every turn. The model calls one only when it judges that
past sessions would help, or when **you explicitly ask**. Each call is a visible
tool use.

### Example

> **You:** "search my past sessions for why we picked dense over hybrid search"
> → the model calls `search_sessions("dense vs hybrid ranking decision")`,
> reads the hits, and cites the session id + date.

> **You (CLI):** `ctx search --rerank "reranker latency decision"`
> → runs the deep, reranked search from the terminal (with `CTX_RERANK=on`).

### Deep search (reranker) — off by default

Deep search takes the candidates the normal search found and re-sorts them with a
cross-encoder (`jina-v2`, on CPU). It is **off** unless `CTX_RERANK=on` is set for
both the daemon and the MCP server (the shared `.env`). When off:

- the daemon never loads the reranker model (~1.1 GB on disk, several GB of RAM
  while it runs);
- the MCP server does not list `deep_search_sessions`, so agents cannot call it;
- `ctx search --rerank` says deep search is off and prints the normal results.

Why off: measured on the live index (DEV-63, 2026-10-09, 8 CPU threads):

| Test set | Candidates | R@1 normal → reranked | Added time, p95 |
|---|---|---|---|
| Natural questions (60) | 10 | 0.83 → 0.88 | 7.2 s |
| Natural questions (60) | 20 | 0.83 → 0.88 | 14.4 s |
| Bare identifier (30) | 10 | 0.90 → 0.97 | 6.9 s |
| Identifier in a sentence (30) | 10 | 0.87 → 0.90 | 7.1 s |

The bar was p95 ≤ 800 ms at 30 candidates plus a quality gain outside noise. It
costs about 0.6 s per candidate, and the gain on natural questions is within noise.
Turn it on if you are happy to wait several seconds for an occasional harder search.

Settings (`.env`): `CTX_RERANK=on`, `CTX_RERANK_MODEL` (default `jina-v2`),
`CTX_RERANK_POOL` (candidates, default 30; 10 halves the wait), `CTX_RERANK_THREADS`.
Restart the daemon and the agent after changing them. To re-measure:
`ctx eval --rerank-model jina-v2 --rerank-pool 10 --rerank-threads 8`, and the same
with `--target identifiers --samples 30`.

---

## 4. Keeping the context small — `/clear` handoff and the status line

Claude Code cannot be cleared or compacted from outside: no hook, SDK call or other
process can run `/clear` or `/compact`. The cheap way to keep context small is to
`/clear` often, so ctx makes clearing lose nothing and tells you when to do it.

**Handoff after `/clear`.** On `/clear`, Claude Code ends the old session
(`SessionEnd`, reason `clear`) and starts a new one (`SessionStart`, source `clear`):

```json
"SessionStart": [{"matcher": "clear", "hooks": [{"command": "…\\ctx.exe", "args": ["hook", "--source", "claude"]}]}]
```

The `SessionEnd` hook records which transcript was cleared; the `SessionStart` hook
adds a `<previous-session>` block to the new conversation. The block holds the last
prompts, the files edited, the end of the last answer, and the ctx summary when
one has been written yet. It is built from the transcript, so it is immediate and
uses no LLM. `CTX_HANDOFF=off` turns it off. After `/compact` nothing is added:
Claude Code's own compaction summary is already in the conversation.

**Status line.** `ctx statusline` shows the model, how full the context is, and
health markers:

```
Opus 5.5 · ctx ▰▰▰▰▰▰▰▰▱▱ 81% · /clear → ctx hands over · ctx daemon down
```

The usage turns yellow at `CTX_STATUS_WARN` and red, with the `/clear` hint, at
`CTX_STATUS_CRIT`. "ctx daemon down" and a dim "Teams stale" appear only when they
apply. Register it in `~/.claude/settings.json`:

```json
"statusLine": {"type": "command", "command": "…\\ctx.exe statusline"}
```

How to read it and use it day to day: [status-line.md](status-line.md).

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
| `CTX_HANDOFF` | `on` | After `/clear`, hand the cleared session over to the new one |
| `CTX_STATUS_WARN` | `50` | Status line: context usage (%) shown in yellow |
| `CTX_STATUS_CRIT` | `75` | Status line: context usage (%) shown in red, with the `/clear` hint |
| `CTX_DATA_DIR` | `%LOCALAPPDATA%\agents-context` | Local working data (state DB, handoff records, …) |

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
