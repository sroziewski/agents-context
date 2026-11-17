# The ctx status line

A one-line bar at the bottom of Claude Code that shows how full the conversation's
context is, and tells you when to `/clear`. It costs no tokens: Claude Code runs
`ctx statusline` locally on every update and never sends its output to the model.

```
Opus 5.5 · ctx ▰▰▰▰▰▰▱▱▱▱ 63%
Opus 5.5 · ctx ▰▰▰▰▰▰▰▰▱▱ 81% · /clear → ctx hands over · ctx daemon down
```

## Reading it

| Part | Meaning | What to do |
|---|---|---|
| `Opus 5.5` | The model of this session | — |
| `ctx ▰▰▰▱▱▱▱▱▱▱ 34%` (plain) | Context used, out of the model's full window | Nothing; keep working |
| Same, **yellow** (≥ 50%) | The conversation is getting long: every turn resends all of it | Finish the current task, then `/clear` |
| Same, **red** (≥ 75%), with `/clear → ctx hands over` | Close to auto-compact; long contexts also get less precise | `/clear` now. ctx hands the session over (see below) |
| `ctx daemon down` (red) | The ctx daemon is not answering: no indexing, no auto-inject, no search | Run `scripts/install.ps1`, or start the daemon via the Startup launcher |
| `Teams stale` (dim) | `teams.db` hasn't been synced for `CTX_TEAMS_STALE_HOURS` (3 h) of working time: weekdays within `CTX_TEAMS_SYNC_HOURS` (8–18). Nights and weekends don't count, so it means a scheduled sync really failed | Check the end of `D:\dev\my\teams\data\sync.log`; run `npm run login` there only if it says you are logged out |

The usage part is missing until the first reply of a session (Claude Code has no
number before then) and right after `/compact`.

## Why `/clear` is safe: the handoff

`/clear` throws away the conversation, which is what keeps cost and context small,
but it used to lose what the session knew. With ctx it doesn't: the new
conversation starts with a `<previous-session>` note holding

- your last 5 prompts,
- the files edited,
- the end of the last answer (usually the outcome and any open question),
- ctx's summary of the session, when one has been written (marked "as of turn N of
  M" when it's behind).

So the workflow is: **one task per conversation; when the bar turns yellow and the
task is done, or when it turns red, type `/clear` and carry on.** If you need more
than the note gives, ask for the old session: the note names its id for
`get_session`.

After `/compact` there is no note: Claude Code's own compaction summary is already
in the conversation. `/compact focus on <topic>` is the alternative when you want to
stay in the same conversation and keep a specific thread.

## Setup

Both pieces live in `~/.claude/settings.json` (already set up on this machine):

```json
"statusLine": {
  "type": "command",
  "command": "C:/Users/you/AppData/Local/agents-context/bin/ctx.exe statusline",
  "padding": 0
},
"hooks": {
  "SessionStart": [
    { "matcher": "clear",
      "hooks": [{ "type": "command",
                  "command": "C:\\Users\\you\\AppData\\Local\\agents-context\\bin\\ctx.exe",
                  "args": ["hook", "--source", "claude"], "timeout": 5 }] }
  ]
}
```

Use forward slashes in `statusLine.command`: it is a shell command line, and a shell
such as Git Bash would eat the backslashes. The `SessionEnd` hook that ctx already
had records which session was cleared; nothing else is needed.

## Settings

In `%APPDATA%\agents-context\.env`; read on every update, so no restart is needed.

| Variable | Default | Effect |
|---|---|---|
| `CTX_STATUS_WARN` | `50` | Usage (%) from which the bar is yellow |
| `CTX_STATUS_CRIT` | `75` | Usage (%) from which it is red, with the `/clear` hint |
| `CTX_HANDOFF` | `on` | `off` stops the `/clear` note; the red hint then says `/clear or /compact` |
| `CTX_TEAMS_STALE_HOURS` | `3` | Working hours without a sync before `Teams stale` appears (and the daemon re-runs the scraper) |

Related Claude Code settings, if you want auto-compact to run earlier rather than
relying on the nudge: `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` (1–100, can only lower the
threshold) or `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (in tokens). Note that the percentage
shown is measured against the model's full window, so with a smaller auto-compact
window compaction happens before the bar reaches 100%.

## Checking it by hand

```bash
echo '{"model":{"display_name":"Opus 5.5"},"context_window":{"used_percentage":81}}' \
  | C:/Users/you/AppData/Local/agents-context/bin/ctx.exe statusline
```

prints the line with its colours. Bad or empty input never fails; it just prints
less.

## Troubleshooting

- **No status line at all**: check the `statusLine` entry and that the path uses
  forward slashes; run the command above to see that it prints.
- **No note after `/clear`**: check the `SessionStart` hook (matcher `clear`) and
  that `CTX_HANDOFF` isn't `off`. The note needs a previous session with at least one
  prompt in the same project.
- **`ctx daemon down` won't go away**: `scripts/install.ps1` rebuilds, installs and
  restarts the daemon, and fails loudly if it can't start; see
  `%LOCALAPPDATA%\agents-context\embed-server.log`.
