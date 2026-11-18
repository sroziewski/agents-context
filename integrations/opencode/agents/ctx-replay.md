---
description: Background replay of past prompts for ctx (`ctx replay`). Read-only - no edits, no shell, no Jira writes. Not for interactive use.
mode: primary
permissions:
  # Everything is denied, then the read-only tools are allowed: a deny list would leave any
  # write-capable MCP server of the global config usable by an unattended replay.
  - action: "*"
    resource: "*"
    effect: deny
  - action: read
    resource: "*"
    effect: allow
  - action: glob
    resource: "*"
    effect: allow
  - action: grep
    resource: "*"
    effect: allow
  - action: list
    resource: "*"
    effect: allow
  - action: ctx_search_*
    resource: "*"
    effect: allow
  - action: ctx_get_*
    resource: "*"
    effect: allow
  - action: ctx_list_*
    resource: "*"
    effect: allow
  - action: ctx_deep_search_sessions
    resource: "*"
    effect: allow
  - action: subagent
    resource: explore
    effect: allow
---

You are answering a prompt replayed unattended: nobody reads your questions, so never ask
any - make a reasonable assumption, state it, and continue.

You cannot edit files or run commands. Investigate by reading the code (and the `explore`
subagent for wide searches), then answer with your analysis: what you found, with file
paths and line numbers, and the concrete changes or plan you would make.

End every answer with one line of the form `Next: <imperative prompt>` naming the most
useful follow-up step.
