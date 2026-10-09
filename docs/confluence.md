# Confluence in ctx

ctx indexes the Confluence spaces you map to your projects, so `search_confluence`
(MCP) and `ctx confluence search` find documentation, runbooks and specs by meaning.

## Setup

1. Create a Personal Access Token in Confluence: avatar → Settings → Personal Access
   Tokens. Read access is enough.
2. Add it to the ctx config `.env` (`%APPDATA%\agents-context\.env`):

   ```
   CONFLUENCE_URL=https://confluence.example.org
   CONFLUENCE_TOKEN=<the token>
   ```

3. Find the spaces for each project:

   ```
   ctx confluence spaces sedn
   ctx confluence spaces helpdesk
   ctx confluence spaces ai4sd
   ```

4. Map them in `%APPDATA%\agents-context\confluence-projects.json`. It has the same
   shape as `jira-projects.json`, project directory → space keys:

   ```json
   { "sedn26": ["SEDN"], "helpdesk": ["HD"], "atlassian-gatewa-ai4sd": ["AI4SD"] }
   ```

5. Restart the daemon (`scripts/install.ps1`) and index once:

   ```
   ctx index --source confluence
   ```

After that the daemon re-indexes at 10:30 and 17:30 (`CTX_CONFLUENCE_INDEX_TIMES`).
`ctx confluence mapped` shows what the current directory maps to.

## What a run does

- It lists every current page of each mapped space with its **version number only**,
  without bodies.
- A page whose version has not changed since the last run is skipped: it is neither
  fetched nor embedded again.
- A new or edited page is fetched and re-chunked. Only chunks whose text changed are
  re-embedded.
- A page missing from a complete listing is removed from the index. That covers
  deleted pages and pages moved to another space; a moved page is re-indexed under
  its new space when that space is mapped too.
- Every chunk of an edited page gets the new editor, date and version, including
  chunks whose text did not change and so were not re-embedded.
- If a space's listing fails, nothing in that space is removed, and the run moves on
  to the other spaces.
- A space taken out of the mapping is removed from the index on the next run.

Text is redacted with the same rules as everything else in ctx. Each chunk starts with
`[SPACE · Title] breadcrumb`, so an excerpt says where it comes from.

| Setting | Default |
|---|---|
| `CONFLUENCE_URL`, `CONFLUENCE_TOKEN` | unset: Confluence off |
| `CONFLUENCE_PROJECTS` | `confluence-projects.json` in the config dir |
| `CTX_CONFLUENCE_COLLECTION` | `confluence` |
| `CTX_CONFLUENCE_INDEX_TIMES` | `10:30,17:30` |
