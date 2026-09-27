// node --test integrations/opencode/ctx-inject.test.mjs
import { test } from "node:test"
import assert from "node:assert/strict"
import { createServer } from "node:http"

import * as plugin from "./ctx-inject.js"

/** A fetch that returns the given /context response and records requests. */
const recorder = (context) => {
  const calls = []
  const fetchImpl = async (url, init) => {
    calls.push({ url, init })
    return new Response(JSON.stringify({ context, sessions: [], threads: [] }), { status: 200, headers: { "content-type": "application/json" } })
  }
  return { calls, fetchImpl }
}

const userParts = (text) => [{ type: "text", text }]

test("exports only the plugin function", () => {
  assert.deepEqual(Object.keys(plugin), ["CtxInject"])
})

test("appends exactly one synthetic part carrying the context", async () => {
  const { calls, fetchImpl } = recorder("<past-sessions>\ngist\n</past-sessions>")
  const hooks = await plugin.CtxInject({ worktree: "D:/dev/helpdesk" }, fetchImpl)
  const parts = userParts("how did we restore the prod database")
  const out = { message: { id: "msg_1", sessionID: "ses_1" }, parts }
  await hooks["chat.message"]({ sessionID: "ses_1", messageID: "msg_1" }, out)

  assert.equal(parts.length, 2)
  const added = parts[1]
  assert.equal(added.synthetic, true)
  assert.equal(added.type, "text")
  assert.match(added.text, /past-sessions/)
  assert.equal(added.sessionID, "ses_1")
  // request shape
  assert.equal(calls.length, 1)
  assert.match(calls[0].url, /\/context$/)
  const body = JSON.parse(calls[0].init.body)
  assert.equal(body.client, "opencode")
  assert.equal(body.cwd, "D:/dev/helpdesk")
  assert.equal(body.session_id, "ses_1")
  assert.equal(body.prompt, "how did we restore the prod database")
})

test("adds nothing for an empty context response", async () => {
  const { fetchImpl } = recorder("")
  const hooks = await plugin.CtxInject({}, fetchImpl)
  const parts = userParts("some question here")
  await hooks["chat.message"]({ sessionID: "s" }, { parts })
  assert.equal(parts.length, 1)
})

test("skips slash commands, empty prompts, and synthetic-only messages", async () => {
  const { calls, fetchImpl } = recorder("<past-sessions>x</past-sessions>")
  const hooks = await plugin.CtxInject({}, fetchImpl)
  for (const parts of [userParts("/compact keep notes"), userParts("   "), [{ type: "text", text: "nudge", synthetic: true }]]) {
    await hooks["chat.message"]({ sessionID: "s" }, { parts })
    assert.equal(parts.length, parts.filter((p) => !p.synthetic || p.text === "nudge").length) // unchanged
  }
  assert.equal(calls.length, 0, "no /context call for trivial prompts")
})

test("the prompt joins only the user's own text parts", async () => {
  const { calls, fetchImpl } = recorder("")
  const hooks = await plugin.CtxInject({}, fetchImpl)
  const parts = [
    { type: "text", text: "first line" },
    { type: "file", filename: "x.rs" },
    { type: "text", text: "second line", synthetic: true },
    { type: "text", text: "third line" },
  ]
  await hooks["chat.message"]({ sessionID: "s" }, { parts })
  assert.equal(JSON.parse(calls[0].init.body).prompt, "first line\nthird line")
})

test("never throws when the daemon is down or the response is bad", async () => {
  const down = await plugin.CtxInject({}, async () => {
    throw new TypeError("fetch failed")
  })
  const parts = userParts("a real question")
  await down["chat.message"]({ sessionID: "s" }, { parts })
  assert.equal(parts.length, 1)

  const bad = await plugin.CtxInject({}, async () => new Response("not json", { status: 200 }))
  const parts2 = userParts("another question")
  await bad["chat.message"]({ sessionID: "s" }, { parts2: undefined, parts: parts2 })
  assert.equal(parts2.length, 1)
})

test("a non-ok status is ignored", async () => {
  const hooks = await plugin.CtxInject({}, async () => new Response("nope", { status: 404 }))
  const parts = userParts("a real question")
  await hooks["chat.message"]({ sessionID: "s" }, { parts })
  assert.equal(parts.length, 1)
})

test("reaches a real HTTP endpoint and attaches its context", async () => {
  let body
  const server = createServer((req, res) => {
    let data = ""
    req.on("data", (c) => (data += c))
    req.on("end", () => {
      body = JSON.parse(data)
      res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ context: "<past-sessions>real</past-sessions>" }))
    })
  })
  await new Promise((r) => server.listen(0, "127.0.0.1", r))
  const { port } = server.address()
  const hooks = await plugin.CtxInject({ directory: "D:/dev/x" }, (url, init) => fetch(url.replace(/http:\/\/[^/]+/, `http://127.0.0.1:${port}`), init))
  const parts = userParts("real question")
  await hooks["chat.message"]({ sessionID: "ses_9" }, { parts })
  server.close()
  assert.equal(body.client, "opencode")
  assert.equal(parts.length, 2)
  assert.match(parts[1].text, /real/)
})
