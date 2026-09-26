// node --test integrations/opencode/ctx-index.test.mjs
import { test } from "node:test"
import assert from "node:assert/strict"
import { createServer } from "node:http"

import * as plugin from "./ctx-index.js"

const recorder = () => {
  const calls = []
  const fetchImpl = async (url, init) => {
    calls.push({ url, init })
    return new Response(null, { status: 202 })
  }
  return { calls, fetchImpl }
}

test("exports only the plugin function", () => {
  assert.deepEqual(Object.keys(plugin), ["CtxIndex"])
})

test("session.idle posts an opencode index request", async () => {
  const { calls, fetchImpl } = recorder()
  const hooks = await plugin.CtxIndex({}, fetchImpl)
  assert.equal(await hooks["session.idle"]({ sessionID: "s1" }), true)
  assert.equal(calls.length, 1)
  assert.match(calls[0].url, /\/index$/)
  assert.equal(calls[0].init.method, "POST")
  assert.deepEqual(JSON.parse(calls[0].init.body), { source: "opencode" })
})

test("both hook styles for one idle event send a single request", async () => {
  const { calls, fetchImpl } = recorder()
  const hooks = await plugin.CtxIndex({}, fetchImpl)
  await hooks["session.idle"]({ sessionID: "s1" })
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s1" } } })
  assert.equal(calls.length, 1)
})

test("unrelated events are ignored", async () => {
  const { calls, fetchImpl } = recorder()
  const hooks = await plugin.CtxIndex({}, fetchImpl)
  await hooks.event({ event: { type: "message.updated" } })
  assert.equal(calls.length, 0)
})

test("a non-function second argument falls back to global fetch", async () => {
  const original = globalThis.fetch
  let used = false
  globalThis.fetch = async () => {
    used = true
    return new Response(null, { status: 202 })
  }
  try {
    const hooks = await plugin.CtxIndex({}, { directory: "D:/dev/x" })
    assert.equal(await hooks["session.idle"]({}), true)
    assert.equal(used, true)
  } finally {
    globalThis.fetch = original
  }
})

test("a daemon that is down never throws", async () => {
  const hooks = await plugin.CtxIndex({}, async () => {
    throw new TypeError("fetch failed")
  })
  assert.equal(await hooks["session.idle"]({}), false)
})

test("reaches a real HTTP endpoint", async () => {
  let body
  const server = createServer((req, res) => {
    let data = ""
    req.on("data", (c) => (data += c))
    req.on("end", () => {
      body = JSON.parse(data)
      res.writeHead(202).end()
    })
  })
  await new Promise((r) => server.listen(0, "127.0.0.1", r))
  const { port } = server.address()
  const hooks = await plugin.CtxIndex({}, (url, init) => fetch(url.replace(/http:\/\/[^/]+/, `http://127.0.0.1:${port}`), init))
  assert.equal(await hooks["session.idle"]({}), true)
  server.close()
  assert.deepEqual(body, { source: "opencode" })
})
