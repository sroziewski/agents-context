//! Auto-inject end to end: the hook client posts a Claude Code payload to a real daemon router,
//! which searches real summaries in Qdrant.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{ALPHA, BETA, DIM, Env, Fixture, indexed};
use ctx::embed_server::{SharedEmbedder, daemon_router};
use ctx::inject::{InjectConfig, Injector};
use ctx::rank::Recency;
use embed::{Embedder, HashEmbedder};
use ingest::WordCounter;
use serde_json::{Value, json};

const FIXTURES: &[Fixture] = &[
    Fixture { id: "gpu-hang", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["directml gpu hang timeout batch size fix", "thanks"] },
    Fixture { id: "css", cwd: ALPHA, at: "2026-09-21T10:00:00Z", prompts: &["style the login button colours", "thanks"] },
    Fixture { id: "beta-gpu", cwd: BETA, at: "2026-09-22T10:00:00Z", prompts: &["directml gpu hang timeout batch size fix", "thanks"] },
    Fixture { id: "current", cwd: ALPHA, at: "2026-09-23T10:00:00Z", prompts: &["directml gpu hang timeout batch size fix again", "thanks"] },
];

fn injector(e: &Env, min_score: f32) -> Injector {
    Injector::new(
        e.store.clone(),
        None,
        Arc::new(Mutex::new(Box::new(HashEmbedder::new(DIM)) as Box<dyn Embedder>)),
        Arc::new(WordCounter),
        Recency::OFF,
        InjectConfig { enabled: true, teams: false, min_score, max_tokens: 1500 },
        e.cfg.redact_file.clone(),
    )
}

async fn serve(injector: Injector) -> String {
    let app = daemon_router(SharedEmbedder::new(Box::new(HashEmbedder::new(DIM))), "hash", None, Some(Arc::new(injector)));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

async fn hook(url: &str, session: &str, cwd: &str, prompt: &str) -> Option<String> {
    let payload = json!({"session_id": session, "cwd": cwd, "prompt": prompt, "hook_event_name": "UserPromptSubmit"}).to_string();
    let url = url.to_owned();
    let out = tokio::task::spawn_blocking(move || ctx::hook::inject(&url, &payload, Duration::from_secs(5), |_| None)).await.unwrap()?;
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit");
    Some(v["hookSpecificOutput"]["additionalContext"].as_str().unwrap().to_owned())
}

#[tokio::test(flavor = "multi_thread")]
async fn injects_this_projects_matching_summary_once_per_conversation() {
    let e = indexed(FIXTURES).await;
    let url = serve(injector(&e, 0.5)).await;
    let prompt = "how did we fix the directml gpu hang timeout";

    let ctx = hook(&url, "current", ALPHA, prompt).await.expect("a matching summary is injected");
    assert!(ctx.contains("session gpu-hang · 2026-09-20"), "{ctx}");
    assert!(!ctx.contains("beta-gpu"), "another project's session leaked: {ctx}");
    assert!(!ctx.contains("session current"), "the conversation's own summary is not injected: {ctx}");
    assert!(!ctx.contains("session css"), "unrelated summary injected: {ctx}");

    assert_eq!(hook(&url, "current", ALPHA, prompt).await, None, "not repeated within one conversation");
    assert!(hook(&url, "other-conversation", ALPHA, prompt).await.is_some(), "a new conversation gets it again");
}

#[tokio::test(flavor = "multi_thread")]
async fn stays_silent_below_the_threshold_and_for_trivial_prompts() {
    let e = indexed(FIXTURES).await;
    let url = serve(injector(&e, 0.5)).await;
    assert_eq!(hook(&url, "s", ALPHA, "please rename variable foo into bar within utils").await, None);
    assert_eq!(hook(&url, "s", ALPHA, "ok go").await, None);
    assert_eq!(hook(&url, "s", r"Z:\work\gamma", "how did we fix the directml gpu hang timeout").await, None, "project without sessions");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_busy_embedder_skips_the_prompt_without_marking_anything_injected() {
    let e = indexed(FIXTURES).await;
    let inj = injector(&e, 0.5);
    let req = ctx::inject::ContextRequest { prompt: "how did we fix the directml gpu hang timeout".into(), cwd: ALPHA.into(), session_id: "conv".into() };

    let embedder = inj.embedder.clone();
    let (locked_tx, locked_rx) = std::sync::mpsc::channel();
    let holder = std::thread::spawn(move || {
        let _guard = embedder.lock().unwrap();
        locked_tx.send(()).unwrap();
        std::thread::sleep(Duration::from_millis(2500));
    });
    locked_rx.recv().unwrap();
    let start = std::time::Instant::now();
    let r = inj.context(&req).await.unwrap();
    assert!(r.context.is_empty(), "no context while indexing holds the model");
    assert!(start.elapsed() < Duration::from_millis(2300), "gave up within the hook's budget: {:?}", start.elapsed());
    holder.join().unwrap();

    let r = inj.context(&req).await.unwrap();
    assert!(r.sessions.contains(&"gpu-hang".to_string()), "the skipped prompt did not mark it as already injected: {:?}", r.sessions);
}

#[tokio::test(flavor = "multi_thread")]
async fn already_injected_sessions_make_room_for_the_next_match() {
    let fixtures = &[
        Fixture { id: "gpu-1", cwd: ALPHA, at: "2026-09-20T10:00:00Z", prompts: &["directml gpu hang timeout batch size fix", "thanks"] },
        Fixture { id: "gpu-2", cwd: ALPHA, at: "2026-09-21T10:00:00Z", prompts: &["directml gpu hang timeout batch size fix one", "thanks"] },
        Fixture { id: "gpu-3", cwd: ALPHA, at: "2026-09-22T10:00:00Z", prompts: &["directml gpu hang timeout batch size fix two", "thanks"] },
        Fixture { id: "gpu-4", cwd: ALPHA, at: "2026-09-23T10:00:00Z", prompts: &["directml gpu hang timeout batch size fix three", "thanks"] },
    ];
    let e = indexed(fixtures).await;
    let inj = injector(&e, 0.3);
    let req = ctx::inject::ContextRequest { prompt: "directml gpu hang timeout batch size fix".into(), cwd: ALPHA.into(), session_id: "conv".into() };
    let first = inj.context(&req).await.unwrap().sessions;
    assert_eq!(first.len(), 3);
    let second = inj.context(&req).await.unwrap().sessions;
    assert_eq!(second.len(), 1, "the fourth match is offered instead of nothing: {second:?}");
    assert!(!first.contains(&second[0]));
}

#[tokio::test(flavor = "multi_thread")]
async fn preview_and_hook_agree_and_respect_the_token_cap() {
    let e = indexed(FIXTURES).await;
    let mut small = injector(&e, 0.0);
    small.config.max_tokens = 60;
    let req = ctx::inject::ContextRequest { prompt: "directml gpu hang timeout batch size fix".into(), cwd: ALPHA.into(), session_id: String::new() };
    let r = small.context(&req).await.unwrap();
    assert!(!r.context.is_empty());
    assert!(ingest::TokenCounter::count(&WordCounter, &r.context) <= 60, "{}", r.context);
}
