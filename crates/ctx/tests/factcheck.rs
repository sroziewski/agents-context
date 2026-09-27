//! Memory suggestions are checked against the real turns of the sessions they came from.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{ALPHA, DIM, Fixture, indexed};
use ctx::factcheck::{Repo, VerifyEnv, verify};
use ctx::memories::{Candidate, Verdict, acceptable};
use embed::{Embedder, HashEmbedder};
use ingest::{Redactor, Summarizer};

const FIXTURES: &[Fixture] = &[Fixture {
    id: "redact-sess",
    cwd: ALPHA,
    at: "2026-09-26T10:00:00Z",
    prompts: &["ctx redact add reads the secret from stdin so it never lands in shell history", "then rescan the index"],
}];

/// Records the prompt and answers with fixed verdicts.
struct FakeVerifier {
    prompt: Mutex<String>,
    answer: String,
}

impl Summarizer for FakeVerifier {
    fn id(&self) -> &str {
        "fake"
    }
    fn summarize(&self, prompt: &str, _timeout: Duration) -> Result<String, String> {
        *self.prompt.lock().unwrap() = prompt.to_owned();
        Ok(self.answer.clone())
    }
}

fn cand(name: &str, body: &str) -> Candidate {
    Candidate { name: name.into(), kind: "project".into(), description: "d".into(), body: body.into(), session: "redact-sess".into(), review: None }
}

#[tokio::test(flavor = "multi_thread")]
async fn wrong_usage_is_caught_from_the_turns_and_duplicates_are_flagged() {
    let e = indexed(FIXTURES).await;
    let memory_dir = e.tmp.path().join("memory");
    std::fs::create_dir_all(&memory_dir).unwrap();
    std::fs::write(memory_dir.join("feedback_no-secrets.md"), "---\nname: no-secrets\n---\n\nNever paste secret values into Linear or git.\n").unwrap();

    let verifier = FakeVerifier {
        prompt: Mutex::new(String::new()),
        answer: r#"[{"name": "redact-usage", "verdict": "revise", "reasons": ["the turns show it reads stdin"], "body": "Pipe the secret into ctx redact add (stdin), never as an argument."},
                    {"name": "dup", "verdict": "drop", "reasons": ["already a memory"]}]"#
            .into(),
    };
    let repo = Repo::from_parts(None, vec![], "#[arg(long)]\n    local: bool,".into());
    let env = VerifyEnv {
        store: &e.store,
        embedder: Arc::new(Mutex::new(Box::new(HashEmbedder::new(DIM)) as Box<dyn Embedder>)),
        verifier: &verifier,
        redactor: &Redactor::default(),
        project_key: &sources::project::key(ALPHA),
        repo: &repo,
        memory_dir: Some(&memory_dir),
        timeout: Duration::from_secs(5),
    };
    let out = verify(&env, vec![
        cand("redact-usage", "Run `ctx redact add <value>` then `ctx redact rescan --force-reread`."),
        cand("dup", "Never paste secret values into Linear or git."),
    ])
    .await
    .unwrap();

    let prompt = verifier.prompt.lock().unwrap().clone();
    assert!(prompt.contains("reads the secret from stdin"), "the verifier sees the real turn: {prompt}");
    assert!(prompt.contains("--force-reread appears neither in the repository nor in the checked sessions"), "{prompt}");
    assert!(prompt.contains("Never paste secret values into Linear or git."), "existing memories are included");

    let usage = &out[0];
    let review = usage.review.as_ref().unwrap();
    assert_eq!(review.verdict, Verdict::Revise);
    assert!(usage.body.contains("stdin"));
    assert!(review.warnings.is_empty(), "the corrected text no longer names the invented flag: {:?}", review.warnings);
    assert_eq!(review.sources, ["redact-sess"]);
    assert!(acceptable(usage).is_ok());

    let dup = out[1].review.as_ref().unwrap();
    assert_eq!(dup.verdict, Verdict::Drop);
    assert!(dup.warnings.iter().any(|w| w.contains("near-duplicate of existing memory feedback_no-secrets.md")), "{:?}", dup.warnings);
    assert!(acceptable(&out[1]).is_err());
}
