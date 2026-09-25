//! Secret redaction, applied to every chunk before it is embedded or stored.
//!
//! Transcripts routinely contain tokens pasted into prompts, env dumps and connection strings.
//! Once embedded, a secret lives in the vector store and gets handed back to an LLM on retrieval,
//! so redaction happens before anything leaves the parser. Patterns favour recall: a redacted
//! non-secret costs a little search precision, a leaked secret costs a rotation.

use std::sync::LazyLock;

use regex::{Captures, Regex};

struct Rule {
    kind: &'static str,
    re: Regex,
    /// Capture group holding the secret; everything else in the match is kept.
    group: usize,
}

static RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    let rule = |kind, pattern: &str, group| Rule { kind, re: Regex::new(pattern).expect(kind), group };
    vec![
        rule("private-key", r"(-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----)", 1),
        rule("aws-key", r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b", 1),
        rule("github-token", r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b", 1),
        rule("gitlab-token", r"\b(glpat-[A-Za-z0-9_\-]{20,})", 1),
        rule("linear-key", r"\b(lin_(?:api|oauth)_[A-Za-z0-9]{32,})", 1),
        rule("slack-token", r"\b(xox[abposr]-[A-Za-z0-9\-]{10,})", 1),
        rule("api-key", r"\b(sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,})", 1),
        rule("google-key", r"\b(AIza[0-9A-Za-z_\-]{35})", 1),
        rule("jwt", r"\b(eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})", 1),
        rule("bearer", r"(?i)\bbearer\s+([A-Za-z0-9._~+/=\-]{16,})", 1),
        // scheme://user:password@host — keeps scheme, user and host so the context stays readable.
        rule("url-password", r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:([^\s@/]+)@", 1),
        // NAME=value / "name": "value" where the name says it is a secret. Values starting with
        // `$`, `{`, `<` or `%` are references (env vars, templates), not secrets.
        rule(
            "assignment",
            r#"(?i)\b[A-Z0-9_.\-]*(?:password|passwd|pwd|has[łl]o|secret|token|api[_\-]?key|access[_\-]?key|private[_\-]?key|client[_\-]?secret)[A-Z0-9_.\-]*["']?\s*[:=]\s*["']?([^\s"',;$\{<%][^\s"',;]{5,})"#,
            1,
        ),
    ]
});

/// Returns `text` with every secret replaced by `[REDACTED:<kind>]`, and how many were replaced.
pub fn scrub(text: &str) -> (String, usize) {
    let mut out = text.to_owned();
    let mut count = 0;
    for rule in RULES.iter() {
        let replaced = rule.re.replace_all(&out, |c: &Captures| {
            let whole = c.get(0).expect("match");
            let secret = c.get(rule.group).expect("group");
            if secret.as_str().starts_with("[REDACTED") {
                return whole.as_str().to_owned();
            }
            count += 1;
            let (start, end) = (secret.start() - whole.start(), secret.end() - whole.start());
            format!("{}[REDACTED:{}]{}", &whole.as_str()[..start], rule.kind, &whole.as_str()[end..])
        });
        out = replaced.into_owned();
    }
    (out, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn redacted(text: &str) -> String {
        scrub(text).0
    }

    #[test]
    fn provider_tokens_are_redacted() {
        let cases = [
            ("ghp_abcdefghijklmnopqrstuvwxyz0123456789AB", "github-token"),
            ("github_pat_11ABCDEFG0abcdefghijklmnop", "github-token"),
            ("glpat-abcDEF123456789_xyz-00", "gitlab-token"),
            ("xoxb-1234567890-abcdefghij", "slack-token"),
            ("lin_api_abcdefghijklmnopqrstuvwxyz0123456789ABCD", "linear-key"),
            ("sk-ant-api03-abcdefghijklmnopqrstuvwx", "api-key"),
            ("AKIAIOSFODNN7EXAMPLE", "aws-key"),
            ("AIzaSyA1234567890abcdefghijklmnopqrstuv", "google-key"),
        ];
        for (secret, kind) in cases {
            let out = redacted(&format!("use {secret} here"));
            assert_eq!(out, format!("use [REDACTED:{kind}] here"), "{secret}");
        }
    }

    #[test]
    fn private_key_block_is_redacted_whole() {
        let pem = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk\nAAAA\n-----END OPENSSH PRIVATE KEY-----";
        assert_eq!(redacted(&format!("key:\n{pem}\nend")), "key:\n[REDACTED:private-key]\nend");
    }

    #[test]
    fn jwt_and_bearer_are_redacted() {
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N";
        assert_eq!(redacted(&format!("t={jwt}")), "t=[REDACTED:jwt]");
        assert_eq!(
            redacted("Authorization: Bearer abcdefghijklmnop1234"),
            "Authorization: Bearer [REDACTED:bearer]"
        );
    }

    #[test]
    fn url_password_keeps_user_and_host() {
        assert_eq!(
            redacted("postgres://app:s3cr3tPass@localhost:5434/hd"),
            "postgres://app:[REDACTED:url-password]@localhost:5434/hd"
        );
    }

    #[test]
    fn secret_named_assignments_keep_the_name() {
        assert_eq!(redacted("DB_PASSWORD=hunter2hunter"), "DB_PASSWORD=[REDACTED:assignment]");
        assert_eq!(redacted(r#""api_key": "abcdef123456""#), r#""api_key": "[REDACTED:assignment]""#);
        assert_eq!(redacted("client_secret: 9f8e7d6c5b4a"), "client_secret: [REDACTED:assignment]");
    }

    #[test]
    fn references_and_short_values_are_not_secrets() {
        for text in [
            "QDRANT_API_KEY=${QDRANT_API_KEY}",
            "password: <your password>",
            "TOKEN=%TOKEN%",
            "max_tokens: 1024",
            "the token count is 5",
            "run npm test and check the password field validation",
        ] {
            assert_eq!(redacted(text), text, "{text}");
        }
    }

    #[test]
    fn ordinary_code_and_prose_pass_through_unchanged() {
        let text = "Fixed the Selenium form bug from HD-609.\nlet id = point_id(\"claude\", key);\nhttps://jira.example.org/browse/HD-609";
        assert_eq!(scrub(text), (text.to_owned(), 0));
    }

    #[test]
    fn count_reports_each_redaction_once() {
        let (out, n) = scrub("a ghp_abcdefghijklmnopqrstuvwxyz0123456789AB b AKIAIOSFODNN7EXAMPLE");
        assert_eq!(n, 2);
        assert!(!out.contains("ghp_") && !out.contains("AKIA"));
    }

    #[test]
    fn scrubbing_is_idempotent() {
        let once = redacted("TOKEN=abcdefghij1234567890 and Bearer abcdefghijklmnop1234");
        assert_eq!(scrub(&once), (once.clone(), 0));
    }

    #[test]
    fn polish_password_assignment_is_redacted() {
        assert_eq!(redacted("hasło: Tajne123!"), "hasło: [REDACTED:assignment]");
        assert_eq!(redacted("HASLO=Tajne123!"), "HASLO=[REDACTED:assignment]");
    }

    #[test]
    fn polish_text_survives() {
        let text = "Hasło do bazy jest w menedżerze haseł, nie w repozytorium.";
        assert_eq!(redacted(text), text);
    }
}
