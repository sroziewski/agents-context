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
        rule("atlassian-token", r"\b(AT(?:ATT|CTT)[A-Za-z0-9_\-=]{20,})", 1),
        rule("bearer", r"(?i)\bbearer\s+([A-Za-z0-9._~+/=\-]{16,})", 1),
        rule("basic-auth", r"(?i)\bauthorization[\x22']?\s*[:=]\s*[\x22']?basic\s+([A-Za-z0-9+/]{8,}={0,2})", 1),
        // scheme://user:password@host — keeps scheme, user and host so the context stays readable.
        rule("url-password", r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:([^\s@/]+)@", 1),
        // scheme://:password@host, as Redis URLs carry a password without a user.
        rule("url-password", r"(?i)\b[a-z][a-z0-9+.\-]*://:([^\s@/]+)@", 1),
        // A quoted value may contain spaces (a passphrase); the plain assignment below stops at
        // the first one and would leave the rest of the phrase behind.
        rule(
            "assignment",
            r#"(?i)\b[A-Z0-9_.\-]*(?:password|passwd|passphrase|pwd|has[łl]o|secret|token|api[_\-]?key|access[_\-]?key|private[_\-]?key|client[_\-]?secret)[A-Z0-9_.\-]*["'`]?\s*[:=]\s*["'`]([^"'`\n$\{<%][^"'`\n]*\s[^"'`\n]*)["'`]"#,
            1,
        ),
        // NAME=value / "name": "value" where the name says it is a secret. Values starting with
        // `$`, `{`, `<` or `%` are references (env vars, templates), not secrets.
        rule(
            "assignment",
            r#"(?i)\b[A-Z0-9_.\-]*(?:password|passwd|passphrase|pwd|has[łl]o|secret|token|api[_\-]?key|access[_\-]?key|private[_\-]?key|client[_\-]?secret)[A-Z0-9_.\-]*["'`]?\s*[:=]\s*["'`]?([^\s"'`,;$\{<%][^\s"'`,;]{5,})"#,
            1,
        ),
    ]
});

/// Words that announce a secret in prose ("password to …", "hasło do bazy to …").
static SECRET_WORD: LazyLock<Regex> = LazyLock::new(|| {
    // Polish inflections of hasło as whole words only: a prefix match would also hit English
    // identifiers such as `hasLanguage(`, since matching is case-insensitive.
    Regex::new(r"(?i)\b(?:pass(?:word|wd|phrase)s?|pwd|has[łl](?:o|a|em|ami|ach|om)|haśle|secrets?|credentials?)\b")
        .expect("secret word")
});
static MARKER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[REDACTED:[a-z\-]+\]").expect("marker"));
/// `host:port` and colon-joined lowercase names (CI job ids like `build:docker`).
static COLON_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9_.\-]+(?::[a-z0-9_.\-]+)+$").expect("colon name"));
static UUID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$").expect("uuid")
});

/// How far after a secret word a bare value is looked for, in whitespace-separated words.
const PROXIMITY_WORDS: usize = 8;

/// Shortest literal accepted from the redaction list; shorter ones would hit ordinary words.
pub const MIN_LITERAL_LEN: usize = 6;

/// Redaction with an optional list of literal secrets (e.g. from `redact.txt`) on top of the
/// built-in rules. Literals catch what no pattern can: a known password written in plain prose.
#[derive(Default, Clone)]
pub struct Redactor {
    /// Longest first, so a literal containing another is replaced whole.
    literals: Vec<String>,
}

impl std::fmt::Debug for Redactor {
    // Never print the secrets themselves.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Redactor({} literals)", self.literals.len())
    }
}

impl Redactor {
    /// Trims entries and drops blank lines, `#` comments and entries shorter than
    /// [`MIN_LITERAL_LEN`].
    pub fn new(literals: impl IntoIterator<Item = String>) -> Self {
        let mut literals: Vec<String> = literals
            .into_iter()
            // A UTF-8 BOM (files saved by Notepad) or a Ctrl+Z typed mid-line would otherwise
            // become part of the entry, which then never matches.
            .map(|l| l.trim_matches(|c: char| c.is_whitespace() || c == '\u{feff}' || c == '\u{1a}').to_owned())
            .filter(|l| l.chars().count() >= MIN_LITERAL_LEN && !l.starts_with('#'))
            .collect();
        literals.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        literals.dedup();
        Self { literals }
    }

    /// A missing file means no literals; any other read error is reported, because silently
    /// indexing without the user's list would leak exactly what they asked to hide.
    pub fn from_file(path: &std::path::Path) -> std::io::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(Self::new(text.lines().map(str::to_owned))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Returns `text` with every secret replaced by `[REDACTED:<kind>]`, and how many were
    /// replaced.
    pub fn scrub(&self, text: &str) -> (String, usize) {
        let mut out = text.to_owned();
        let mut count = 0;
        for literal in &self.literals {
            let n = out.matches(literal.as_str()).count();
            if n > 0 {
                out = out.replace(literal.as_str(), "[REDACTED:listed]");
                count += n;
            }
        }
        let (out, n) = scrub_rules(&out);
        let (out, m) = scrub_near_secret_words(&out);
        (out, count + n + m)
    }
}

/// Built-in rules only, no literal list.
pub fn scrub(text: &str) -> (String, usize) {
    Redactor::default().scrub(text)
}

/// A bare value shortly after a secret word on the same line, e.g. "set the password to
/// `Xy9…`". Only tokens that look generated count: 8-64 characters mixing at least three of
/// lowercase, uppercase, digits and symbols. Hex hashes (two classes), UUIDs, paths, URLs and
/// file names are left alone. Accepted false positive: a name like `JanKowalski2` right after
/// "password for".
fn scrub_near_secret_words(text: &str) -> (String, usize) {
    let mut count = 0;
    let lines: Vec<String> = text
        .split('\n')
        .map(|line| {
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            // The kind inside `[REDACTED:url-password]` is not a secret word announcing what follows.
            let markers: Vec<(usize, usize)> = MARKER.find_iter(line).map(|m| (m.start(), m.end())).collect();
            for m in SECRET_WORD.find_iter(line).filter(|m| !markers.iter().any(|&(s, e)| s <= m.start() && m.end() <= e)) {
                let rest = &line[m.end()..];
                let mut offset = m.end();
                // Punctuation-only tokens ("password: …") do not count as words.
                for word in rest.split_whitespace().filter(|w| w.chars().any(char::is_alphanumeric)).take(PROXIMITY_WORDS) {
                    let at = offset + line[offset..].find(word).expect("word from this slice");
                    offset = at + word.len();
                    // Quotes, brackets and markdown emphasis (`**bold**`, `~~x~~`) around a word;
                    // a trailing `!` stays, it is common at the end of real passwords.
                    let trimmed = word.trim_matches(|c: char| "`'\"()[]{}<>,;:.?*~".contains(c));
                    let trimmed = trimmed.trim_end_matches(['.', ',', ';', ':']);
                    if looks_generated(trimmed) {
                        let start = at + word.find(trimmed).expect("trimmed from word");
                        ranges.push((start, start + trimmed.len()));
                    }
                }
            }
            if ranges.is_empty() {
                return line.to_owned();
            }
            ranges.sort_unstable();
            ranges.dedup();
            let mut out = String::with_capacity(line.len());
            let mut last = 0;
            for (start, end) in ranges {
                if start < last {
                    continue;
                }
                out.push_str(&line[last..start]);
                out.push_str("[REDACTED:near-secret-word]");
                count += 1;
                last = end;
            }
            out.push_str(&line[last..]);
            out
        })
        .collect();
    (lines.join("\n"), count)
}

fn looks_generated(token: &str) -> bool {
    let len = token.chars().count();
    // Also the inside of an existing `[REDACTED:kind]` marker once its brackets are trimmed.
    if !(8..=64).contains(&len) || token.contains("REDACTED") {
        return false;
    }
    if token.contains("://") || token.contains('/') || token.contains('\\') || token.contains('=') || UUID.is_match(token) || COLON_NAME.is_match(token)
    {
        return false;
    }
    // file.ext, Module.method
    if let Some((_, ext)) = token.rsplit_once('.') {
        if (1..=5).contains(&ext.len()) && ext.chars().all(|c| c.is_ascii_alphabetic()) {
            return false;
        }
    }
    let classes = [
        token.chars().any(|c| c.is_lowercase()),
        token.chars().any(|c| c.is_uppercase()),
        token.chars().any(|c| c.is_ascii_digit()),
        // `-`, `_` and `.` are what identifiers are made of (k8s names, hostnames, snake_case),
        // so they do not count as the symbol class; real passwords use other symbols or mixed case.
        // Apostrophes too: possessives like GitLab's.
        token.chars().any(|c| !c.is_alphanumeric() && !matches!(c, '-' | '_' | '.' | '\'' | '’')),
    ];
    classes.iter().filter(|&&b| b).count() >= 3
}

fn scrub_rules(text: &str) -> (String, usize) {
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
    fn bare_value_after_password_word_is_redacted() {
        let text = "Update SOCN system password to bcrypt 2a hash `Xq7dHgwQaKi82p` on the test DB";
        let out = redacted(text);
        assert!(!out.contains("Xq7dHgwQaKi82p"), "{out}");
        assert_eq!(out, "Update SOCN system password to bcrypt 2a hash `[REDACTED:near-secret-word]` on the test DB");
    }

    #[test]
    fn polish_hasło_with_bare_value_is_redacted() {
        assert_eq!(redacted("hasło do bazy to Tajne123!x"), "hasło do bazy to [REDACTED:near-secret-word]");
        assert_eq!(redacted("Hasłem jest pX9#kLm2qQ."), "Hasłem jest [REDACTED:near-secret-word].");
        assert_eq!(redacted("w haśle było Qw3rTy99z"), "w haśle było [REDACTED:near-secret-word]");
        assert_eq!(redacted("hasło to **Tajne123!x**"), "hasło to **[REDACTED:near-secret-word]**");
    }

    #[test]
    fn prose_hashes_paths_and_uuids_near_secret_words_survive() {
        for text in [
            "password field validation fails on submit",
            "the password hash is 5d41402abc4b2a76b9719d911017c592",
            "password handling lives in D:\\dev\\x\\PasswordConfig2.java",
            "secret stored at https://vault.example/v1/Secret9x",
            "credentials id 0d965223-5fab-415b-9a5e-5992443b920c",
            "password policy: see SecurityConfig.java",
            "Użytkownik zmienił hasło wczoraj po południu",
            "fill the secret store behind `ai4sd-gitlab-token2` with these keys",
            "the same user and password on sedn-dev01 and test02-db",
            "ExternalSecret builds `k8s_secret_v2` from project variables",
            "profile.getDisplayedNames().hasLanguage(\"pl\") and hasLocation2(\"x\")",
            "recreates the secrets from GitLab's CI/CD variables",
            "the password differs on the prod host zsun-prod01:8080 today",
            "deploy:secrets then build:docker then index:trigger2 run",
            "the deploy:secrets job even **deletes the ExternalSecret** and rebuilds it",
            "secrets rotated **2060's** style",
            "the flag `offer_to_set_password=false` stays",
        ] {
            assert_eq!(redacted(text), text, "{text}");
        }
    }

    #[test]
    fn value_too_far_or_on_another_line_survives() {
        let far = "password was discussed at length yesterday by the whole team and then Xq7dHgwQaKi82p";
        assert_eq!(redacted(far), far);
        let other_line = "set the password\nXq7dHgwQaKi82p";
        assert_eq!(redacted(other_line), other_line);
    }

    #[test]
    fn proximity_rule_is_idempotent_and_counted() {
        let (once, n) = scrub("password: new one is Xq7dHgwQaKi82p and old was Pq8dHgwQaKi82z");
        assert_eq!(n, 2, "{once}");
        assert_eq!(scrub(&once), (once.clone(), 0));
    }

    #[test]
    fn listed_literals_are_redacted_anywhere() {
        let r = Redactor::new(["plainsecret".to_owned(), "   ".to_owned(), "# comment".to_owned(), "abc".to_owned()]);
        assert_eq!(r.len(), 1, "blank, comment and too-short entries are dropped");
        let (out, n) = r.scrub("value plainsecret here and plainsecret again");
        assert_eq!(out, "value [REDACTED:listed] here and [REDACTED:listed] again");
        assert_eq!(n, 2);
    }

    #[test]
    fn longer_literal_wins_over_its_prefix() {
        let r = Redactor::new(["secret-part".to_owned(), "secret-part-longer".to_owned()]);
        assert_eq!(r.scrub("x secret-part-longer y").0, "x [REDACTED:listed] y");
    }

    #[test]
    fn redactor_debug_never_shows_secrets() {
        let r = Redactor::new(["topsecret1".to_owned()]);
        assert_eq!(format!("{r:?}"), "Redactor(1 literals)");
    }

    #[test]
    fn literal_file_missing_is_empty_and_present_is_loaded() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Redactor::from_file(&dir.path().join("none.txt")).unwrap().is_empty());
        let f = dir.path().join("redact.txt");
        std::fs::write(&f, "\u{feff}hunter2hunter2\r\n# my secrets\n\nsecond-secret\u{1a}\n").unwrap();
        let r = Redactor::from_file(&f).unwrap();
        assert_eq!(r.len(), 2, "CRLF, BOM and Ctrl+Z are trimmed");
        assert!(!r.scrub("a hunter2hunter2 b").0.contains("hunter2"), "BOM-prefixed first entry still matches");
        assert!(!r.scrub("x second-secret y").0.contains("second-secret"));
    }

    #[test]
    fn polish_text_survives() {
        let text = "Hasło do bazy jest w menedżerze haseł, nie w repozytorium.";
        assert_eq!(redacted(text), text);
    }

    #[test]
    fn new_rules_redact_their_secret_and_keep_the_context() {
        let cases = [
            ("redis-url", "REDIS_URL redis://:Rd7pw-Q2x@cache.local:6380/2 ok", "REDIS_URL redis://:[REDACTED:url-password]@cache.local:6380/2 ok"),
            ("rediss-url-upper-scheme", "Rediss://:x9Yq@redis.example:6379/1", "Rediss://:[REDACTED:url-password]@redis.example:6379/1"),
            ("quoted-phrase", r#"password: "correct horse battery staple" end"#, r#"password: "[REDACTED:assignment]" end"#),
            ("quoted-phrase-single", "db_secret = 'two words here' x", "db_secret = '[REDACTED:assignment]' x"),
            ("quoted-phrase-json", r#"{"smtp_password": "open sesame 42"}"#, r#"{"smtp_password": "[REDACTED:assignment]"}"#),
            ("passphrase-env", "GPG_PASSPHRASE=k3yPhr4seValue", "GPG_PASSPHRASE=[REDACTED:assignment]"),
            ("passphrase-lower", "passphrase: wordwordword", "passphrase: [REDACTED:assignment]"),
            ("basic-header", "Authorization: Basic dXNlcjpwYXNzd29yZA== sent", "Authorization: Basic [REDACTED:basic-auth] sent"),
            ("basic-curl", "curl -H 'authorization=basic YWxhZGRpbjpvcGVu' url", "curl -H 'authorization=basic [REDACTED:basic-auth]' url"),
            ("atlassian-api", "use ATATT3xFfGF0T4kZ-ab_cd=1A2B3C4D now", "use [REDACTED:atlassian-token] now"),
            ("atlassian-cloud", "ATCTT3xFfGF0abcdefghijklmnop", "[REDACTED:atlassian-token]"),
            ("atlassian-in-assignment", "JIRA_TOKEN=ATATT3xFfGF0T4kZabcdefghij", "JIRA_TOKEN=[REDACTED:atlassian-token]"),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter(|(_, text, want)| redacted(text) != *want)
            .map(|(name, text, want)| format!("{name}: got {:?}, want {want:?}", redacted(text)))
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    #[test]
    fn a_redis_url_without_a_path_keeps_its_host() {
        assert_eq!(redacted("REDIS_URL=redis://:x9Yq@redis.example:6379"), "REDIS_URL=redis://:[REDACTED:url-password]@redis.example:6379");
    }

    #[test]
    fn a_quoted_basic_auth_header_is_redacted() {
        assert_eq!(
            redacted(r#"headers: {"Authorization": "Basic dXNlcjpwYXNzd29yZA=="}"#),
            r#"headers: {"Authorization": "Basic [REDACTED:basic-auth]"}"#
        );
    }

    #[test]
    fn each_new_rule_counts_one_redaction_and_is_idempotent() {
        for text in [
            "redis://:Rd7pw-Q2x@cache.local:6380/2",
            r#"password: "correct horse battery staple""#,
            "passphrase=k3yPhr4seValue",
            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
            "ATATT3xFfGF0T4kZ-ab_cd=1A2B3C4D",
        ] {
            let (once, n) = scrub(text);
            assert_eq!(n, 1, "{text} -> {once}");
            assert_eq!(scrub(&once), (once.clone(), 0), "{text}");
        }
    }

    #[test]
    fn near_misses_of_the_new_rules_pass_through() {
        for text in [
            "redis://cache.local:6379/0",
            "redis://@cache.local:6379",
            r#"title: "correct horse battery staple""#,
            r#"password: "<your pass phrase>""#,
            r#"password: "${DB PASS}""#,
            "Authorization: Basic auth is off",
            "ATATT is a prefix, ATATTshort too",
            "the passphrase field is optional",
        ] {
            assert_eq!(scrub(text), (text.to_owned(), 0), "{text}");
        }
    }
}
