//! The user's literal redaction list (`redact.txt`).

use std::io::Write;
use std::path::Path;

use ingest::Redactor;
use ingest::scrub::MIN_LITERAL_LEN;

const HEADER: &str = "# Secrets to redact from the agents-context index, one per line.\n\
# Lines starting with # are comments. Add entries with `ctx redact add` (reads stdin).\n";

/// Appends `secret` to the list (creating the file and its directory) and returns how many
/// entries the list now has. Rejects values too short to redact safely.
pub fn append_literal(path: &Path, secret: &str) -> anyhow::Result<usize> {
    let secret = secret.trim_matches(|c: char| c.is_whitespace() || c == '\u{feff}' || c == '\u{1a}');
    anyhow::ensure!(
        secret.chars().count() >= MIN_LITERAL_LEN,
        "a listed secret must be at least {MIN_LITERAL_LEN} characters, or it would redact ordinary words"
    );
    anyhow::ensure!(!secret.contains('\n'), "one secret per call");
    anyhow::ensure!(!secret.starts_with('#'), "a line starting with # is a comment in the list");
    let existing = Redactor::from_file(path)?;
    if existing.scrub(secret).0 != secret {
        // Already covered, by the list or a built-in rule; keep the file free of duplicates.
        return Ok(existing.len());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let is_new = !path.exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if is_new {
        f.write_all(HEADER.as_bytes())?;
    }
    writeln!(f, "{secret}")?;
    Ok(Redactor::from_file(path)?.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_append_creates_file_with_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/redact.txt");
        assert_eq!(append_literal(&path, "  plainsecret1  ").unwrap(), 1);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# Secrets to redact"));
        assert!(text.ends_with("plainsecret1\n"), "trimmed: {text:?}");
    }

    #[test]
    fn duplicates_and_already_covered_values_are_not_appended() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("redact.txt");
        append_literal(&path, "plainsecret1").unwrap();
        assert_eq!(append_literal(&path, "plainsecret1").unwrap(), 1);
        // Covered by a built-in rule (GitHub token shape).
        append_literal(&path, "ghp_abcdefghijklmnopqrstuvwxyz0123456789AB").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches("plainsecret1").count(), 1);
        assert!(!text.contains("ghp_"));
    }

    #[test]
    fn ctrl_z_and_bom_from_console_input_are_stripped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("redact.txt");
        append_literal(&path, "\u{feff}plainsecret1\u{1a}\r\n").unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().ends_with("\nplainsecret1\n"));
    }

    #[test]
    fn short_multiline_and_comment_values_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("redact.txt");
        assert!(append_literal(&path, "abc").unwrap_err().to_string().contains("at least"));
        assert!(append_literal(&path, "line-one\nline-two").is_err());
        assert!(append_literal(&path, "#notasecret").is_err());
        assert!(!path.exists(), "nothing written on rejection");
    }
}
