use std::path::Path;

/// Counts model tokens. Chunks must fit the embedder's window exactly: past it the tokenizer
/// truncates silently and the tail of the chunk simply vanishes from the index.
pub trait TokenCounter {
    fn count(&self, text: &str) -> usize;
}

/// The bge-m3 tokenizer (`tokenizer.json` of the model export); counts include the two special
/// tokens the model adds, since those occupy the window too.
pub struct HfTokenizer(tokenizers::Tokenizer);

impl HfTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let mut t = tokenizers::Tokenizer::from_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // The file ships with truncation at 512; counting must see the real length.
        t.with_truncation(None).map_err(|e| e.to_string())?;
        t.with_padding(None);
        Ok(Self(t))
    }
}

impl TokenCounter for HfTokenizer {
    fn count(&self, text: &str) -> usize {
        self.0.encode(text, true).map_or_else(|_| text.len(), |e| e.len())
    }
}

/// Whitespace word count; stands in for the real tokenizer in tests.
pub struct WordCounter;

impl TokenCounter for WordCounter {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_counter_counts_whitespace_separated_words() {
        assert_eq!(WordCounter.count("a  b\nc"), 3);
        assert_eq!(WordCounter.count(""), 0);
    }

    #[test]
    fn missing_tokenizer_file_is_a_readable_error() {
        let err = HfTokenizer::from_file(Path::new("no/such/tokenizer.json")).err().unwrap();
        assert!(err.contains("no/such/tokenizer.json"), "{err}");
    }

    fn real() -> Option<HfTokenizer> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/bge-m3-fp32/tokenizer.json");
        p.exists().then(|| HfTokenizer::from_file(&p).unwrap())
    }

    #[test]
    fn real_tokenizer_counts_beyond_512_and_includes_specials() {
        let Some(t) = real() else { return eprintln!("skipped: model not downloaded") };
        assert_eq!(t.count(""), 2, "<s> and </s>");
        let long = "słowo ".repeat(2000);
        assert!(t.count(&long) > 1000, "truncation must be disabled");
    }
}
