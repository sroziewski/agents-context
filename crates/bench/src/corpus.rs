//! Synthetic mixed Polish/English texts sized like the chunks the indexer will produce:
//! a Teams one-liner, a prompt plus reply, and a full chunk near the 512-token budget.

const SENTENCES: &[&str] = &[
    "The database migration failed with a timeout on port 5439.",
    "Migracja bazy danych nie powiodła się z powodu przekroczenia limitu czasu.",
    "I refactored the sync loop so the watermark only advances after a successful save.",
    "Użytkownicy nie mogą się zalogować przez SSO od dzisiejszego rana.",
    "Added a unit test that pins the behaviour of the diacritics-insensitive name matcher.",
    "Proszę o przegląd zmian w module helpdesku przed piątkowym wdrożeniem.",
    "The Qdrant collection uses a dense and a sparse vector fused with reciprocal rank fusion.",
    "Spotkanie zespołu SEDN przeniesiono na czwartek na godzinę dziesiątą.",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthClass {
    Short,
    Medium,
    Long,
}

impl LengthClass {
    pub const ALL: [LengthClass; 3] = [Self::Short, Self::Medium, Self::Long];

    pub fn name(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Medium => "medium",
            Self::Long => "long",
        }
    }

    /// Approximate words per text; bge-m3's tokenizer yields roughly 1.3–2 tokens per word
    /// for this PL/EN mix, so `Long` lands just under the 512-token default.
    pub fn words(self) -> usize {
        match self {
            Self::Short => 12,
            Self::Medium => 110,
            Self::Long => 280,
        }
    }
}

/// `n` distinct texts of the given class. Rotating the starting sentence keeps texts distinct,
/// so no runtime-level caching can flatter the numbers.
pub fn texts(class: LengthClass, n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let mut words: Vec<&str> = Vec::with_capacity(class.words());
            let mut s = i;
            while words.len() < class.words() {
                words.extend(SENTENCES[s % SENTENCES.len()].split_whitespace());
                s += 1;
            }
            words.truncate(class.words());
            format!("[{i}] {}", words.join(" "))
        })
        .collect()
}

pub const PAIRS: &[(&str, &str)] = &[
    ("Dlaczego migracja bazy danych się nie udała?", "The database migration failed due to a timeout."),
    ("Dlaczego migracja bazy danych się nie udała?", "Lunch is at noon in the kitchen."),
    ("login problems with single sign-on", "Użytkownicy nie mogą się zalogować przez SSO."),
    ("login problems with single sign-on", "Faktura za serwery została opłacona."),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texts_have_requested_count_and_word_budget() {
        for class in LengthClass::ALL {
            let t = texts(class, 5);
            assert_eq!(t.len(), 5);
            for text in &t {
                // +1 for the "[i]" prefix that keeps texts distinct.
                assert_eq!(text.split_whitespace().count(), class.words() + 1, "{class:?}");
            }
        }
    }

    #[test]
    fn texts_are_distinct() {
        let t = texts(LengthClass::Short, 20);
        let unique: std::collections::HashSet<_> = t.iter().collect();
        assert_eq!(unique.len(), t.len());
    }

    #[test]
    fn corpus_mixes_polish_and_english() {
        let all = texts(LengthClass::Long, 1).concat();
        assert!(all.contains("bazy") && all.contains("database"));
    }

    #[test]
    fn classes_grow_in_length() {
        let w: Vec<_> = LengthClass::ALL.iter().map(|c| c.words()).collect();
        assert!(w.windows(2).all(|p| p[0] < p[1]));
    }

    #[test]
    fn zero_texts_is_empty() {
        assert!(texts(LengthClass::Medium, 0).is_empty());
    }
}
