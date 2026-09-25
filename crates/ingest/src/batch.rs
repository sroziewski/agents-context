use std::ops::Range;

/// Splits consecutive items into embedding batches. A batch's cost is `longest × count`, because
/// every input is padded to the longest one — that padded size, not the sum, is what has to stay
/// under `max_padded_tokens` so one GPU call finishes before Windows' ~2 s TDR watchdog resets
/// the device.
pub fn batches(tokens: &[usize], max_padded_tokens: usize, max_items: usize) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut longest = 0;
    for (i, &t) in tokens.iter().enumerate() {
        let count = i - start;
        let grown = longest.max(t) * (count + 1);
        if count > 0 && (count >= max_items || grown > max_padded_tokens) {
            out.push(start..i);
            start = i;
            longest = 0;
        }
        longest = longest.max(t);
    }
    if start < tokens.len() {
        out.push(start..tokens.len());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_has_no_batches() {
        assert!(batches(&[], 100, 8).is_empty());
    }

    #[test]
    fn short_items_fill_up_to_max_items() {
        assert_eq!(batches(&[10; 10], 1000, 4), [0..4, 4..8, 8..10]);
    }

    #[test]
    fn padded_cost_limits_batch_of_long_items() {
        // 3 × 400 = 1200 > 1000, so only two long items per batch.
        assert_eq!(batches(&[400, 400, 400, 400], 1000, 8), [0..2, 2..4]);
    }

    #[test]
    fn one_long_item_pads_the_whole_batch() {
        // 10 + 10 + 500×3 would pad to 1500.
        assert_eq!(batches(&[10, 10, 500, 10], 1000, 8), [0..2, 2..4]);
    }

    #[test]
    fn oversized_item_still_gets_its_own_batch() {
        assert_eq!(batches(&[5000, 10], 1000, 8), [0..1, 1..2]);
    }

    #[test]
    fn batches_cover_every_item_exactly_once_in_order() {
        let tokens: Vec<usize> = (0..50).map(|i| (i * 37) % 480 + 1).collect();
        let b = batches(&tokens, 2000, 8);
        let flat: Vec<usize> = b.iter().flat_map(|r| r.clone()).collect();
        assert_eq!(flat, (0..50).collect::<Vec<_>>());
        for r in &b {
            let longest = tokens[r.clone()].iter().max().unwrap();
            assert!(r.len() == 1 || longest * r.len() <= 2000);
            assert!(r.len() <= 8);
        }
    }
}
