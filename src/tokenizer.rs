//! Byte-level BPE tokenization for CLIP text encoding
//! (issue #61), driven by the `tokenizer.json` that ships
//! with CLIP ONNX exports. Only what the export format
//! declares is implemented: whitespace-collapsing lowercase
//! normalization, the export's pre-tokenizer split (letter
//! runs, single digits, punctuation runs, contractions,
//! whitespace pieces), the GPT-2 byte-to-unicode mapping,
//! merges ranked by the file's merge list, the `</w>`
//! end-of-word suffix, and SOT/EOT wrapping with padding
//! to the model's 77-token context.
//!
//! The tokenizer is only ever fed operator label strings,
//! which are short phrases; NFC normalization is the one
//! declared step not reproduced (labels with combining
//! characters should be written precomposed).

use std::collections::HashMap;

/// The CLIP context length: every encoding is exactly
/// this many tokens.
pub const CONTEXT_LEN: usize = 77;

pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    sot: u32,
    eot: u32,
}

impl std::fmt::Debug for ClipTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipTokenizer")
            .field("vocab", &self.vocab.len())
            .field("merges", &self.ranks.len())
            .finish_non_exhaustive()
    }
}

impl ClipTokenizer {
    /// Load a `tokenizer.json` (Hugging Face tokenizer
    /// format) from disk.
    pub fn load(path: &str) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read the tokenizer file {path}: {e}"))?;
        Self::from_json(&raw)
    }

    /// Parse a tokenizer from its JSON text.
    pub fn from_json(raw: &str) -> Result<Self, String> {
        let doc: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| format!("cannot parse the tokenizer file: {e}"))?;
        let model = doc
            .get("model")
            .ok_or("the tokenizer file has no model section")?;
        if model.get("type").and_then(|t| t.as_str()) != Some("BPE") {
            return Err("the tokenizer file is not a BPE tokenizer".to_string());
        }
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or("the tokenizer file has no vocab")?;
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(vocab_obj.len());
        for (token, id) in vocab_obj {
            let id = id
                .as_u64()
                .ok_or_else(|| format!("vocab id for {token:?} is not a number"))?;
            vocab.insert(token.clone(), id as u32);
        }
        let mut ranks: HashMap<(String, String), u32> = HashMap::new();
        if let Some(merges) = model.get("merges").and_then(|m| m.as_array()) {
            for (rank, merge) in merges.iter().enumerate() {
                let text = merge.as_str().ok_or("a merge entry is not a string")?;
                let (a, b) = text
                    .split_once(' ')
                    .ok_or_else(|| format!("malformed merge entry {text:?}"))?;
                ranks.insert((a.to_string(), b.to_string()), rank as u32);
            }
        }
        let sot = *vocab
            .get("<|startoftext|>")
            .ok_or("the tokenizer vocab has no <|startoftext|>")?;
        let eot = *vocab
            .get("<|endoftext|>")
            .ok_or("the tokenizer vocab has no <|endoftext|>")?;
        Ok(Self {
            vocab,
            ranks,
            sot,
            eot,
        })
    }

    /// Encode a string to exactly [`CONTEXT_LEN`] token ids
    /// plus its attention mask (1 for real tokens including
    /// SOT/EOT, 0 for padding). Padding uses the EOT id,
    /// the convention for this export family.
    pub fn encode(&self, text: &str) -> (Vec<i64>, Vec<i64>) {
        let mut ids: Vec<i64> = vec![self.sot as i64];
        for piece in pretokenize(&normalize(text)) {
            ids.extend(self.encode_piece(&piece));
        }
        ids.push(self.eot as i64);
        if ids.len() > CONTEXT_LEN {
            // Overlong input: keep the head and restore the
            // EOT as the final token.
            ids.truncate(CONTEXT_LEN);
            ids[CONTEXT_LEN - 1] = self.eot as i64;
        }
        let mut mask = vec![1i64; ids.len()];
        while ids.len() < CONTEXT_LEN {
            ids.push(self.eot as i64);
            mask.push(0);
        }
        (ids, mask)
    }

    /// BPE-encode one pre-token: byte-map it, suffix the
    /// final symbol with `</w>`, then merge by rank.
    fn encode_piece(&self, piece: &str) -> Vec<i64> {
        let mut symbols: Vec<String> = piece.bytes().map(|b| byte_char(b).to_string()).collect();
        if let Some(last) = symbols.last_mut() {
            last.push_str("</w>");
        }
        loop {
            // The lowest-rank adjacent pair wins; merge all
            // of its occurrences, left to right.
            let mut best: Option<u32> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self
                    .ranks
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    best = Some(best.map_or(rank, |r: u32| r.min(rank)));
                }
            }
            let Some(best) = best else { break };
            let mut merged: Vec<String> = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len()
                    && self
                        .ranks
                        .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                        == Some(&best)
                {
                    let mut joined = symbols[i].clone();
                    joined.push_str(&symbols[i + 1]);
                    merged.push(joined);
                    i += 2;
                } else {
                    merged.push(symbols[i].clone());
                    i += 1;
                }
            }
            symbols = merged;
        }
        symbols
            .iter()
            .filter_map(|s| self.vocab.get(s).map(|&id| id as i64))
            .collect()
    }
}

/// The declared normalization: collapse each whitespace
/// run to one space, then lowercase.
fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_ws = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !in_ws {
                out.push(' ');
                in_ws = true;
            }
        } else {
            out.extend(ch.to_lowercase());
            in_ws = false;
        }
    }
    out
}

/// Split normalized text the way the export's
/// pre-tokenizer does: contractions (`'s`, `'t`, `'re`,
/// `'ve`, `'m`, `'ll`, `'d`) stand alone, letter runs
/// group, each digit stands alone, punctuation runs
/// group, and whitespace between them forms its own
/// pieces.
fn pretokenize(text: &str) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let chars: Vec<char> = text.chars().collect();
    let mut pieces: Vec<String> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '\'' {
            let rest: String = chars[i..].iter().collect();
            if let Some(c) = CONTRACTIONS.iter().find(|c| rest.starts_with(**c)) {
                pieces.push(c.to_string());
                i += c.chars().count();
                continue;
            }
        }
        if ch.is_whitespace() {
            let start = i;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            pieces.push(chars[start..i].iter().collect());
        } else if ch.is_alphabetic() {
            let start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            pieces.push(chars[start..i].iter().collect());
        } else if ch.is_numeric() {
            pieces.push(ch.to_string());
            i += 1;
        } else {
            let start = i;
            while i < chars.len()
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_numeric()
            {
                i += 1;
            }
            pieces.push(chars[start..i].iter().collect());
        }
    }
    pieces
}

/// The GPT-2 byte-to-unicode mapping: printable bytes map
/// to themselves, the rest to codepoints from U+0100 up,
/// assigned in byte order.
fn byte_char(b: u8) -> char {
    let b = b as u32;
    if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
        char::from_u32(b).unwrap_or('\u{100}')
    } else {
        // Count how many non-printable bytes precede this
        // one: that index offsets from U+0100.
        let mut n = 0u32;
        for prev in 0..b {
            if !((33..=126).contains(&prev)
                || (161..=172).contains(&prev)
                || (174..=255).contains(&prev))
            {
                n += 1;
            }
        }
        char::from_u32(256 + n).unwrap_or('\u{100}')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy tokenizer: specials at 0/1, three suffix
    /// forms, one merge.
    fn toy() -> ClipTokenizer {
        ClipTokenizer::from_json(
            r#"{"model": {"type": "BPE",
                "vocab": {"<|startoftext|>": 0, "<|endoftext|>": 1,
                          "a</w>": 10, "b</w>": 11, "Ġ</w>": 12,
                          "a": 13, "b": 14, "ab</w>": 15},
                "merges": ["a b</w>"]}}"#,
        )
        .unwrap()
    }

    #[test]
    fn encode_wraps_pads_and_masks() {
        let t = toy();
        let (ids, mask) = t.encode("");
        assert_eq!(ids.len(), CONTEXT_LEN);
        assert_eq!(ids[0], 0, "SOT first");
        assert_eq!(ids[1], 1, "EOT second on empty input");
        assert_eq!(mask[0], 1);
        assert_eq!(mask[1], 1);
        assert!(mask[2..].iter().all(|&m| m == 0), "padding masked out");
        assert!(ids[2..].iter().all(|&i| i == 1), "padding is the EOT id");
    }

    #[test]
    fn encode_merges_by_rank_and_suffixes_word_ends() {
        let t = toy();
        let (ids, _) = t.encode("ab");
        assert_eq!(&ids[..3], &[0, 15, 1], "a + b</w> merge to ab</w>");
        // No merge across pieces: "a b" is a, space, b.
        let (ids, _) = t.encode("a b");
        assert_eq!(&ids[..5], &[0, 10, 12, 11, 1]);
    }

    #[test]
    fn encode_lowercases_and_collapses_whitespace() {
        let t = toy();
        let (ids, _) = t.encode("A   B");
        assert_eq!(&ids[..5], &[0, 10, 12, 11, 1]);
    }

    #[test]
    fn pretokenize_splits_like_the_export() {
        assert_eq!(pretokenize("kite surfing"), ["kite", " ", "surfing"]);
        assert_eq!(pretokenize("top 10"), ["top", " ", "1", "0"]);
        assert_eq!(pretokenize("it's me"), ["it", "'s", " ", "me"]);
        assert_eq!(pretokenize("sun-set!"), ["sun", "-", "set", "!"]);
    }

    #[test]
    fn byte_char_maps_space_and_printables() {
        assert_eq!(byte_char(b' '), 'Ġ');
        assert_eq!(byte_char(b'a'), 'a');
        assert_eq!(byte_char(0), '\u{100}');
    }

    #[test]
    fn from_json_rejects_a_non_bpe_file() {
        assert!(ClipTokenizer::from_json(r#"{"model": {"type": "WordPiece"}}"#).is_err());
        assert!(ClipTokenizer::from_json("not json").is_err());
    }
}
