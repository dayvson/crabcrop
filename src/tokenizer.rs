//! Byte-level BPE tokenization for CLIP text encoding
//! (issue #61), driven by the `tokenizer.json` that ships
//! with CLIP ONNX exports. Only what the export format
//! declares is implemented: whitespace-collapsing lowercase
//! normalization, the export's pre-tokenizer split (letter
//! runs, single digits, punctuation runs, contractions, with
//! whitespace dropped), the GPT-2 byte-to-unicode mapping,
//! merges ranked by the file's merge list, the `</w>`
//! end-of-word suffix, and SOT/EOT wrapping with padding
//! to the model's 77-token context.
//!
//! The tokenizer is only ever fed operator label strings,
//! which are short phrases. Two declared details are not
//! reproduced exactly: NFC normalization (labels with
//! combining characters should be written precomposed), and
//! the pattern's `\p{L}` letter class, approximated with
//! `char::is_alphabetic`, which admits everything Unicode
//! calls Alphabetic: also letter numbers (Roman numerals),
//! the combining vowel signs of scripts such as Devanagari
//! or Thai, and a few symbols such as circled letters.
//! Labels with those may split differently from the
//! reference. ASCII labels, and precomposed accented
//! letters, split exactly.

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
            let id = u32::try_from(id)
                .map_err(|_| format!("vocab id for {token:?} exceeds the u32 range"))?;
            vocab.insert(token.clone(), id);
        }
        // The merges are not optional: without them BPE
        // degenerates to single characters and every
        // embedding is silently wrong, so a file that lacks
        // them is malformed, not minimal.
        let merges = model
            .get("merges")
            .and_then(|m| m.as_array())
            .ok_or("the tokenizer file has no merges")?;
        let mut ranks: HashMap<(String, String), u32> = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let text = merge.as_str().ok_or("a merge entry is not a string")?;
            let (a, b) = text
                .split_once(' ')
                .ok_or_else(|| format!("malformed merge entry {text:?}"))?;
            // A merge whose product is not itself a vocab
            // entry can only come from a malformed export;
            // the product would vanish at encode time.
            let joined = format!("{a}{b}");
            if !vocab.contains_key(&joined) {
                return Err(format!(
                    "merge {text:?} produces {joined:?}, which is not in the vocab"
                ));
            }
            ranks.insert((a.to_string(), b.to_string()), rank as u32);
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
    /// the convention for this export family. A symbol the
    /// BPE produces that is missing from the vocab is an
    /// error, never a silent skip: dropping it would shift
    /// every later position and produce plausible, wrong
    /// embeddings.
    pub fn encode(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>), String> {
        let mut ids: Vec<i64> = vec![self.sot as i64];
        for piece in pretokenize(&normalize(text)) {
            ids.extend(self.encode_piece(&piece)?);
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
        Ok((ids, mask))
    }

    /// BPE-encode one pre-token: byte-map it, suffix the
    /// final symbol with `</w>`, then merge by rank.
    fn encode_piece(&self, piece: &str) -> Result<Vec<i64>, String> {
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
        let mut out = Vec::with_capacity(symbols.len());
        for s in &symbols {
            match self.vocab.get(s) {
                Some(&id) => out.push(id as i64),
                None => {
                    return Err(format!(
                        "token {s:?} produced by BPE is missing from the tokenizer vocab"
                    ));
                }
            }
        }
        Ok(out)
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
/// pre-tokenizer does. CLIP's `tokenizer.json` declares a
/// `Split` on `'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`
/// with `behavior: Removed, invert: true`: the matches are
/// the pieces, and what lies between them, which can only be
/// whitespace, is dropped before BPE (open_clip's
/// `re.findall` on the same pattern does the same). So
/// contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`)
/// stand alone, letter runs group, each digit stands alone,
/// punctuation runs group, and no piece carries a space.
/// Neither lone whitespace pieces nor a space prefix on the
/// following piece (GPT-2's ` ?\p{L}+`, which CLIP's pattern
/// does not have) is right: both put a space token CLIP
/// never trained on between every two words (issues #377,
/// #404).
fn pretokenize(text: &str) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let chars: Vec<char> = text.chars().collect();
    let mut pieces: Vec<String> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '\'' {
            let contraction = CONTRACTIONS.iter().find(|c| {
                let n = c.chars().count();
                chars.len() - i >= n && c.chars().eq(chars[i..i + n].iter().copied())
            });
            if let Some(c) = contraction {
                pieces.push(c.to_string());
                i += c.chars().count();
                continue;
            }
        }
        let start = i;
        if ch.is_alphabetic() {
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
        } else if ch.is_numeric() {
            i += 1;
        } else {
            while i < chars.len()
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_numeric()
            {
                i += 1;
            }
        }
        pieces.push(chars[start..i].iter().collect());
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
                          "a": 13, "b": 14, "ab</w>": 15, "Ġ": 16},
                "merges": ["a b</w>"]}}"#,
        )
        .unwrap()
    }

    #[test]
    fn encode_wraps_pads_and_masks() {
        let t = toy();
        let (ids, mask) = t.encode("").unwrap();
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
        let (ids, _) = t.encode("ab").unwrap();
        assert_eq!(&ids[..3], &[0, 15, 1], "a + b</w> merge to ab</w>");
        // No merge across pieces, and the space between them
        // is dropped (issue #404): neither the bare space
        // byte (16) nor the end-of-word space (12) appears.
        let (ids, _) = t.encode("a b").unwrap();
        assert_eq!(&ids[..4], &[0, 10, 11, 1]);
        // Leading and trailing spaces are dropped too.
        let (ids, _) = t.encode(" a ").unwrap();
        assert_eq!(&ids[..3], &[0, 10, 1]);
    }

    #[test]
    fn malformed_exports_fail_at_load() {
        // No merges at all: BPE would degenerate to single
        // characters instead of erroring.
        let err = ClipTokenizer::from_json(
            r#"{"model": {"type": "BPE",
                "vocab": {"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2}}}"#,
        )
        .unwrap_err();
        assert!(err.contains("no merges"), "{err}");
        // A merge whose product is not a vocab entry: the
        // product would vanish at encode time.
        let err = ClipTokenizer::from_json(
            r#"{"model": {"type": "BPE",
                "vocab": {"<|startoftext|>": 0, "<|endoftext|>": 1,
                          "a</w>": 2, "b</w>": 3},
                "merges": ["a b</w>"]}}"#,
        )
        .unwrap_err();
        assert!(err.contains("not in the vocab"), "{err}");
        // An id past the u32 range would truncate on cast.
        let err = ClipTokenizer::from_json(
            r#"{"model": {"type": "BPE",
                "vocab": {"<|startoftext|>": 0, "<|endoftext|>": 1,
                          "a</w>": 4294967296},
                "merges": []}}"#,
        )
        .unwrap_err();
        assert!(err.contains("u32 range"), "{err}");
    }

    #[test]
    fn a_symbol_missing_from_the_vocab_is_an_encode_error() {
        // The toy vocab knows a and b only: encoding "c"
        // must fail loudly, not drop the symbol and shift
        // every later position.
        let t = toy();
        let err = t.encode("c").unwrap_err();
        assert!(err.contains("missing from the tokenizer vocab"), "{err}");
    }

    #[test]
    fn encode_lowercases_and_collapses_whitespace() {
        let t = toy();
        let (ids, _) = t.encode("A   B").unwrap();
        assert_eq!(&ids[..4], &[0, 10, 11, 1]);
        let (ids, _) = t.encode("\tA\n\u{3000}B ").unwrap();
        assert_eq!(&ids[..4], &[0, 10, 11, 1]);
    }

    #[test]
    fn pretokenize_splits_like_the_export() {
        // Whitespace separates pieces and is dropped, never
        // a piece or a prefix of its own (issue #404).
        assert_eq!(pretokenize("kite surfing"), ["kite", "surfing"]);
        assert_eq!(pretokenize("top 10"), ["top", "1", "0"]);
        assert_eq!(pretokenize("it's me"), ["it", "'s", "me"]);
        assert_eq!(pretokenize("sun-set!"), ["sun", "-", "set", "!"]);
        assert_eq!(pretokenize("kite "), ["kite"]);
        assert_eq!(pretokenize("a  b"), ["a", "b"]);
        assert_eq!(pretokenize(" kite"), ["kite"]);
        assert_eq!(pretokenize("dog, cat"), ["dog", ",", "cat"]);
        // A contraction only where the pattern has one; any
        // other apostrophe joins its punctuation run.
        assert_eq!(
            pretokenize("'twas o'clock"),
            ["'t", "was", "o", "'", "clock"]
        );
        assert_eq!(pretokenize("!'s"), ["!'", "s"]);
        assert_eq!(pretokenize("rock'n'roll"), ["rock", "'", "n", "'", "roll"]);
        assert_eq!(pretokenize("'"), ["'"]);
        assert!(pretokenize("").is_empty());
        assert!(pretokenize(" \t\n").is_empty());
    }

    /// Phrases with their token ids from the reference CLIP
    /// tokenizer (Hugging Face `tokenizers` 0.23.3 on the
    /// 49,408-entry vocabulary, matching open_clip's
    /// `SimpleTokenizer`), not from this implementation. The
    /// pre-#404 splitter produced, for the first,
    /// `[49406, 320, 220, 1125, 220, 539, 220, 320, 220, 1929, 49407]`:
    /// a bare space token (220) before every word but the
    /// first.
    const CLIP_REFERENCE_IDS: [(&str, &[i64]); 9] = [
        (
            "a photo of a dog",
            &[49406, 320, 1125, 539, 320, 1929, 49407],
        ),
        (
            "aerial view of a beach",
            &[49406, 12440, 1093, 539, 320, 2117, 49407],
        ),
        ("kite surfing", &[49406, 19867, 15762, 49407]),
        ("top 10", &[49406, 1253, 272, 271, 49407]),
        ("it's me", &[49406, 585, 568, 614, 49407]),
        ("sun-set!", &[49406, 2176, 268, 1167, 256, 49407]),
        ("dog", &[49406, 1929, 49407]),
        ("a dog, a cat", &[49406, 320, 1929, 267, 320, 2368, 49407]),
        (
            "  A\tPhoto   of a DOG\n",
            &[49406, 320, 1125, 539, 320, 1929, 49407],
        ),
    ];

    fn assert_reference_ids(t: &ClipTokenizer) {
        for (phrase, expected) in CLIP_REFERENCE_IDS {
            let (ids, mask) = t.encode(phrase).unwrap();
            let n = mask.iter().filter(|&&m| m == 1).count();
            assert_eq!(&ids[..n], expected, "{phrase:?}");
        }
    }

    /// The fixture is an excerpt of OpenAI CLIP's BPE
    /// vocabulary (`bpe_simple_vocab_16e6`, MIT License,
    /// Copyright (c) 2021 OpenAI): the 49 entries with their
    /// real ids, and the 39 merges in their real order, that
    /// the phrases above touch. Every merge the full
    /// vocabulary applies to these phrases is present, so BPE
    /// on the excerpt takes the same steps as on the whole
    /// file. It also carries the bare (220) and end-of-word
    /// (476) space tokens, so a splitter that emits spaces
    /// fails here with those ids rather than on a lookup.
    #[test]
    fn real_clip_ids_from_a_vocab_excerpt() {
        let t = ClipTokenizer::from_json(include_str!(
            "../tests/fixtures/clip-tokenizer-excerpt.json"
        ))
        .unwrap();
        assert_reference_ids(&t);
    }

    /// The same ids against a full CLIP `tokenizer.json`.
    /// Ignored: needs the file, CRABCROP_TEST_CLIP_TOKENIZER=...
    /// (the ai-models workflow provides it).
    #[test]
    #[ignore = "needs a CLIP tokenizer.json (env-configured)"]
    fn real_clip_ids_from_the_full_tokenizer() {
        let Some(path) = crate::ai::env_or_skip("CRABCROP_TEST_CLIP_TOKENIZER") else {
            return;
        };
        assert_reference_ids(&ClipTokenizer::load(&path).unwrap());
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
