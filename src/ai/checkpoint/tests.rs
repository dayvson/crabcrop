use crate::ai::{AestheticHead, AiError};
use crate::test_support::{ScratchDir, scratch_dir};

/// Write bytes to a fresh temp file and return the
/// scratch directory it lives in and its path (issue
/// #526).
fn temp_file(tag: &str, contents: &[u8]) -> (ScratchDir, String) {
    let dir = scratch_dir(&format!("embedding-{tag}"));
    let path = dir
        .join(format!("{tag}.bin"))
        .to_string_lossy()
        .into_owned();
    std::fs::write(&path, contents).unwrap();
    (dir, path)
}

/// A path that does not exist: a name inside a scratch
/// directory that is removed again before this returns.
fn missing_path(tag: &str) -> String {
    scratch_dir(&format!("embedding-missing-{tag}"))
        .join(format!("{tag}.pth"))
        .to_string_lossy()
        .into_owned()
}

fn expect_load<T>(result: Result<T, AiError>) -> String {
    match result {
        Err(AiError::Load(message)) => message,
        Err(other) => panic!("expected a load error, got {other}"),
        Ok(_) => panic!("expected a load error, but the load succeeded"),
    }
}

/// A pickle string (BINUNICODE) for the synthetic
/// head saves below.
fn pkl_str(out: &mut Vec<u8>, s: &str) {
    out.push(b'X');
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// The pickle `torch.save` writes for a linear
/// head's state dict, reduced to the shape the
/// loader reads: an ordered dict holding a weight
/// tensor over storage "0" and a bias tensor over
/// storage "1", closed by the metadata BUILD tail.
/// Memo indices follow the real files.
fn head_pickle(dim: usize, bias_count: usize) -> Vec<u8> {
    let mut p = Vec::new();
    let binint2 = |p: &mut Vec<u8>, v: usize| {
        p.push(b'M');
        p.extend_from_slice(&(v as u16).to_le_bytes());
    };
    // PROTO 2, then the root OrderedDict.
    p.extend_from_slice(b"\x80\x02");
    p.extend_from_slice(b"ccollections\nOrderedDict\n");
    p.extend_from_slice(b"q\x00)Rq\x01");
    p.push(b'(');
    // The weight entry.
    pkl_str(&mut p, "weight");
    p.extend_from_slice(b"q\x02");
    p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
    p.extend_from_slice(b"q\x03");
    p.push(b'(');
    p.push(b'(');
    pkl_str(&mut p, "storage");
    p.extend_from_slice(b"q\x04");
    p.extend_from_slice(b"ctorch\nFloatStorage\n");
    p.extend_from_slice(b"q\x05");
    pkl_str(&mut p, "0");
    p.extend_from_slice(b"q\x06");
    pkl_str(&mut p, "cpu");
    p.extend_from_slice(b"q\x07");
    binint2(&mut p, dim);
    p.extend_from_slice(b"tq\x08");
    p.push(b'Q'); // BINPERSID: the storage tuple
    p.extend_from_slice(b"K\x00"); // offset 0
    p.extend_from_slice(b"K\x01"); // size (1, dim)
    binint2(&mut p, dim);
    p.extend_from_slice(b"\x86q\x09");
    binint2(&mut p, dim); // stride (dim, 1)
    p.extend_from_slice(b"K\x01\x86q\x0a");
    p.extend_from_slice(b"\x89"); // requires_grad: false
    p.extend_from_slice(b"h\x00)"); // hooks: an empty OrderedDict, reduced next
    p.extend_from_slice(b"Rq\x0btq\x0cRq\x0d");
    // The bias entry, reusing the memoized globals.
    pkl_str(&mut p, "bias");
    p.extend_from_slice(b"q\x0e");
    p.extend_from_slice(b"h\x03");
    p.push(b'(');
    p.push(b'(');
    p.extend_from_slice(b"h\x04h\x05");
    pkl_str(&mut p, "1");
    p.extend_from_slice(b"q\x0f");
    p.extend_from_slice(b"h\x07");
    p.push(b'K');
    p.push(bias_count as u8);
    p.extend_from_slice(b"tq\x10");
    p.push(b'Q');
    p.extend_from_slice(b"K\x00"); // offset 0
    p.push(b'K');
    p.push(bias_count as u8); // size (bias_count,)
    p.extend_from_slice(b"\x85q\x11");
    p.extend_from_slice(b"K\x01"); // stride (1,)
    p.extend_from_slice(b"\x85q\x12");
    p.extend_from_slice(b"\x89");
    p.extend_from_slice(b"h\x00)");
    p.extend_from_slice(b"Rq\x13tq\x14Rq\x15");
    p.push(b'u'); // SETITEMS into the root dict
    // The BUILD tail: a state dict whose only
    // entry is an empty _metadata dict.
    p.push(b'}');
    p.extend_from_slice(b"q\x16");
    pkl_str(&mut p, "_metadata");
    p.extend_from_slice(b"q\x17");
    p.extend_from_slice(b"h\x00)Rq\x18");
    p.push(b's');
    p.push(b'b');
    p.push(b'.');
    p
}

/// Wrap a pickle and its storages in a stored-entry
/// zip archive of the shape `torch.save` writes.
fn head_archive(pkl: &[u8], storages: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut central: Vec<u8> = Vec::new();
    let mut entries: Vec<(&str, &[u8])> = vec![("archive/data.pkl", pkl)];
    entries.extend_from_slice(storages);
    entries.push(("archive/version", b"3\n"));
    for (name, data) in &entries {
        let local_at = out.len() as u32;
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        out.extend_from_slice(&0u32.to_le_bytes()); // time and date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc (the reader ignores it)
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
        central.extend_from_slice(b"PK\x01\x02");
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&local_at.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let cd_at = out.len() as u32;
    let cd_len = central.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_len.to_le_bytes());
    out.extend_from_slice(&cd_at.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

fn floats_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn aesthetic_head_loads_a_synthetic_pth() {
    let weights = [0.5f32, -0.25, 2.0];
    let archive = head_archive(
        &head_pickle(3, 1),
        &[
            ("archive/data/0", &floats_bytes(&weights)),
            ("archive/data/1", &floats_bytes(&[0.75])),
        ],
    );
    let (_head, path) = temp_file("head-ok", &archive);
    let head = AestheticHead::load(&path).unwrap();
    // The identity meta reports is the file name,
    // never the path it was loaded from.
    assert_eq!(
        head.identity(),
        std::path::Path::new(&path)
            .file_name()
            .unwrap()
            .to_string_lossy()
    );
    assert!(!head.identity().contains('/'));
    assert_eq!(head.expected_dim(), 3);
    let score = head.score(&[2.0, 4.0, 0.5]).unwrap();
    assert!((score - 1.75).abs() < 1e-6, "score {score}");
}

#[test]
fn aesthetic_head_load_rejects_garbage_and_missing_files() {
    let (_head, path) = temp_file("head-garbage", b"this is not a zip archive at all");
    let err = expect_load(AestheticHead::load(&path));
    assert!(err.contains("not usable"), "{err}");
    let err = expect_load(AestheticHead::load(&missing_path("head")));
    assert!(err.contains("cannot read the aesthetic head"), "{err}");
}

#[test]
fn aesthetic_head_rejects_an_archive_without_a_pickle() {
    let archive = head_archive(b"", &[]);
    // An empty pickle payload is not a state dict;
    // the failure must be a load error either way.
    let (_head, path) = temp_file("head-empty-pkl", &archive);
    let err = expect_load(AestheticHead::load(&path));
    assert!(err.contains("not usable"), "{err}");
}

#[test]
fn aesthetic_head_rejects_a_multi_value_bias() {
    let archive = head_archive(
        &head_pickle(3, 2),
        &[
            ("archive/data/0", &floats_bytes(&[0.5, -0.25, 2.0])),
            ("archive/data/1", &floats_bytes(&[0.75, 0.5])),
        ],
    );
    let (_head, path) = temp_file("head-wide-bias", &archive);
    let err = expect_load(AestheticHead::load(&path));
    assert!(err.contains("bias is a single value"), "{err}");
}

#[test]
fn aesthetic_head_rejects_a_storage_shorter_than_the_tensor() {
    let archive = head_archive(
        &head_pickle(3, 1),
        &[
            ("archive/data/0", &floats_bytes(&[0.5, -0.25])),
            ("archive/data/1", &floats_bytes(&[0.75])),
        ],
    );
    let (_head, path) = temp_file("head-short-storage", &archive);
    let err = expect_load(AestheticHead::load(&path));
    assert!(err.contains("overruns its storage"), "{err}");
}

#[test]
fn aesthetic_head_rejects_a_tensor_size_that_overflows() {
    // A shape whose element count overflows usize (issue
    // #484) is a load error naming the overflow, never a
    // multiply that panics (debug) or wraps to a small
    // count (release) before the storage bound is checked.
    let pkl = head_pickle(3, 1);
    // The weight's size tuple (1, 3): BININT1 1, BININT2 3,
    // TUPLE2. Swap in (2^32, 2^32) as two LONG1 values.
    let size: &[u8] = b"K\x01M\x03\x00\x86";
    let at: Vec<usize> = (0..pkl.len())
        .filter(|&i| pkl[i..].starts_with(size))
        .collect();
    assert_eq!(at.len(), 1, "the size tuple is unique");
    let huge = [0x8a, 5, 0, 0, 0, 0, 1];
    let mut patched = pkl[..at[0]].to_vec();
    patched.extend_from_slice(&huge);
    patched.extend_from_slice(&huge);
    patched.push(0x86);
    patched.extend_from_slice(&pkl[at[0] + size.len()..]);
    let archive = head_archive(
        &patched,
        &[
            ("archive/data/0", &floats_bytes(&[0.5, -0.25, 2.0])),
            ("archive/data/1", &floats_bytes(&[0.75])),
        ],
    );
    let (_head, path) = temp_file("head-size-overflow", &archive);
    let err = expect_load(AestheticHead::load(&path));
    assert!(err.contains("the tensor size overflows"), "{err}");
}

#[test]
fn aesthetic_head_refuses_memo_copies_and_deep_nesting() {
    // The two shapes from issue #609 are load errors
    // naming the file, reached in bounded time and
    // memory. The load runs on a 256 KiB stack, so a
    // deep recursion anywhere in it would abort the test.
    let mut memo = b"(\x80\x02)q\x00".to_vec();
    for _ in 0..40 {
        memo.extend_from_slice(b"h\x00h\x00\x86q\x00");
    }
    memo.extend_from_slice(b"t.");
    let mut deep = b"\x80\x02)".to_vec();
    deep.resize(1024 * 1024 - 1, 0x85);
    deep.push(b'.');
    for (tag, pkl, want) in [
        ("head-memo", memo, "values and string bytes"),
        ("head-deep", deep, "levels deep"),
    ] {
        let (_head, path) = temp_file(tag, &head_archive(&pkl, &[]));
        let err = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || expect_load(AestheticHead::load(&path)))
            .unwrap()
            .join()
            .unwrap();
        assert!(
            err.contains("is not usable") && err.contains(want),
            "{tag}: {err}"
        );
    }
}
