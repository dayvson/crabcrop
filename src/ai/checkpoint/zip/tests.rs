use super::*;
use crate::ai::checkpoint::HEAD_BUDGET;
use std::io::Write;

/// A zip archive of deflated entries, each given as its
/// name, its plain bytes, and the plain size its
/// directory record declares (the real size, unless a
/// test wants the directory to lie).
fn deflated_archive(entries: &[(&str, &[u8], u32)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, plain, declared) in entries {
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(plain).unwrap();
        let packed = encoder.finish().unwrap();
        let local_at = out.len() as u32;
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&8u16.to_le_bytes()); // method: deflate
        out.extend_from_slice(&0u32.to_le_bytes()); // time and date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc (the reader ignores it)
        out.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        out.extend_from_slice(&declared.to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&packed);
        central.extend_from_slice(b"PK\x01\x02");
        central.extend_from_slice(&20u16.to_le_bytes()); // made by
        central.extend_from_slice(&20u16.to_le_bytes()); // version needed
        central.extend_from_slice(&0u16.to_le_bytes()); // flags
        central.extend_from_slice(&8u16.to_le_bytes()); // method: deflate
        central.extend_from_slice(&0u32.to_le_bytes()); // time and date
        central.extend_from_slice(&0u32.to_le_bytes()); // crc
        central.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        central.extend_from_slice(&declared.to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0; 8]); // extra, comment, disk, internal attributes
        central.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        central.extend_from_slice(&local_at.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let cd_at = out.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&[0; 4]); // disk numbers
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_at.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment
    out
}

#[test]
fn deflated_entries_within_the_budget_inflate() {
    let archive = deflated_archive(&[("a", &[7; 1000], 1000), ("b", b"bias", 4)]);
    let entries = read_zip_entries(&archive).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].name, "a");
    assert_eq!(entries[0].data, vec![7; 1000]);
    assert_eq!(entries[1].data, b"bias");
}

#[test]
fn an_entry_larger_than_the_head_budget_is_refused() {
    // Issue #484: the file is capped at the head budget,
    // but a deflated entry used to inflate up to the size
    // its directory declares (up to 4 GiB). These 16 MiB
    // and one byte of zeros pack into well under 1% of
    // the budget, so the file passes its own cap, and
    // they inflated in full.
    let plain = vec![0; HEAD_BUDGET + 1];
    let archive = deflated_archive(&[("archive/data/0", &plain, plain.len() as u32)]);
    assert!(archive.len() * 100 < HEAD_BUDGET, "{} bytes", archive.len());
    let err = read_zip_entries(&archive)
        .err()
        .expect("an over-budget entry must be refused");
    assert!(
        err.contains("archive/data/0") && err.contains("budget"),
        "{err}"
    );
}

#[test]
fn the_entries_share_one_head_budget() {
    // Two entries that each fit, but not together: the
    // budget covers the whole archive, so a save cannot
    // spread an expansion across entries.
    let half = vec![0; HEAD_BUDGET / 2 + 1];
    let archive = deflated_archive(&[
        ("a", &half, half.len() as u32),
        ("b", &half, half.len() as u32),
    ]);
    let err = read_zip_entries(&archive)
        .err()
        .expect("the archive is over budget");
    assert!(
        err.contains("the entry b") && err.contains("budget"),
        "{err}"
    );
}

#[test]
fn the_budget_is_charged_before_inflating() {
    // A directory that declares more than the budget is
    // refused on the declaration alone: the error is the
    // budget's, not the size mismatch an inflate of this
    // small payload would report.
    let archive = deflated_archive(&[("a", &[1; 10], u32::MAX - 1)]);
    let err = read_zip_entries(&archive)
        .err()
        .expect("an over-budget declaration must be refused");
    assert!(err.contains("budget"), "{err}");
}
