//! The bounds on reading a manifest store, for meta's
//! summary of an asset and for the status a signed output
//! records about its parent: the store's size, its layout,
//! how many manifests it holds, and whether any of them is
//! compressed. A store past any of them is not read, as a
//! store that does not parse is not.

use super::*;

/// JUMBF types: a manifest store, a compressed manifest, and
/// one no reader knows, whose manifest a reader skips.
const STORE_TYPE: &str = "6332706100110010800000AA00389B71";
const COMPRESSED_TYPE: &str = "6332636D00110010800000AA00389B71";
const UNKNOWN_TYPE: &str = "7A7A7A7A00110010800000AA00389B71";

/// The `uuid` box type an ISO BMFF file carries its manifest
/// store in.
const C2PA_BOX: [u8; 16] = [
    0xD8, 0xFE, 0xC3, 0xD6, 0x1B, 0x0E, 0x48, 0x3C, 0x92, 0x97, 0x58, 0x28, 0x87, 0x7E, 0xC4, 0x81,
];

/// A box of type `kind` around `payload`.
fn bmff_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(8 + payload.len()).expect("the box fits a 32-bit size");
    let mut out = size.to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

/// A JUMBF superbox of type `uuid` (32 hex digits), labelled
/// `label`, around `children`.
fn superbox(uuid: &str, label: &str, children: &[Vec<u8>]) -> Vec<u8> {
    let mut description: Vec<u8> = (0..16)
        .map(|i| u8::from_str_radix(&uuid[2 * i..2 * i + 2], 16).expect("a hex type"))
        .collect();
    // Requestable and labelled, as C2PA requires.
    description.push(0x03);
    description.extend_from_slice(label.as_bytes());
    description.push(0);
    let mut payload = bmff_box(b"jumd", &description);
    for child in children {
        payload.extend_from_slice(child);
    }
    bmff_box(b"jumb", &payload)
}

/// A manifest of a type no reader knows, stored as it is: a
/// reader skips it, inflating nothing.
fn skipped_manifest(n: usize) -> Vec<u8> {
    let label = format!("urn:c2pa:skipped-{n}");
    superbox(UNKNOWN_TYPE, &label, &[bmff_box(b"cbor", &[0xA0])])
}

/// The Brotli stream of `bytes`.
fn brotli_stream(bytes: &[u8]) -> Vec<u8> {
    let params = brotli::enc::BrotliEncoderParams {
        quality: 1,
        ..Default::default()
    };
    let mut stream = Vec::new();
    brotli::BrotliCompress(&mut Cursor::new(bytes), &mut stream, &params)
        .expect("the bytes compress");
    stream
}

/// A compressed manifest: a `brob` box whose stream inflates
/// to exactly `inflated` bytes, a manifest of a type no reader
/// knows.
fn compressed_manifest(n: usize, inflated: usize) -> Vec<u8> {
    compressed_manifest_behind(n, inflated, &[])
}

/// [`compressed_manifest`], with `ahead` before its `brob`
/// box.
fn compressed_manifest_behind(n: usize, inflated: usize, ahead: &[Vec<u8>]) -> Vec<u8> {
    let label = format!("urn:c2pa:compressed-{n}");
    let filler = inflated
        .checked_sub(superbox(UNKNOWN_TYPE, &label, &[]).len() + 8)
        .expect("room for the filler box");
    let content = superbox(UNKNOWN_TYPE, &label, &[bmff_box(b"zzzz", &vec![0; filler])]);
    assert_eq!(content.len(), inflated);
    let mut boxes = ahead.to_vec();
    boxes.push(bmff_box(b"brob", &brotli_stream(&content)));
    superbox(COMPRESSED_TYPE, &label, &boxes)
}

/// `asset`, a `format` asset carrying a manifest store, with
/// `extra` in its store ahead of the manifests already there,
/// so the active manifest (the last) stays the asset's own.
/// That manifest no longer matches the asset's bytes, since
/// its store moved, so it reads `invalid`; but it reads.
fn with_extra_manifests(asset: &[u8], format: &str, extra: &[Vec<u8>]) -> Vec<u8> {
    let store =
        c2pa::jumbf_io::load_jumbf_from_memory(format, asset).expect("the asset carries a store");
    let description = u32::from_be_bytes(store[8..12].try_into().expect("a box size"));
    let (description, manifests) = store[8..].split_at(description as usize);
    let mut payload = description.to_vec();
    for manifest in extra {
        payload.extend_from_slice(manifest);
    }
    payload.extend_from_slice(manifests);
    c2pa::jumbf_io::save_jumbf_to_memory(format, asset, &bmff_box(b"jumb", &payload))
        .expect("the store is written back")
}

/// The self-test image signed with the test identity: a PNG
/// whose store holds one manifest, which validates.
fn signed_png() -> Vec<u8> {
    test_signer()
        .sign(
            SELF_TEST_PNG,
            "image/png",
            &SignRequest {
                ingredient: None,
                ai_edited: false,
            },
        )
        .expect("signing works")
}

/// `count` compressed manifests of 64 KiB each.
fn compressed(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|n| compressed_manifest(n, 64 * 1024))
        .collect()
}

#[test]
fn a_store_with_a_compressed_manifest_is_not_read() {
    let asset = signed_png();
    // A manifest stored as it is: the store is read (its own
    // manifest reads `invalid`, since the store moved).
    let without = with_extra_manifests(&asset, "image/png", &[skipped_manifest(0)]);
    let summary = read_manifest(&without, "image/png").expect("the store is read");
    assert_eq!(summary.state, "invalid", "the store moved: {summary:?}");
    assert_eq!(
        summary.manifests, 1,
        "a manifest of a type no reader knows is skipped"
    );
    assert!(source_status(&without, "image/png").is_some());
    // One small compressed manifest instead, and neither meta
    // nor signing reads the store.
    let with = with_extra_manifests(&asset, "image/png", &[compressed_manifest(0, 1024)]);
    assert_eq!(read_manifest(&with, "image/png"), None);
    assert_eq!(source_status(&with, "image/png"), None);
    // The reader fails it as well when it is handed the store
    // directly: it inflates no compressed manifest.
    let store = c2pa::jumbf_io::load_jumbf_from_memory("image/png", &with).expect("a store");
    assert_eq!(store_shape(&store).map(|shape| shape.compressed), Some(1));
    let read = Reader::from_context(reading_context()).with_manifest_data_and_stream(
        &store,
        "image/png",
        Cursor::new(&with),
    );
    assert!(read.is_err(), "the reader read a compressed manifest");
}

#[test]
fn a_store_with_more_manifests_than_the_bound_is_not_read() {
    let asset = signed_png();
    let skipped = |count: usize| -> Vec<Vec<u8>> { (0..count).map(skipped_manifest).collect() };
    // The asset's own manifest is one of the store's.
    let within = with_extra_manifests(&asset, "image/png", &skipped(MAX_STORE_MANIFESTS - 1));
    let summary = read_manifest(&within, "image/png").expect("a store within the bound is read");
    assert_eq!(summary.manifests, 1, "{summary:?}");
    assert!(source_status(&within, "image/png").is_some());
    let over = with_extra_manifests(&asset, "image/png", &skipped(MAX_STORE_MANIFESTS));
    assert_eq!(read_manifest(&over, "image/png"), None);
    assert_eq!(source_status(&over, "image/png"), None);
}

#[test]
fn meta_reads_a_store_no_larger_than_signing_does() {
    // Meta's summary of a source is held to the size bound
    // the parent's status already was (issue #610).
    let within = test_source_with_credentials(MAX_SOURCE_STORE_BYTES / 2);
    let summary = read_manifest(&within, "image/jpeg").expect("the store is read");
    assert_eq!(summary.state, "valid");
    let over = test_source_with_credentials(MAX_SOURCE_STORE_BYTES + 64 * 1024);
    assert_eq!(read_manifest(&over, "image/jpeg"), None);
    assert_eq!(source_status(&over, "image/jpeg"), None);
}

/// A C2PA `uuid` box whose purpose is `purpose`, holding
/// `store`.
fn c2pa_box(purpose: &str, store: &[u8]) -> Vec<u8> {
    let mut payload = C2PA_BOX.to_vec();
    // Version and flags.
    payload.extend_from_slice(&[0; 4]);
    payload.extend_from_slice(purpose.as_bytes());
    payload.push(0);
    // The offset of the first merkle box: none.
    payload.extend_from_slice(&[0; 8]);
    payload.extend_from_slice(store);
    bmff_box(b"uuid", &payload)
}

/// The HEIC fixture signed with the test identity: its store
/// sits in a top-level C2PA box whose purpose is `manifest`.
fn signed_heic() -> Vec<u8> {
    test_signer()
        .sign(
            include_bytes!("../../tests/fixtures/orient6.heic"),
            "image/heic",
            &SignRequest {
                ingredient: None,
                ai_edited: false,
            },
        )
        .expect("the HEIC signs")
}

/// `signed`, a signed BMFF file, with its store's purpose
/// made `original` and an `update` store holding `extra`
/// after it: a pair the SDK merges into one store as it finds
/// it, parsing both on the way.
fn as_an_update_pair(signed: &[u8], extra: &[Vec<u8>]) -> Vec<u8> {
    let mut marker = C2PA_BOX.to_vec();
    marker.extend_from_slice(&[0; 4]);
    marker.extend_from_slice(b"manifest\0");
    let at = signed
        .windows(marker.len())
        .position(|w| w == marker)
        .expect("the file carries a C2PA box");
    let mut pair = signed.to_vec();
    let purpose = at + C2PA_BOX.len() + 4;
    pair[purpose..purpose + 8].copy_from_slice(b"original");
    pair.extend(c2pa_box("update", &superbox(STORE_TYPE, "c2pa", extra)));
    pair
}

#[test]
fn a_bmff_store_that_comes_as_an_update_pair_is_not_read() {
    // The SDK merges an original and update store inside the
    // call that finds the store, before anything here can
    // measure either, and under its own limits: the pair is
    // not read.
    let signed = signed_heic();
    let summary = read_manifest(&signed, "image/heic").expect("a signed HEIC is read");
    assert_eq!(summary.state, "valid");
    assert!(source_status(&signed, "image/heic").is_some());
    let pair = as_an_update_pair(&signed, &compressed(1));
    assert_eq!(read_manifest(&pair, "image/heic"), None);
    assert_eq!(source_status(&pair, "image/heic"), None);
}

/// See [`super::test_sources_without_and_with_a_compressed_manifest`].
pub(super) fn without_and_with_a_compressed_manifest(
    asset: &[u8],
    format: &str,
) -> (Vec<u8>, Vec<u8>) {
    let signed = test_signer()
        .sign(
            asset,
            format,
            &SignRequest {
                ingredient: None,
                ai_edited: false,
            },
        )
        .expect("signing works");
    (
        with_extra_manifests(&signed, format, &[skipped_manifest(0)]),
        with_extra_manifests(&signed, format, &[compressed_manifest(0, 1024)]),
    )
}

/// A store holding `manifests`.
fn store_of(manifests: &[Vec<u8>]) -> Vec<u8> {
    superbox(STORE_TYPE, "c2pa", manifests)
}

#[test]
fn the_walk_counts_the_manifests_of_a_store() {
    let signed = c2pa::jumbf_io::load_jumbf_from_memory("image/png", &signed_png())
        .expect("the output carries a store");
    assert_eq!(
        store_shape(&signed),
        Some(StoreShape {
            manifests: 1,
            compressed: 0,
        })
    );
    let mixed = store_of(&[
        compressed_manifest(0, 4096),
        skipped_manifest(1),
        compressed_manifest(2, 4096),
    ]);
    let shape = StoreShape {
        manifests: 3,
        compressed: 2,
    };
    assert_eq!(store_shape(&mixed), Some(shape));
    // Bytes after the store's own end are not part of it.
    let mut trailing = mixed.clone();
    trailing.extend_from_slice(b"trailing bytes");
    assert_eq!(store_shape(&trailing), Some(shape));
    assert_eq!(
        store_shape(&store_of(&[])),
        Some(StoreShape {
            manifests: 0,
            compressed: 0,
        })
    );
}

#[test]
fn a_brob_box_anywhere_in_a_manifest_counts_it_as_compressed() {
    // A `brob` box behind a box of another type counts too, and
    // the reader, handed such a store directly, fails it.
    let ahead = [bmff_box(b"zzzz", &[0; 16])];
    let late = compressed_manifest_behind(0, 4096, &ahead);
    let shape = store_shape(&store_of(std::slice::from_ref(&late))).expect("the store tiles");
    assert_eq!(shape.compressed, 1);
    let asset = with_extra_manifests(&signed_png(), "image/png", &[late]);
    assert_eq!(read_manifest(&asset, "image/png"), None);
    let store = c2pa::jumbf_io::load_jumbf_from_memory("image/png", &asset).expect("a store");
    let read = Reader::from_context(reading_context()).with_manifest_data_and_stream(
        &store,
        "image/png",
        Cursor::new(&asset),
    );
    assert!(read.is_err(), "the reader read a compressed manifest");
}

#[test]
fn a_store_whose_boxes_do_not_tile_is_not_walked() {
    let manifest = skipped_manifest(0);
    let store = store_of(std::slice::from_ref(&manifest));
    assert!(store_shape(&store).is_some());
    // The offset of the manifest's size, and of the size of its
    // first box (its description).
    let at = store.len() - manifest.len();
    let inner = at + 8;
    let with_size = |at: usize, size: u32| {
        let mut bytes = store.clone();
        bytes[at..at + 4].copy_from_slice(&size.to_be_bytes());
        bytes
    };
    let manifest_len = manifest.len() as u32;
    for (case, bytes) in [
        ("a manifest smaller than its header", with_size(at, 4)),
        ("a manifest of size 0", with_size(at, 0)),
        ("a manifest with a 64-bit size", with_size(at, 1)),
        ("a manifest past the store", with_size(at, manifest_len + 1)),
        (
            "a manifest short of the store",
            with_size(at, manifest_len - 1),
        ),
        ("a box past its manifest", with_size(inner, manifest_len)),
        ("a store cut short", store[..store.len() - 1].to_vec()),
        (
            "a store past its bytes",
            with_size(0, store.len() as u32 + 8),
        ),
    ] {
        assert_eq!(store_shape(&bytes), None, "{case}");
    }
    let mut not_a_superbox = store.clone();
    not_a_superbox[4..8].copy_from_slice(b"junk");
    assert_eq!(store_shape(&not_a_superbox), None);
    let mut no_description = store.clone();
    no_description[12..16].copy_from_slice(b"junk");
    assert_eq!(store_shape(&no_description), None);
    assert_eq!(store_shape(&[]), None);
}

#[test]
fn the_bmff_walk_finds_an_update_pair_and_nothing_else() {
    let signed = signed_heic();
    assert!(!bmff_store_pair(&signed), "a store kept whole");
    assert!(bmff_store_pair(&as_an_update_pair(&signed, &[])));
    let store = store_of(&[]);
    for purpose in ["original", "update"] {
        let mut file = signed.clone();
        file.extend(c2pa_box(purpose, &store));
        assert!(bmff_store_pair(&file), "{purpose}");
    }
    // Neither a merkle box nor a `uuid` box of another type is
    // one.
    let mut merkle = signed.clone();
    merkle.extend(c2pa_box("merkle", &store));
    assert!(!bmff_store_pair(&merkle));
    let mut other = signed.clone();
    let mut payload = [0x11; 16].to_vec();
    payload.extend_from_slice(b"\0\0\0\0original\0");
    other.extend(bmff_box(b"uuid", &payload));
    assert!(!bmff_store_pair(&other));
    // A box with a 64-bit size is followed like any other.
    let pair = c2pa_box("original", &store);
    let mut large = signed.clone();
    large.extend_from_slice(&1u32.to_be_bytes());
    large.extend_from_slice(b"uuid");
    large.extend_from_slice(&(pair.len() as u64 + 8).to_be_bytes());
    large.extend_from_slice(&pair[8..]);
    assert!(bmff_store_pair(&large));
    // A box whose size does not cover its header counts as one:
    // the SDK would read on from inside it, out of step with
    // this walk.
    let mut undersized = signed.clone();
    undersized.extend_from_slice(&4u32.to_be_bytes());
    undersized.extend_from_slice(b"free");
    assert!(bmff_store_pair(&undersized));
    // Assets that are not ISO BMFF end the walk at once.
    for asset in [signed_png(), test_source_with_credentials(0), Vec::new()] {
        assert!(!bmff_store_pair(&asset));
    }
}

#[test]
fn the_walks_take_any_truncation_or_flipped_byte() {
    // Both walks read untrusted bytes: cut short anywhere, or
    // with any one byte flipped, they answer without panicking.
    let store = store_of(&[compressed_manifest(0, 4096), skipped_manifest(1)]);
    let mut file = bmff_box(b"ftyp", b"heic\0\0\0\0mif1heic");
    file.extend(c2pa_box("manifest", &store));
    file.extend(bmff_box(b"mdat", &[0; 64]));
    for bytes in [&store, &file] {
        for len in 0..bytes.len() {
            store_shape(&bytes[..len]);
            bmff_store_pair(&bytes[..len]);
        }
        for at in 0..bytes.len() {
            let mut flipped = bytes.clone();
            flipped[at] ^= 0xFF;
            store_shape(&flipped);
            bmff_store_pair(&flipped);
        }
    }
    assert!(!bmff_store_pair(&file));
}

/// A box whose type is four zero bytes.
fn zero_type_box() -> Vec<u8> {
    bmff_box(&[0; 4], &[])
}

/// A box of type `kind` around `payload`, its size written in
/// 64 bits (a size field of 1).
fn large_size_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = 1u32.to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(&(16 + payload.len() as u64).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// A box of type `kind` around `payload`, its size field 0.
fn zero_size_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = 0u32.to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

/// A manifest of a type no reader knows, labelled `label`,
/// holding `boxes` after its description box.
fn manifest_holding(label: &str, boxes: &[Vec<u8>]) -> Vec<u8> {
    superbox(UNKNOWN_TYPE, label, boxes)
}

/// `boxes`, then one manifest more than a read takes.
fn ahead_of_manifests(boxes: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut all = boxes.to_vec();
    all.extend((0..=MAX_STORE_MANIFESTS).map(skipped_manifest));
    all
}

/// The payload of a store's description box: its type,
/// `toggles`, its label, and `rest` after the label.
fn store_description(toggles: u8, rest: &[u8]) -> Vec<u8> {
    let mut description: Vec<u8> = (0..16)
        .map(|i| u8::from_str_radix(&STORE_TYPE[2 * i..2 * i + 2], 16).expect("a hex type"))
        .collect();
    description.push(toggles);
    description.extend_from_slice(b"c2pa\0");
    description.extend_from_slice(rest);
    description
}

/// The manifests of the store `asset` carries, as its store
/// holds them after its description box.
fn manifests_of(asset: &[u8]) -> Vec<u8> {
    let store = c2pa::jumbf_io::load_jumbf_from_memory("image/png", asset).expect("a store");
    let description = u32::from_be_bytes(store[8..12].try_into().expect("a box size"));
    store[8 + description as usize..].to_vec()
}

/// A store whose description box holds `description` and
/// whose boxes after it are `boxes`.
fn store_with(description: &[u8], boxes: &[u8]) -> Vec<u8> {
    bmff_box(
        b"jumb",
        &[bmff_box(b"jumd", description), boxes.to_vec()].concat(),
    )
}

#[test]
fn a_box_whose_type_is_zero_is_refused_at_any_depth() {
    // A box whose type is four zero bytes, in a manifest or in
    // any superbox below one: the walk refuses the store, and
    // neither meta nor signing reads it.
    let asset = signed_png();
    let zero = |count: usize| vec![zero_type_box(); count];
    for (case, manifest) in [
        (
            "in a manifest",
            manifest_holding("urn:c2pa:outer", &ahead_of_manifests(&zero(1))),
        ),
        (
            "a level below",
            manifest_holding(
                "urn:c2pa:outer",
                &[manifest_holding(
                    "urn:c2pa:inner",
                    &ahead_of_manifests(&zero(2)),
                )],
            ),
        ),
        (
            "two levels below",
            manifest_holding(
                "urn:c2pa:outer",
                &[manifest_holding(
                    "urn:c2pa:middle",
                    &[manifest_holding(
                        "urn:c2pa:inner",
                        &ahead_of_manifests(&zero(3)),
                    )],
                )],
            ),
        ),
    ] {
        let source = with_extra_manifests(&asset, "image/png", &[manifest]);
        let store = c2pa::jumbf_io::load_jumbf_from_memory("image/png", &source).expect("a store");
        assert!(store.len() <= MAX_SOURCE_STORE_BYTES, "{case}");
        assert_eq!(store_shape(&store), None, "{case}");
        assert_eq!(read_manifest(&source, "image/png"), None, "{case}");
        assert_eq!(source_status(&source, "image/png"), None, "{case}");
    }
}

#[test]
fn a_box_with_a_size_field_of_0_or_1_is_refused_at_any_depth() {
    // A size field of 0 or 1 (a size written in 64 bits), in a
    // manifest or in any superbox below one: the walk refuses
    // the store, and neither meta nor signing reads it.
    let asset = signed_png();
    let zeros = [zero_type_box(), zero_type_box()].concat();
    let below = |boxes: &[Vec<u8>]| {
        manifest_holding(
            "urn:c2pa:outer",
            &[manifest_holding("urn:c2pa:inner", boxes)],
        )
    };
    let after = ahead_of_manifests(&[]).concat();
    for (case, manifest) in [
        (
            "a 64-bit size below a manifest",
            below(&[large_size_box(
                b"free",
                &[zeros.clone(), after.clone()].concat(),
            )]),
        ),
        (
            "a size of 0 below a manifest",
            below(&ahead_of_manifests(&[
                zero_size_box(b"cbor", &[]),
                zero_type_box(),
                zero_type_box(),
            ])),
        ),
        (
            "a 64-bit size in a manifest",
            manifest_holding("urn:c2pa:outer", &[large_size_box(b"free", &[0; 16])]),
        ),
    ] {
        let source = with_extra_manifests(&asset, "image/png", &[manifest]);
        let store = c2pa::jumbf_io::load_jumbf_from_memory("image/png", &source).expect("a store");
        assert_eq!(store_shape(&store), None, "{case}");
        assert_eq!(read_manifest(&source, "image/png"), None, "{case}");
        assert_eq!(source_status(&source, "image/png"), None, "{case}");
    }
}

#[test]
fn a_description_box_is_walked_to_its_exact_end() {
    let asset = signed_png();
    let manifests = manifests_of(&asset);
    // A label, then the id, the signature and the private salt
    // box its toggles announce: walked, and the store is read.
    let salt = bmff_box(b"c2sh", &[0x5A; 16]);
    for (case, toggles, rest) in [
        ("a label", 0x03, Vec::new()),
        ("an id", 0x07, vec![0; 4]),
        ("an id and a signature", 0x0F, vec![0; 36]),
        (
            "an id, a signature and a salt",
            0x1F,
            [&[0; 36][..], &salt].concat(),
        ),
    ] {
        let store = store_with(&store_description(toggles, &rest), &manifests);
        assert!(store_shape(&store).is_some(), "{case}");
        let source = c2pa::jumbf_io::save_jumbf_to_memory("image/png", &asset, &store)
            .expect("the store is written");
        assert!(read_manifest(&source, "image/png").is_some(), "{case}");
    }
    // Anything else is refused.
    let mut no_end: Vec<u8> = store_description(0x03, &[]);
    no_end.pop();
    let mut empty_label = store_description(0x03, &[]);
    empty_label.truncate(17);
    empty_label.push(0);
    for (case, description) in [
        ("not labelled", store_description(0x01, &[])),
        ("a label with no end", no_end),
        ("an empty label", empty_label),
        ("an id cut short", store_description(0x07, &[0; 2])),
        ("bytes after the label", store_description(0x03, &[0; 3])),
        (
            "a private box that is not a salt",
            store_description(0x13, &bmff_box(b"zzzz", &[0; 16])),
        ),
        (
            "a salt short of the end",
            store_description(0x13, &[salt.clone(), vec![0; 4]].concat()),
        ),
        (
            "a salt with a 64-bit size",
            store_description(0x13, &large_size_box(b"c2sh", &[0x5A; 16])),
        ),
    ] {
        assert_eq!(
            store_shape(&store_with(&description, &manifests)),
            None,
            "{case}"
        );
    }
}

#[test]
fn a_superbox_deeper_than_the_reader_goes_is_refused() {
    // The store sits at depth 0: a superbox at depth 31 is
    // walked, one at depth 32 is not.
    let nested = |depth: usize| {
        let mut boxes = vec![bmff_box(b"cbor", &[0xA0])];
        for level in (1..=depth).rev() {
            boxes = vec![manifest_holding(&format!("urn:c2pa:level-{level}"), &boxes)];
        }
        store_of(&boxes)
    };
    assert!(store_shape(&nested(MAX_SUPERBOX_DEPTH - 1)).is_some());
    assert_eq!(store_shape(&nested(MAX_SUPERBOX_DEPTH)), None);
}

#[test]
fn a_store_the_walk_refuses_is_not_read_with_any_contents() {
    // Each refused layout, with compressed manifests or more
    // manifests than a read takes after it: neither meta nor
    // signing reads the store.
    let asset = signed_png();
    let manifests = manifests_of(&asset);
    let compressed_after = |boxes: &[Vec<u8>]| {
        let mut all = boxes.to_vec();
        all.extend(compressed(2));
        all
    };
    let zeros = [zero_type_box(), zero_type_box()].concat();
    let below = |boxes: &[Vec<u8>]| {
        manifest_holding(
            "urn:c2pa:outer",
            &[manifest_holding("urn:c2pa:inner", boxes)],
        )
    };
    let extra = |manifest: Vec<u8>| with_extra_manifests(&asset, "image/png", &[manifest]);
    let salted = {
        let description = store_description(0x13, &large_size_box(b"c2sh", &[0x5A; 16]));
        let boxes = [compressed(2).concat(), manifests.clone()].concat();
        let store = store_with(&description, &bmff_box(b"free", &boxes));
        c2pa::jumbf_io::save_jumbf_to_memory("image/png", &asset, &store)
            .expect("the store is written")
    };
    for (case, source) in [
        (
            "a zero type, then compressed manifests",
            extra(manifest_holding(
                "urn:c2pa:outer",
                &compressed_after(&[zero_type_box()]),
            )),
        ),
        (
            "a zero type, then manifests",
            extra(manifest_holding(
                "urn:c2pa:outer",
                &ahead_of_manifests(&[zero_type_box()]),
            )),
        ),
        (
            "zero types a level below, then compressed manifests",
            extra(below(&compressed_after(&[
                zero_type_box(),
                zero_type_box(),
            ]))),
        ),
        (
            "a 64-bit size a level below, then compressed manifests",
            extra(below(&[large_size_box(
                b"free",
                &[zeros.clone(), compressed(2).concat()].concat(),
            )])),
        ),
        ("a salt with a 64-bit size", salted),
    ] {
        assert_eq!(read_manifest(&source, "image/png"), None, "meta: {case}");
        assert_eq!(source_status(&source, "image/png"), None, "signing: {case}");
    }
}
