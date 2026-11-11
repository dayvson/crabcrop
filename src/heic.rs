//! HEIC/HEIF source decoding (issue #163): the pure-Rust
//! `oxideav-heif` container + `oxideav-h265` codec, compiled in
//! unconditionally — no cargo feature, no configuration, no
//! external tool. Decode speed is what it is (measured at
//! roughly 3.7x FFmpeg's wall time on large single-stream
//! photos); the trade buys a decoder that ships in every
//! artifact. Only HEVC-coded items are decoded: the container
//! crate also carries AV1, H.264, and H.266 codecs, and a file
//! that would reach any of them is refused before the decoder
//! runs (issue #466).
//!
//! The decoded pixels join the normal pipeline: this module
//! returns the image with the EXIF orientation already applied
//! (from the decoder's EXIF blob, through the engine's own
//! orientation machinery — the container bytes carry no
//! orientation the generic reader can see) plus the
//! [`SourceMeta`] the pipeline's color conversion and metadata
//! preservation consume. Container transforms (`irot`, mirror,
//! crop) are the decoder's own work and are never re-applied.

use image::DynamicImage;

use crate::color::SourceMeta;
use crate::engine::{EngineError, MAX_DECODE_ALLOC, MAX_DECODE_DIMENSION};

/// The decode budget in pixels: the 256 MiB allocation limit
/// expressed as RGBA pixels.
const MAX_DECODE_PIXELS: u64 = MAX_DECODE_ALLOC / 4;

/// The coded item types the HEVC codec handles (issue #466):
/// `hvc1` and `hev1` (parameter sets out of band or in band),
/// `hvt1` (an HEVC tile), and `lhv1` (layered HEVC).
/// Every other coded type (`av01`, `avc1`, `vvc1`, `jpeg`, and
/// anything newer) is refused before decoding, so source bytes
/// never reach the crate's other, pre-1.0 codecs.
const HEVC_ITEM_TYPES: [[u8; 4]; 4] = [
    oxideav_heif::meta::ITEM_TYPE_HVC1,
    oxideav_heif::meta::ITEM_TYPE_HEV1,
    oxideav_heif::meta::ITEM_TYPE_HVT1,
    oxideav_heif::meta::ITEM_TYPE_LHV1,
];

/// How far an H.265 coded picture may run past its item's
/// declared size before the file is rejected as mismatched.
/// Encoders pad the coded picture to a coding-tree-block
/// boundary (64 samples at most) and signal the difference as
/// the conformance window, so legitimate files stay well under
/// this; the audit's attack shape (a 64x64 declaration in front
/// of a 16384x16384 coded picture) is nowhere near it.
const SPS_SIZE_TOLERANCE: u32 = 128;

/// The decoded image's own axes are bounded through the decoder's
/// limits, set from the same constants the image-crate path uses,
/// so a HEIC source cannot allocate past the shared budget:
/// 16384 px per axis and 256 MiB of RGBA pixels.
fn decode_options() -> oxideav_heif::DecodeOptions {
    oxideav_heif::DecodeOptions::default()
        .with_max_width(Some(MAX_DECODE_DIMENSION))
        .with_max_height(Some(MAX_DECODE_DIMENSION))
        .with_max_pixels(Some(MAX_DECODE_PIXELS))
}

fn decode_failed(cause: impl std::fmt::Display) -> EngineError {
    EngineError::Decode(format!("HEIC/HEIF source: {cause}"))
}

/// Enforce the shared decode budget on everything the primary
/// decode is about to decode, before the decoder allocates for
/// any of it (issue #184; audit findings R2-02 and R2-11).
///
/// The decoder's own limit check covers only the primary item's
/// announced output size. Inside a coded decode, though, the
/// H.265 codec allocates its planes from the SPS geometry in
/// the bitstream, which nothing ties to the announced size: a
/// 64x64 ispe in front of a 16384x16384 SPS decodes into
/// roughly 768 MiB of planes per picture, an allocation failure
/// there aborts the process instead of unwinding, and the depth
/// and gain map items decode just as eagerly with no size check
/// at all. So this walks every item the decode can reach (the
/// primary's derivation tree with its alpha and depth
/// auxiliaries, the gain map item, and the primary's `altr`
/// alternatives with their own gain maps, which the decoder
/// falls back to when the primary itself fails) and checks each
/// one three times:
///
/// - its codec: a coded item (or a tiled item's tiles) must be
///   HEVC, through [`check_hevc_coded`] (issue #466), so the
///   SPS checks below cover every coded picture the decode can
///   reach, and no other codec ever sees the bytes;
/// - its announced size against the shared budget, through
///   [`check_announced_size`], and
/// - every HEVC SPS it carries, in its configuration record
///   and in band in its payload, against that announced size,
///   through [`check_coded_geometry`].
///
/// Residual, honestly: the `tili` packed-tile stream is not a
/// plain NAL stream to scan, so a tiled item is covered through
/// its configuration record only.
fn validate_decode_budget(bytes: &[u8]) -> Result<(), EngineError> {
    let file = oxideav_heif::HeifFile::parse_borrowed(bytes).map_err(decode_failed)?;
    check_item_graphs(&file, bytes.len())?;
    let mut seen = std::collections::HashSet::new();
    if let Ok(node) = oxideav_heif::build_primary_graph(&file) {
        validate_node(&file, &node, &mut seen)?;
    }
    let Ok(primary) = file.primary_item() else {
        return Ok(());
    };
    if let Some(gain) = gain_map_node(&file, primary.id) {
        validate_node(&file, &gain, &mut seen)?;
    }
    if let Ok(meta) = file.meta() {
        for group in meta.groups_containing(primary.id, b"altr") {
            for &alt in &group.entity_ids {
                if alt == primary.id {
                    continue;
                }
                if let Ok(node) = oxideav_heif::build_graph(&file, alt) {
                    validate_node(&file, &node, &mut seen)?;
                }
                // The decode of an alternative decodes its gain
                // map too, as it does the primary's.
                if let Some(gain) = gain_map_node(&file, alt) {
                    validate_node(&file, &gain, &mut seen)?;
                }
            }
        }
    }
    Ok(())
}

/// The most work the derivation-graph walks of one decode may
/// do, in the units [`GraphCost`] counts: table entries scanned
/// plus the property, payload, and name bytes read. A grid of
/// 1,024 distinct tiles (the most 256 px tiles the decode
/// budget holds) with a thumbnail and two metadata items costs
/// about 4.5 million, some 15 times less.
const MAX_GRAPH_WORK: u64 = 1 << 26;

/// The most the graphs a decode builds may hold at once, in
/// bytes as [`GraphCost`] estimates them. The same grid of
/// 1,024 tiles, each with the usual four properties, holds
/// about 12 MiB by that estimate, some 11 times less.
const MAX_GRAPH_BYTES: u64 = 128 << 20;

/// Refuse a file whose item graphs would cost more than
/// [`MAX_GRAPH_WORK`] to build, or hold more than
/// [`MAX_GRAPH_BYTES`] once built, before any of them is built.
///
/// oxideav-heif builds a derivation graph by visiting an item
/// once for every reference that reaches it, with no memo, up
/// to 65,536 visits. Each visit scans the file's item,
/// association, and reference tables, looks up every metadata
/// item that describes the item, parses its properties, and
/// copies a derived item's payload. Each visit also adds a node
/// to the graph, holding a copy of the item's entry (its name
/// included), its parsed properties, and the entries of the
/// metadata items that describe it. So an item that many
/// references reach is walked, and held, many times over. The
/// probe, the pre-decode validation, the colour lookup, and the
/// decode itself each build these graphs.
///
/// This works out what the walks will cost from the tables
/// alone: each item's own cost once, summed along the
/// references with a memo, so the estimate is linear in the
/// tables. Where the walk would stop (a reference cycle, a
/// chain deeper than it follows), the estimate stops too.
///
/// The graphs counted are the ones the decode builds, each as
/// many times as the decode holds it at once. The decode tries
/// the primary item first and then, should that fail, each of
/// the primary's `altr` alternatives in turn. Each try builds
/// the item's graph, builds it a second time while holding the
/// first, and then, while holding both, builds the graph of the
/// item's gain map image: the second input of a `tmap` item,
/// whether the item is that `tmap` or its first input. So a try
/// counts the item's graph twice and the gain map's graph once
/// more, with the `tmap` item's payload and properties, which
/// the gain map lookup copies. The tries are added up, though
/// each lets go of its graphs before the next starts.
fn check_item_graphs<D: AsRef<[u8]>>(
    file: &oxideav_heif::HeifFile<D>,
    source_len: usize,
) -> Result<(), EngineError> {
    let total = decode_graph_cost(file, source_len);
    if total.work > MAX_GRAPH_WORK {
        return Err(decode_failed(format!(
            "its item graph takes more than {MAX_GRAPH_WORK} steps to walk"
        )));
    }
    if total.held > MAX_GRAPH_BYTES {
        return Err(decode_failed(format!(
            "its item graph would hold more than {} MiB once built",
            MAX_GRAPH_BYTES >> 20
        )));
    }
    Ok(())
}

/// What [`check_item_graphs`] holds against its bounds: the
/// cost of every try the decode makes, added up, counted no
/// further than past [`Cost::LIMIT`].
fn decode_graph_cost<D: AsRef<[u8]>>(file: &oxideav_heif::HeifFile<D>, source_len: usize) -> Cost {
    let (Ok(meta), Ok(primary)) = (file.meta(), file.primary_item()) else {
        return Cost::default();
    };
    let mut cost = GraphCost::new(meta, source_len as u64);
    let mut tries = vec![primary.id];
    for group in meta.groups_containing(primary.id, b"altr") {
        tries.extend(group.entity_ids.iter().filter(|&&id| id != primary.id));
    }
    let mut total = Cost::default();
    for item in tries {
        total = total.plus(cost.decode(item, Cost::LIMIT.less(total)));
        if total.exceeds(Cost::LIMIT) {
            break;
        }
    }
    total
}

/// What walking part of an item graph costs: the work the walk
/// does, and the bytes the graph it builds holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Cost {
    work: u64,
    held: u64,
}

impl Cost {
    /// The most [`check_item_graphs`] lets the walks cost.
    const LIMIT: Cost = Cost {
        work: MAX_GRAPH_WORK,
        held: MAX_GRAPH_BYTES,
    };

    fn work(work: u64) -> Cost {
        Cost { work, held: 0 }
    }

    fn held(held: u64) -> Cost {
        Cost { work: 0, held }
    }

    fn plus(self, other: Cost) -> Cost {
        Cost {
            work: self.work.saturating_add(other.work),
            held: self.held.saturating_add(other.held),
        }
    }

    /// What is left of `self` once `spent` is taken out of it.
    fn less(self, spent: Cost) -> Cost {
        Cost {
            work: self.work.saturating_sub(spent.work),
            held: self.held.saturating_sub(spent.held),
        }
    }

    /// Whether either count is past `limit`'s.
    fn exceeds(self, limit: Cost) -> bool {
        self.work > limit.work || self.held > limit.held
    }
}

/// The bytes a graph node takes in the list or box that holds
/// it. The derivation inputs' list is sized to fit. The other
/// lists grow as nodes are pushed: a list's first allocation
/// holds four, and it never holds more than twice its length
/// after, so [`GraphCost`] charges a node in one of those at
/// four times this size.
const NODE_BYTES: u64 = std::mem::size_of::<oxideav_heif::derived::ImageNode>() as u64;

/// A property entry of a node, before what its parse holds.
const ENTRY_BYTES: u64 = std::mem::size_of::<oxideav_heif::props::PropertyEntry>() as u64;

/// A copy of an item's entry, before its strings.
const INFO_BYTES: u64 = std::mem::size_of::<oxideav_heif::meta::ItemInfo>() as u64;

/// The allocator's own overhead, charged once for each copy of
/// an item's entry, each list a node starts, and each property
/// whose parse holds anything of its body.
const ALLOCATION_SLACK: u64 = 64;

/// What a property costs each time a visit resolves it: the
/// work of reading its body, and the bytes its parse holds
/// besides its entry. A `tilC` property also names the
/// properties of the tiles of a `tili` item, which a visit
/// resolves too.
struct PropertyCost {
    parse: Cost,
    /// The property index of each tile property a `tilC` names,
    /// `0` (no property) included; `None` for any other property.
    tile_properties: Option<Vec<u16>>,
}

impl PropertyCost {
    fn of(raw: &oxideav_heif::meta::RawProperty) -> Self {
        use oxideav_heif::props::Property;
        let body = raw.body.len() as u64;
        let parsed = Property::parse(raw);
        // A parse keeps an ICC profile, or a box the parser does
        // not know, as bytes: at most twice the body. Any other
        // parse is held at up to 32 times its body, since two
        // bytes of a parameter-set list become a 24-byte list of
        // their own. A parse that fails ends the build. Each
        // parse of a body that is not empty may also take an
        // allocation of its own, which the allocator rounds up.
        let factor = match &parsed {
            Ok(Property::Unknown(_) | Property::Colr(_)) | Err(_) => 2,
            Ok(_) => 32,
        };
        let slack = if body == 0 { 0 } else { ALLOCATION_SLACK };
        let tile_properties = match parsed {
            Ok(Property::TilC(tilc)) => Some(
                tilc.in_file_tiles
                    .map(|(_, list)| list.into_iter().map(|(_, index)| index).collect())
                    .unwrap_or_default(),
            ),
            _ => None,
        };
        PropertyCost {
            parse: Cost {
                work: 1 + body,
                held: body.saturating_mul(factor).saturating_add(slack),
            },
            tile_properties,
        }
    }

    /// The cost of a property entry: the entry itself, which
    /// holds the parse, and what the parse holds besides.
    fn entry(&self) -> Cost {
        self.parse.plus(Cost::held(ENTRY_BYTES))
    }
}

/// The cost model behind [`check_item_graphs`], built in one
/// pass over the tables: what every item visit scans and holds,
/// what each item adds to that, and the references a visit
/// follows, in the walk's own order (derivation inputs, then
/// auxiliary images, then thumbnails).
struct GraphCost {
    /// The table entries every visit scans: the item list once
    /// (the lookup), the associations once (the properties),
    /// the references four times (inputs, auxiliaries,
    /// thumbnails, metadata), and the targets of every
    /// auxiliary, thumbnail, and metadata reference; and the
    /// lists every node starts.
    per_visit: Cost,
    /// What a visit of each item adds: the copy of its entry,
    /// its properties, an item lookup and a copied entry per
    /// metadata item describing it, for a derived item a
    /// location lookup and its payload, and for an item with an
    /// auxiliary image, one more scan of the references and a
    /// copy of its `prem` targets, which the alpha check reads.
    own: std::collections::HashMap<u32, Cost>,
    inputs: std::collections::HashMap<u32, Vec<u32>>,
    auxiliaries: std::collections::HashMap<u32, Vec<u32>>,
    thumbnails: std::collections::HashMap<u32, Vec<u32>>,
    /// The `tmap` item whose gain map a decode of an item
    /// reaches: the item itself when it is a `tmap`, else the
    /// first `tmap` whose first input it is.
    gain_maps: std::collections::HashMap<u32, u32>,
    /// What one gain map lookup scans: the item list, and the
    /// references once for each `tmap` item.
    gain_lookup: u64,
    /// The cost of every subtree walked to its end, but for the
    /// node at its root, which its parent's list holds.
    done: std::collections::HashMap<u32, Cost>,
    /// The items on the current chain, for the cycle check.
    chain: Vec<u32>,
}

impl GraphCost {
    fn new(meta: &oxideav_heif::meta::Meta, source_len: u64) -> Self {
        use oxideav_heif::meta::{self as heif, reference};
        let items = meta.items.len() as u64;
        // The entry each lookup finds: the first with the id.
        let mut entries = std::collections::HashMap::new();
        for item in &meta.items {
            entries.entry(item.id).or_insert(item);
        }
        let entry_bytes = |id: u32| -> u64 {
            entries.get(&id).map_or(0, |item| {
                [
                    Some(&item.name),
                    item.content_type.as_ref(),
                    item.content_encoding.as_ref(),
                    item.item_uri_type.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|s| s.len() as u64)
                .fold(0u64, u64::saturating_add)
            })
        };
        let mut per_visit_work = items + meta.associations.len() as u64;
        let mut own = std::collections::HashMap::<u32, Cost>::new();
        let mut inputs = std::collections::HashMap::<u32, Vec<u32>>::new();
        let mut auxiliaries = std::collections::HashMap::<u32, Vec<u32>>::new();
        let mut thumbnails = std::collections::HashMap::<u32, Vec<u32>>::new();
        let mut premultiplied = std::collections::HashMap::<u32, u64>::new();
        for r in &meta.references {
            per_visit_work += 4;
            if r.reference_type == reference::DIMG {
                inputs
                    .entry(r.from_item_id)
                    .or_default()
                    .extend(&r.to_item_ids);
                continue;
            }
            if r.reference_type == reference::PREM {
                *premultiplied.entry(r.from_item_id).or_default() += r.to_item_ids.len() as u64;
                continue;
            }
            let towards = match r.reference_type {
                reference::AUXL => &mut auxiliaries,
                reference::THMB => &mut thumbnails,
                reference::CDSC => {
                    per_visit_work += r.to_item_ids.len() as u64;
                    // Each visit of a described item looks the
                    // describing item up and keeps a copy of its
                    // entry, in a list that grows by pushes.
                    let strings = entry_bytes(r.from_item_id);
                    let each = Cost {
                        work: items.saturating_add(strings),
                        held: (4 * INFO_BYTES + ALLOCATION_SLACK).saturating_add(strings),
                    };
                    let mut targets = r.to_item_ids.clone();
                    targets.sort_unstable();
                    targets.dedup();
                    for target in targets {
                        let entry = own.entry(target).or_default();
                        *entry = entry.plus(each);
                    }
                    continue;
                }
                _ => continue,
            };
            per_visit_work += r.to_item_ids.len() as u64;
            // A visit finds each reference box naming the item
            // once, however often the box repeats it.
            let mut targets = r.to_item_ids.clone();
            targets.sort_unstable();
            targets.dedup();
            for target in targets {
                towards.entry(target).or_default().push(r.from_item_id);
            }
        }
        let properties: Vec<PropertyCost> = meta.properties.iter().map(PropertyCost::of).collect();
        let property = |index: u16| -> Option<&PropertyCost> {
            (index as usize)
                .checked_sub(1)
                .and_then(|i| properties.get(i))
        };
        let mut associated = std::collections::HashMap::<u32, u64>::new();
        for row in &meta.associations {
            let tiled = entries
                .get(&row.item_id)
                .is_some_and(|item| item.item_type == heif::ITEM_TYPE_TILI);
            let mut cost = Cost::default();
            for entry in &row.entries {
                let Some(p) = property(entry.index) else {
                    cost = cost.plus(Cost::work(1));
                    continue;
                };
                cost = cost.plus(p.entry());
                if tiled && let Some(tiles) = &p.tile_properties {
                    // A `tili` visit copies its `tilC` and
                    // resolves the tile properties it names into
                    // a list sized to fit, with a slot for every
                    // entry the `tilC` lists, an entry that names
                    // no property included.
                    cost = cost.plus(p.parse);
                    if !tiles.is_empty() {
                        let slots = (tiles.len() as u64).saturating_mul(ENTRY_BYTES);
                        cost = cost.plus(Cost::held(slots.saturating_add(ALLOCATION_SLACK)));
                    }
                    for &index in tiles {
                        cost = cost.plus(property(index).map_or(Cost::work(1), |t| t.parse));
                    }
                }
            }
            *associated.entry(row.item_id).or_default() += row.entries.len() as u64;
            let entry = own.entry(row.item_id).or_default();
            *entry = entry.plus(cost);
        }
        // An item's property entries go in a list that grows by
        // pushes: room for four at first, never more than twice
        // their number after. Their own size is in each entry's
        // cost already.
        for (&item, &count) in &associated {
            let room = count.saturating_mul(2).max(4);
            let entry = own.entry(item).or_default();
            *entry = entry.plus(Cost::held(
                room.saturating_sub(count).saturating_mul(ENTRY_BYTES),
            ));
        }
        // A visit that finds an alpha auxiliary checks whether
        // the item's samples are premultiplied: one more scan of
        // the references, and a copy of the item's `prem`
        // targets.
        for &item in auxiliaries.keys() {
            let scan = (meta.references.len() as u64)
                .saturating_add(premultiplied.get(&item).copied().unwrap_or(0));
            let entry = own.entry(item).or_default();
            *entry = entry.plus(Cost::work(scan));
        }
        // The first location of each item, as the lookup finds it.
        let mut locations = std::collections::HashMap::new();
        for loc in &meta.locations {
            locations.entry(loc.item_id).or_insert(loc);
        }
        for (&id, item) in &entries {
            // Each visit copies the item's entry, its strings
            // included.
            let strings = entry_bytes(id);
            let entry = own.entry(id).or_default();
            *entry = entry.plus(Cost {
                work: strings,
                held: strings.saturating_add(ALLOCATION_SLACK),
            });
            let derived = [
                heif::ITEM_TYPE_GRID,
                heif::ITEM_TYPE_IOVL,
                heif::ITEM_TYPE_CFEN,
                heif::ITEM_TYPE_TMAP,
            ]
            .contains(&item.item_type);
            if !derived {
                continue;
            }
            let payload = locations.get(&id).map_or(0, |loc| {
                loc.extents
                    .iter()
                    .map(|e| {
                        if e.length == 0 {
                            source_len
                        } else {
                            e.length.min(source_len)
                        }
                    })
                    .fold(0u64, u64::saturating_add)
            });
            *entry = entry.plus(Cost {
                work: (meta.locations.len() as u64).saturating_add(payload),
                held: payload,
            });
        }
        // The gain map lookup, as the decoder makes it.
        let mut gain_maps = std::collections::HashMap::new();
        for (&id, item) in &entries {
            if item.item_type == heif::ITEM_TYPE_TMAP {
                gain_maps.insert(id, id);
            }
        }
        let mut tone_maps = 0u64;
        for item in &meta.items {
            if item.item_type != heif::ITEM_TYPE_TMAP {
                continue;
            }
            tone_maps += 1;
            if let Some(&base) = inputs.get(&item.id).and_then(|ids| ids.first()) {
                gain_maps.entry(base).or_insert(item.id);
            }
        }
        let gain_lookup = tone_maps
            .saturating_mul(meta.references.len() as u64)
            .saturating_add(items);
        GraphCost {
            per_visit: Cost {
                work: per_visit_work,
                held: ALLOCATION_SLACK,
            },
            own,
            inputs,
            auxiliaries,
            thumbnails,
            gain_maps,
            gain_lookup,
            done: std::collections::HashMap::new(),
            chain: Vec::new(),
        }
    }

    /// What one decode of `item` holds at its peak, and the
    /// work of building what it holds: two copies of the item's
    /// graph, then the graph of its gain map image (see
    /// [`Self::gain_map`]) with the `tmap` item's payload and
    /// properties. The decoder looks the gain map up twice and
    /// builds its graph twice, but lets go of the first copy
    /// before it builds the second, so that part counts twice
    /// as work and once as bytes held.
    fn decode(&mut self, item: u32, budget: Cost) -> Cost {
        // A root is returned by value: its node is held once.
        let graph = self.walk(item, 0, budget).plus(Cost::held(NODE_BYTES));
        let lookups = Cost::work(self.gain_lookup.saturating_mul(2));
        let mut cost = graph.plus(graph).plus(lookups);
        if let Some((tmap, gain)) = self.gain_map(item) {
            let mut extra = self.own.get(&tmap).copied().unwrap_or_default();
            if let Some(gain) = gain {
                let walked = self.walk(gain, 0, budget.less(cost).less(extra));
                extra = extra.plus(walked).plus(Cost::held(NODE_BYTES));
            }
            cost = cost.plus(extra).plus(Cost::work(extra.work));
        }
        cost
    }

    /// The `tmap` item a decode of `item` takes its gain map
    /// from, and the gain map image, that `tmap`'s second input,
    /// should it have one.
    fn gain_map(&self, item: u32) -> Option<(u32, Option<u32>)> {
        let tmap = *self.gain_maps.get(&item)?;
        let gain = self.inputs.get(&tmap).and_then(|ids| ids.get(1)).copied();
        Some((tmap, gain))
    }

    /// The cost of walking the graph from `item`, reached at
    /// `depth`, but for the node at its root. Past `budget` the
    /// count stops early, as the caller refuses anything over it
    /// anyway.
    fn walk(&mut self, item: u32, depth: usize, budget: Cost) -> Cost {
        self.walk_inner(item, depth, budget).0
    }

    /// [`Self::walk`], also saying whether the walk reached the
    /// end of the subtree (`true`) or stops inside it, as the
    /// graph builder stops at a cycle or past its depth cap. A
    /// subtree that stops is not remembered: reached again at
    /// another depth, it may not.
    fn walk_inner(&mut self, item: u32, depth: usize, budget: Cost) -> (Cost, bool) {
        if let Some(&cost) = self.done.get(&item) {
            return (cost, true);
        }
        if depth > oxideav_heif::derived::MAX_DERIVATION_DEPTH || self.chain.contains(&item) {
            return (Cost::default(), false);
        }
        let mut cost = self
            .per_visit
            .plus(self.own.get(&item).copied().unwrap_or_default());
        // Each child with the bytes its node takes in this one:
        // the inputs' list is sized to fit, the others grow.
        let children: Vec<(u32, u64)> = [
            (&self.inputs, NODE_BYTES),
            (&self.auxiliaries, 4 * NODE_BYTES),
            (&self.thumbnails, 4 * NODE_BYTES),
        ]
        .into_iter()
        .filter_map(|(edges, node)| Some((edges.get(&item)?, node)))
        .flat_map(|(ids, node)| ids.iter().map(move |&id| (id, node)))
        .collect();
        self.chain.push(item);
        for (child, node) in children {
            if cost.exceeds(budget) {
                break;
            }
            cost = cost.plus(Cost::held(node));
            let (child_cost, complete) = self.walk_inner(child, depth + 1, budget.less(cost));
            cost = cost.plus(child_cost);
            if !complete {
                self.chain.pop();
                return (cost, false);
            }
        }
        self.chain.pop();
        if !cost.exceeds(budget) {
            self.done.insert(item, cost);
        }
        (cost, true)
    }
}

/// The gain map image of an item the decode may output (the
/// primary, or one of its `altr` alternatives) that is not
/// itself a `tmap` (a `tmap` item's gain map is its second
/// input, already in its graph): the second derivation input of
/// the `tmap` item whose first input is that item, mirroring the
/// decoder's own lookup in `find_gain_map`.
fn gain_map_node<D: AsRef<[u8]>>(
    file: &oxideav_heif::HeifFile<D>,
    item_id: u32,
) -> Option<oxideav_heif::ImageNode> {
    let meta = file.meta().ok()?;
    let tmap = meta.items.iter().find(|item| {
        item.item_type == oxideav_heif::meta::ITEM_TYPE_TMAP
            && meta.derivation_inputs(item.id).first() == Some(&item_id)
    })?;
    let inputs = meta.derivation_inputs(tmap.id);
    oxideav_heif::build_graph(file, *inputs.get(1)?).ok()
}

/// Check one graph node and, recursively, everything it pulls
/// into the decode: derivation inputs, the alpha auxiliary, and
/// the depth auxiliary. Items are visited once per validation;
/// an item's properties do not depend on where it is referenced
/// from, so a repeat visit could only repeat the verdict.
fn validate_node<D: AsRef<[u8]>>(
    file: &oxideav_heif::HeifFile<D>,
    node: &oxideav_heif::ImageNode,
    seen: &mut std::collections::HashSet<u32>,
) -> Result<(), EngineError> {
    if !seen.insert(node.item.id) {
        return Ok(());
    }
    check_hevc_coded(node)?;
    if let Ok((width, height)) = node.output_size() {
        check_announced_size(node.item.id, width, height)?;
    }
    validate_coded_geometry(file, node)?;
    for input in &node.inputs {
        validate_node(file, input, seen)?;
    }
    if let Some(alpha) = &node.alpha {
        validate_node(file, alpha, seen)?;
    }
    if let Some(depth) = &node.depth {
        validate_node(file, depth, seen)?;
    }
    Ok(())
}

/// Refuse a node whose pixels would come from a codec other
/// than HEVC (issue #466): a coded item of another type, or a
/// tiled item whose in-file tiles are. Derived items (grid,
/// overlay, identity, tone map, colour enhancement) code no
/// pixels themselves; their inputs are checked as the walk
/// recurses. A tiled item with external tiles names no codec
/// here, and the decoder refuses external tiles itself.
fn check_hevc_coded(node: &oxideav_heif::ImageNode) -> Result<(), EngineError> {
    let coded = match &node.kind {
        oxideav_heif::ImageKind::Coded(item_type) => Some(item_type),
        oxideav_heif::ImageKind::Tiled(tiled) => tiled
            .config
            .in_file_tiles
            .as_ref()
            .map(|(tile_type, _)| tile_type),
        _ => None,
    };
    match coded {
        Some(item_type) if !HEVC_ITEM_TYPES.contains(item_type) => Err(decode_failed(format!(
            "item {} is coded as '{}', not HEVC; only HEVC-coded items are decoded",
            node.item.id,
            item_type.escape_ascii()
        ))),
        _ => Ok(()),
    }
}

/// One item's announced size against the shared decode budget:
/// the same 16384 px per axis and 256 MiB of RGBA pixels every
/// other decoder is held to. The decoder itself applies this
/// only to the primary's output; auxiliaries, grid tiles, and
/// gain maps announce sizes nothing else checks.
fn check_announced_size(item_id: u32, width: u32, height: u32) -> Result<(), EngineError> {
    if width > MAX_DECODE_DIMENSION || height > MAX_DECODE_DIMENSION {
        return Err(decode_failed(format!(
            "item {item_id} announces {width}x{height}, past the {MAX_DECODE_DIMENSION} px decode limit"
        )));
    }
    if width as u64 * height as u64 > MAX_DECODE_PIXELS {
        return Err(decode_failed(format!(
            "item {item_id} announces {width}x{height}, past the decode pixel budget"
        )));
    }
    Ok(())
}

/// Check every HEVC SPS an item carries against its declared
/// size: the configuration record's arrays (hvcC, or lhvC for a
/// layered item) and any SPS travelling in band in the item
/// payload, which the codec would honor just the same. Items of
/// other codecs, and items whose SPS does not parse (the codec
/// rejects those itself), pass without a verdict.
fn validate_coded_geometry<D: AsRef<[u8]>>(
    file: &oxideav_heif::HeifFile<D>,
    node: &oxideav_heif::ImageNode,
) -> Result<(), EngineError> {
    let mut configs: Vec<(&[oxideav_heif::hvcc::NalArray], u8)> = Vec::new();
    if let Some(hvcc) = node.properties.hvcc() {
        configs.push((&hvcc.arrays, hvcc.length_size));
    }
    if let Some(lhvc) = node.properties.lhvc() {
        configs.push((&lhvc.arrays, lhvc.length_size));
    }
    if configs.is_empty() {
        return Ok(());
    }
    let ispe = node.ispe();
    for (arrays, _) in &configs {
        for array in *arrays {
            if array.nal_unit_type != 33 {
                continue; // not an SPS array
            }
            for nal in &array.nal_units {
                check_sps_nal(nal, node.item.id, ispe)?;
            }
        }
    }
    let length_size = configs[0].1;
    if let Ok(data) = file.item_data(node.item.id)
        && let Ok(nals) = oxideav_heif::hvcc::split_length_prefixed(&data, length_size)
    {
        for nal in nals {
            if nal.len() > 2 && (nal[0] >> 1) & 0x3f == 33 {
                check_sps_nal(nal, node.item.id, ispe)?;
            }
        }
    }
    Ok(())
}

/// Parse one SPS NAL (header included, as the configuration
/// record and the payload both carry it) and check its coded
/// geometry. An SPS that does not parse yields no verdict: the
/// codec runs the same parser and fails the decode itself.
fn check_sps_nal(nal: &[u8], item_id: u32, ispe: Option<(u32, u32)>) -> Result<(), EngineError> {
    let unescaped = oxideav_h265::nal::strip_emulation_prevention(nal);
    if unescaped.len() <= 2 {
        return Ok(());
    }
    let Ok(sps) = oxideav_h265::sps::SeqParameterSet::parse(&unescaped[2..]) else {
        return Ok(());
    };
    check_coded_geometry(
        item_id,
        sps.pic_width_in_luma_samples,
        sps.pic_height_in_luma_samples,
        ispe,
    )
}

/// The coded-geometry rule itself: a coded picture may not pass
/// the per-axis decode cap at all, and may exceed the item's
/// declared size only by [`SPS_SIZE_TOLERANCE`] per axis (the
/// block-alignment padding every encoder writes). With no
/// declared size to compare against, the coded picture is held
/// to the pixel budget directly.
fn check_coded_geometry(
    item_id: u32,
    coded_width: u32,
    coded_height: u32,
    ispe: Option<(u32, u32)>,
) -> Result<(), EngineError> {
    if coded_width > MAX_DECODE_DIMENSION || coded_height > MAX_DECODE_DIMENSION {
        return Err(decode_failed(format!(
            "item {item_id} codes {coded_width}x{coded_height}, past the {MAX_DECODE_DIMENSION} px decode limit"
        )));
    }
    match ispe {
        Some((width, height)) => {
            if coded_width > width.saturating_add(SPS_SIZE_TOLERANCE)
                || coded_height > height.saturating_add(SPS_SIZE_TOLERANCE)
            {
                return Err(decode_failed(format!(
                    "item {item_id} codes {coded_width}x{coded_height} against a declared {width}x{height}"
                )));
            }
        }
        None => {
            if coded_width as u64 * coded_height as u64 > MAX_DECODE_PIXELS {
                return Err(decode_failed(format!(
                    "item {item_id} codes {coded_width}x{coded_height}, past the decode pixel budget"
                )));
            }
        }
    }
    Ok(())
}

/// Metadata the decoder hands back, held to the same size caps
/// the JPEG/PNG extraction path applies (issue #184): the crate
/// is built without its `deflate` feature, so a deflated
/// metadata item cannot inflate toward the crate's 1 GiB
/// cap on every decode, and anything still oversized after
/// decode is dropped rather than cloned into the pipeline.
fn cap_metadata(bytes: Option<Vec<u8>>, cap: usize) -> Option<Vec<u8>> {
    bytes.filter(|bytes| bytes.len() <= cap)
}

/// The primary image's announced dimensions, read from the
/// parsed file structure alone: no item is decoded. The
/// `/explain` probe (issue #196) reports source facts
/// without paying a decode, and HEIC dimensions live in the
/// item graph rather than in any fixed header offset.
/// `None` when the bytes do not parse or the primary graph
/// announces no size.
pub(crate) fn probe_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let file = oxideav_heif::HeifFile::parse_borrowed(bytes).ok()?;
    check_item_graphs(&file, bytes.len()).ok()?;
    let node = oxideav_heif::build_primary_graph(&file).ok()?;
    node.output_size().ok()
}

/// Decode a HEIC/HEIF source to an upright image plus its source
/// metadata. Failures are the pipeline's decode class (a 502
/// upstream), naming HEIC, never a panic: the decoder is pure
/// Rust and its error surface on hostile input is errors, not
/// aborts (issue #163: 320 mutants, zero panics).
pub(crate) fn decode_heic(bytes: &[u8]) -> Result<(DynamicImage, SourceMeta), EngineError> {
    // The budget is enforced on the declared and coded geometry
    // of everything the decode will touch before it starts;
    // see validate_decode_budget for why the decoder's own
    // checks do not cover this.
    validate_decode_budget(bytes)?;
    let mut img = oxideav_heif::decode_with(bytes, &decode_options()).map_err(decode_failed)?;
    // The colour the decoder assumed is only a default when the
    // file carries no nclx colr box; the bitstream's own word on
    // the matter (the SPS VUI) wins where it exists.
    if let Some(color) = vui_color(bytes) {
        img.color = color;
    }
    let (width, height) = (img.width, img.height);
    // The pre-decode validation and the decoder's limits run
    // first; the decoded size is still re-checked against the
    // shared budget in our own code, the same defence in depth
    // decode_image applies.
    crate::engine::check_decoded_dimensions(width as u64, height as u64)?;
    if width as u64 * height as u64 * 4 > MAX_DECODE_ALLOC {
        return Err(decode_failed(format!(
            "image dimensions {width}x{height} exceed the decode allocation limit"
        )));
    }
    let rgba = img.try_to_rgba8().map_err(decode_failed)?;
    let buffer = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(width, height, rgba)
        .ok_or_else(|| decode_failed("decoded planes do not match the decoded size"))?;
    let mut image = DynamicImage::ImageRgba8(buffer);
    // Orientation: the decoder resolves the HEIF offset word and
    // hands back the TIFF stream, which the engine's EXIF reader
    // (the same one JPEG sources go through) reads the tag from.
    if let Some(exif) = &img.metadata.exif
        && let Some(orientation) = crate::engine::exif_orientation(exif)
    {
        image = crate::engine::apply_orientation(image, orientation);
    }
    let mut meta = SourceMeta {
        icc: cap_metadata(img.metadata.icc.clone(), crate::color::MAX_ICC_BYTES),
        exif_tiff: cap_metadata(img.metadata.exif.clone(), crate::color::MAX_META_BYTES),
        xmp: cap_metadata(img.metadata.xmp.clone(), crate::color::MAX_XMP_BYTES),
        xmp_compressed: false,
        photoshop: None,
    };
    // The pixels are upright now; a preserved EXIF must say so,
    // and must not carry a thumbnail of the source, exactly as
    // the JPEG/PNG extraction path records it.
    if let Some(tiff) = &mut meta.exif_tiff {
        crate::color::sanitize_preserved_exif(tiff);
    }
    Ok((image, meta))
}

/// The colour description the SPS VUI prescribes, when the
/// container leaves the choice to the bitstream.
///
/// oxideav-heif converts with the file's `nclx` `colr` box, or
/// the MIAF default (BT.601 matrix, full range) when there is
/// none. It never consults the SPS VUI, so a file whose only
/// colour description lives in the bitstream decodes washed out
/// next to every other tool: FFmpeg and libheif both fall back
/// to the VUI. This reads it: the primary item's first SPS,
/// parsed with oxideav-h265. An `nclx` box suppresses the
/// override entirely (the file's explicit word wins, byte for
/// byte), and any parse failure along the way leaves the
/// decoder's default in place — the VUI is a correction, never
/// a new failure mode.
fn vui_color(bytes: &[u8]) -> Option<oxideav_heif::ColorInfo> {
    let file = oxideav_heif::HeifFile::parse_borrowed(bytes).ok()?;
    check_item_graphs(&file, bytes.len()).ok()?;
    let node = oxideav_heif::build_primary_graph(&file).ok()?;
    // The colour node mirrors the decoder's own choice: a
    // tone-mapped primary describes its base rendition's colour.
    let colour_node = match (&node.kind, node.inputs.first()) {
        (oxideav_heif::ImageKind::ToneMap(_), Some(base)) => base,
        _ => &node,
    };
    if colour_node.properties.nclx().is_some() {
        return None;
    }
    let sps = find_sps(&node)?;
    let signal = sps
        .vui_parameters
        .as_ref()
        .and_then(|vui| vui.video_signal_type.as_ref());
    Some(color_from_signal(signal))
}

/// What a (possibly absent) VUI video signal block means for the
/// conversion: the signalled range and colour description, with
/// two spec inferences filled in. A VUI that omits the colour
/// description leaves the code points unspecified, for which the
/// MIAF default description (BT.709 primaries, sRGB transfer,
/// BT.601 matrix) is the standing assumption; and a stream that
/// signals no video signal type at all infers limited range
/// (H.265 §E.3.1: `video_full_range_flag` is inferred 0), which
/// is the case the whole correction exists for — older encoders
/// wrote limited-range pictures and said nothing anywhere.
fn color_from_signal(
    signal: Option<&oxideav_h265::vui::VideoSignalType>,
) -> oxideav_heif::ColorInfo {
    let (full_range, description) = match signal {
        Some(signal) => (
            signal.video_full_range_flag,
            signal.colour_description.map(|d| {
                (
                    d.colour_primaries,
                    d.transfer_characteristics,
                    d.matrix_coeffs,
                )
            }),
        ),
        None => (false, None),
    };
    let (primaries, transfer, matrix) = description.unwrap_or((1, 13, 6));
    oxideav_heif::ColorInfo::new(
        if full_range {
            oxideav_heif::ColorRange::Full
        } else {
            oxideav_heif::ColorRange::Limited
        },
        primaries,
        transfer,
        matrix,
    )
}

/// The first SPS in the primary item's derivation tree (grid and
/// overlay inputs included: their tiles carry the same parameter
/// sets as the composition they feed).
fn find_sps(node: &oxideav_heif::ImageNode) -> Option<oxideav_h265::sps::SeqParameterSet> {
    if let Some(hvcc) = node.properties.hvcc() {
        for array in &hvcc.arrays {
            if array.nal_unit_type != 33 {
                continue; // not an SPS array
            }
            for nal in &array.nal_units {
                let unescaped = oxideav_h265::nal::strip_emulation_prevention(nal);
                if unescaped.len() > 2
                    && let Ok(sps) = oxideav_h265::sps::SeqParameterSet::parse(&unescaped[2..])
                {
                    return Some(sps);
                }
            }
        }
    }
    for input in &node.inputs {
        if let Some(sps) = find_sps(input) {
            return Some(sps);
        }
    }
    if let Some(alpha) = &node.alpha
        && let Some(sps) = find_sps(alpha)
    {
        return Some(sps);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixture provenance (tests/fixtures/): every HEIC fixture was
    // generated for this test suite on 2026-10-24 with the oxideav
    // encoder (the crates this module decodes with), from synthetic
    // patterns written by the generator — no third-party images.
    //   grid.heic       512x384 quadrant pattern, 256 px grid tiles
    //   main10.heif     128x96 YUV 4:2:0 10-bit gradient/blocks
    //   vui-limited.heic 320x240, colour only in the SPS VUI
    //                   (limited range, BT.709), no colr box at all
    //   p3.heic         64x64 muted blocks under a Display P3 ICC
    //                   profile generated by moxcms
    //   orient6.heic    64x48 quadrants, EXIF Orientation=6, no irot
    // Reference pixel values below were cross-checked against
    // FFmpeg 8.1.2's decode of the same files (max abs error 1 on
    // the VUI fixture once the correction applies, 24 on one
    // 10-bit gradient edge): the tolerances carry that margin.
    const GRID: &[u8] = include_bytes!("../tests/fixtures/grid.heic");
    const MAIN10: &[u8] = include_bytes!("../tests/fixtures/main10.heif");
    const VUI_LIMITED: &[u8] = include_bytes!("../tests/fixtures/vui-limited.heic");
    const P3: &[u8] = include_bytes!("../tests/fixtures/p3.heic");
    const ORIENT6: &[u8] = include_bytes!("../tests/fixtures/orient6.heic");

    fn pixel(img: &DynamicImage, x: u32, y: u32) -> (u8, u8, u8) {
        let p = image::GenericImageView::get_pixel(img, x, y);
        (p[0], p[1], p[2])
    }

    fn assert_near(actual: (u8, u8, u8), expected: (u8, u8, u8), tol: i32, what: &str) {
        for (a, e) in [actual.0, actual.1, actual.2]
            .iter()
            .zip([expected.0, expected.1, expected.2].iter())
        {
            assert!(
                (*a as i32 - *e as i32).abs() <= tol,
                "{what}: got {actual:?}, expected {expected:?} within {tol}"
            );
        }
    }

    #[test]
    fn grid_fixture_composes_the_full_canvas() {
        let (img, _) = decode_heic(GRID).unwrap();
        assert_eq!((img.width(), img.height()), (512, 384));
        // One spot per tile quadrant: tiles the composition had
        // to place, not one coded picture.
        assert_near(pixel(&img, 128, 96), (230, 29, 29), 4, "top-left tile");
        assert_near(pixel(&img, 384, 96), (30, 199, 30), 4, "top-right tile");
        assert_near(pixel(&img, 128, 288), (30, 29, 230), 4, "bottom-left tile");
        assert_near(pixel(&img, 384, 288), (240, 240, 240), 4, "bottom-right");
    }

    #[test]
    fn ten_bit_fixture_decodes_with_its_depth_intact() {
        let raw = oxideav_heif::decode_with(MAIN10, &decode_options()).unwrap();
        assert_eq!(format!("{:?}", raw.format), "Yuv420P10Le");
        let (img, _) = decode_heic(MAIN10).unwrap();
        assert_eq!((img.width(), img.height()), (128, 96));
        // The luma gradient runs dark to bright left to right.
        let (left, _, _) = pixel(&img, 4, 24);
        let (right, _, _) = pixel(&img, 124, 24);
        assert!(right > left + 40, "gradient: {left} -> {right}");
        assert_near(pixel(&img, 64, 24), (83, 171, 15), 12, "gradient middle");
    }

    #[test]
    fn exif_orientation_rotates_the_output() {
        let (img, meta) = decode_heic(ORIENT6).unwrap();
        // Stored 64x48; Orientation=6 turns it upright at 48x64.
        assert_eq!((img.width(), img.height()), (48, 64));
        // A clockwise quarter turn moves the stored quadrants:
        // BL (blue) -> TL, TL (red) -> TR, BR (white) -> BL,
        // TR (green) -> BR.
        let (r, g, b) = pixel(&img, 8, 8);
        assert!(
            b > r && b > g,
            "top-left should be blue, got {:?}",
            (r, g, b)
        );
        let (r, g, b) = pixel(&img, 40, 8);
        assert!(
            r > g && r > b,
            "top-right should be red, got {:?}",
            (r, g, b)
        );
        let (r, g, b) = pixel(&img, 40, 56);
        assert!(
            g > r && g > b,
            "bottom-right should be green, got {:?}",
            (r, g, b)
        );
        // The preserved EXIF reports the orientation the pixels
        // now have (1), not the one they were stored with.
        let tiff = meta.exif_tiff.expect("EXIF survives");
        assert_eq!(crate::engine::exif_orientation(&tiff), Some(1));
    }

    /// The orientation fixture with its EXIF item replaced by
    /// `tiff`. The new item goes at the end of `mdat`, the last
    /// box, and the item's iloc extent (8-byte offset and
    /// length) points at it; the old item's bytes stay where
    /// they were, unreferenced.
    fn orient6_with_exif(tiff: &[u8]) -> Vec<u8> {
        let mut bytes = ORIENT6.to_vec();
        let old_extent = [676u64.to_be_bytes(), 30u64.to_be_bytes()].concat();
        let at = bytes
            .windows(16)
            .position(|w| w == old_extent)
            .expect("the fixture's EXIF extent");
        let item = [&[0u8; 4][..], tiff].concat();
        let new_extent = [
            (bytes.len() as u64).to_be_bytes(),
            (item.len() as u64).to_be_bytes(),
        ]
        .concat();
        bytes[at..at + 16].copy_from_slice(&new_extent);
        let mdat = bytes
            .windows(4)
            .position(|w| w == b"mdat")
            .expect("the fixture's mdat")
            - 4;
        let size = u32::from_be_bytes(bytes[mdat..mdat + 4].try_into().unwrap());
        assert_eq!(mdat + size as usize, bytes.len(), "mdat is the last box");
        bytes[mdat..mdat + 4].copy_from_slice(&(size + item.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&item);
        bytes
    }

    #[test]
    fn exif_thumbnails_are_cut_from_heic_metadata() {
        // Issue #611: the HEIC path hands the decoder's EXIF
        // to a preserving output as JPEG and PNG sources do,
        // so it drops the IFD1 thumbnail the same way.
        use crate::color::fixture;
        let thumbnail = fixture::thumbnail_jpeg();
        let cam = fixture::camera(false, 6, (64, 48), &thumbnail);
        let (img, meta) = decode_heic(&orient6_with_exif(&cam.tiff)).unwrap();
        // The new EXIF is the one read: Orientation 6 still
        // turns the stored 64x48 upright.
        assert_eq!((img.width(), img.height()), (48, 64));
        let tiff = meta.exif_tiff.expect("EXIF survives");
        assert!(
            !fixture::contains(&tiff, &thumbnail),
            "the thumbnail is still there"
        );
        assert_eq!(fixture::next_ifd(&tiff, 8), Some(0));
        assert_eq!(tiff.len(), cam.ifd1);
        assert_eq!(crate::engine::exif_orientation(&tiff), Some(1));
    }

    #[test]
    fn vui_limited_range_matches_the_reference_decode() {
        // Without the correction the same file decodes visibly
        // washed out: the dark block lands ~13 levels high on
        // average against FFmpeg. Pin both sides so the fixture
        // cannot silently stop exercising the gap.
        let plain = oxideav_heif::decode_with(VUI_LIMITED, &decode_options()).unwrap();
        let plain = DynamicImage::ImageRgba8(
            image::ImageBuffer::from_raw(plain.width, plain.height, plain.try_to_rgba8().unwrap())
                .unwrap(),
        );
        let (dark, _, _) = pixel(&plain, 10, 10);
        assert!(
            dark > 48,
            "uncorrected decode should run bright, got {dark}"
        );

        let (img, _) = decode_heic(VUI_LIMITED).unwrap();
        assert_eq!((img.width(), img.height()), (320, 240));
        assert_near(pixel(&img, 10, 10), (39, 39, 39), 3, "dark block");
        assert_near(pixel(&img, 200, 60), (200, 29, 30), 3, "red block");
        assert_near(pixel(&img, 200, 180), (235, 235, 235), 3, "bright block");
    }

    #[test]
    fn signal_inference_follows_the_spec() {
        use oxideav_h265::vui::{ColourDescription, VideoSignalType};
        let signal = |full: bool, desc: Option<(u8, u8, u8)>| VideoSignalType {
            video_format: 5,
            video_full_range_flag: full,
            colour_description: desc.map(|(p, t, m)| ColourDescription {
                colour_primaries: p,
                transfer_characteristics: t,
                matrix_coeffs: m,
            }),
        };
        // No signal at all: limited range is the H.265 inference,
        // the MIAF description the standing assumption. This is
        // the Nokia cheers case from the spike (that file's terms
        // do not allow vendoring it, so the rule is pinned here
        // and the present-VUI path end to end on the fixture).
        let c = color_from_signal(None);
        assert_eq!(c.range, oxideav_heif::ColorRange::Limited);
        assert_eq!((c.primaries, c.transfer, c.matrix), (1, 13, 6));
        // A signalled description is honored whole.
        let c = color_from_signal(Some(&signal(false, Some((9, 16, 9)))));
        assert_eq!(c.range, oxideav_heif::ColorRange::Limited);
        assert_eq!((c.primaries, c.transfer, c.matrix), (9, 16, 9));
        // Full range signalled, description omitted: unspecified
        // code points keep the standing assumption.
        let c = color_from_signal(Some(&signal(true, None)));
        assert_eq!(c.range, oxideav_heif::ColorRange::Full);
        assert_eq!((c.primaries, c.transfer, c.matrix), (1, 13, 6));
    }

    #[test]
    fn a_colr_box_suppresses_the_override() {
        // grid.heic carries an nclx box: its pixels must come out
        // exactly as the bare decoder produces them.
        assert!(vui_color(GRID).is_none());
        assert!(vui_color(P3).is_none());
        let raw = oxideav_heif::decode_with(GRID, &decode_options()).unwrap();
        let raw = raw.try_to_rgba8().unwrap();
        let (img, _) = decode_heic(GRID).unwrap();
        assert_eq!(img.to_rgba8().into_raw(), raw);
    }

    #[test]
    fn the_icc_profile_reaches_the_source_meta() {
        let (_, meta) = decode_heic(P3).unwrap();
        let icc = meta.icc.expect("the P3 profile is extracted");
        assert!(icc.len() > 100 && &icc[36..40] == b"acsp", "ICC header");
        // And the raw decode underneath keeps the stored values:
        // conversion to sRGB is the engine's step, not this one.
        let (img, _) = decode_heic(P3).unwrap();
        assert_near(pixel(&img, 16, 32), (200, 118, 60), 4, "stored P3 values");
    }

    #[test]
    fn the_engine_converts_heic_pixels_to_srgb() {
        // Through the full pipeline the P3 fixture's stored values
        // are reinterpreted by their profile: the served pixel is
        // the sRGB conversion of (200, 118, 60), measurably not
        // the stored triplet.
        let engine = crate::engine::Engine;
        let out = engine
            .process(
                P3,
                &[crate::url::Operation::Filters(vec![
                    crate::url::Filter::Format(crate::url::FormatSpec::plain(
                        crate::url::ImageFormat::Png,
                    )),
                ])],
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap();
        assert_near(pixel(&img, 16, 32), (213, 115, 44), 6, "P3 red in sRGB");
        assert_near(pixel(&img, 48, 32), (27, 122, 206), 6, "P3 blue in sRGB");
    }

    #[test]
    fn truncated_and_corrupt_files_fail_cleanly() {
        for bytes in [&ORIENT6[..ORIENT6.len() / 2], &ORIENT6[..24], &GRID[..100]] {
            match decode_heic(bytes) {
                Err(EngineError::Decode(msg)) => assert!(msg.contains("HEIC"), "{msg}"),
                other => panic!("expected a HEIC decode error, got {other:?}"),
            }
        }
    }

    #[test]
    fn decode_limits_reject_oversized_declared_dimensions() {
        // Rewrite the ispe box of the orientation fixture: the
        // declared size alone must trip the shared decode limit
        // before any pixel is produced.
        for (w, h) in [(20_000u32, 48u32), (64, 20_000)] {
            let mut bytes = ORIENT6.to_vec();
            let pos = bytes
                .windows(4)
                .position(|w| w == b"ispe")
                .expect("fixture carries ispe");
            // Past the type: version/flags (4), then width, height.
            bytes[pos + 8..pos + 12].copy_from_slice(&w.to_be_bytes());
            bytes[pos + 12..pos + 16].copy_from_slice(&h.to_be_bytes());
            match decode_heic(&bytes) {
                Err(EngineError::Decode(_)) => {}
                other => panic!("{w}x{h}: expected a decode error, got {other:?}"),
            }
        }
    }

    #[test]
    fn coded_geometry_far_past_the_declared_size_is_rejected() {
        // The audit's mismatch shape, in miniature: shrink the
        // grid tiles' shared ispe to 16x16 while their SPS still
        // codes 256x256 pictures. The coded geometry runs far
        // past the declared size plus the padding tolerance, so
        // validation must refuse the file before the codec
        // allocates a single plane. The rejection message is the
        // coded-geometry one, which only the pre-decode check
        // produces.
        let mut bytes = GRID.to_vec();
        let mut rewritten = 0;
        let mut pos = 0;
        while let Some(found) = bytes[pos..].windows(4).position(|w| w == b"ispe") {
            let at = pos + found;
            let dims = (
                u32::from_be_bytes(bytes[at + 8..at + 12].try_into().unwrap()),
                u32::from_be_bytes(bytes[at + 12..at + 16].try_into().unwrap()),
            );
            if dims == (256, 256) {
                bytes[at + 8..at + 12].copy_from_slice(&16u32.to_be_bytes());
                bytes[at + 12..at + 16].copy_from_slice(&16u32.to_be_bytes());
                rewritten += 1;
            }
            pos = at + 4;
        }
        assert_eq!(rewritten, 1, "the tiles share one ispe box");
        match decode_heic(&bytes) {
            Err(EngineError::Decode(msg)) => {
                assert!(msg.contains("codes 256x256"), "{msg}")
            }
            other => panic!("expected a coded-geometry rejection, got {other:?}"),
        }
    }

    #[test]
    fn non_hevc_items_are_refused_before_decoding() {
        // An AV1 still with its major brand patched to mif1
        // (issue #466): a valid HEIF file the container parses,
        // whose primary item is av01. The refusal names the
        // codec, which only the pre-decode check produces, so
        // the AV1 bytes never reach a decoder.
        let bytes = crate::engine::test_formats::av1_in_mif1();
        let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
        let node = oxideav_heif::build_primary_graph(&file).unwrap();
        assert!(matches!(node.kind, oxideav_heif::ImageKind::Coded(t) if &t == b"av01"));
        match decode_heic(&bytes) {
            Err(EngineError::Decode(msg)) => {
                assert!(msg.contains("coded as 'av01', not HEVC"), "{msg}")
            }
            other => panic!("expected the non-HEVC refusal, got {other:?}"),
        }
        // The rule itself: the four HEVC item types pass, every
        // other coded type is refused, whatever the container
        // around it.
        let mut node = node;
        for item_type in HEVC_ITEM_TYPES {
            node.kind = oxideav_heif::ImageKind::Coded(item_type);
            check_hevc_coded(&node).unwrap();
        }
        for item_type in [*b"av01", *b"avc1", *b"vvc1", *b"jpeg", *b"\0\0\0\0"] {
            node.kind = oxideav_heif::ImageKind::Coded(item_type);
            assert!(check_hevc_coded(&node).is_err(), "{item_type:?}");
        }
        // A tiled item is held to its tiles' codec.
        let tiled = |tile_type: [u8; 4]| {
            oxideav_heif::ImageKind::Tiled(oxideav_heif::derived::TiledItem::new(
                oxideav_heif::props::TilC::new(64, 64, Vec::new(), Some((tile_type, Vec::new()))),
                oxideav_heif::props::ItemProperties::default(),
            ))
        };
        node.kind = tiled(*b"hvc1");
        check_hevc_coded(&node).unwrap();
        node.kind = tiled(*b"av01");
        assert!(check_hevc_coded(&node).is_err());
        // Derived items code nothing themselves: their inputs
        // are checked as the walk recurses.
        node.kind = oxideav_heif::ImageKind::Identity;
        check_hevc_coded(&node).unwrap();
    }

    #[test]
    fn mutated_non_hevc_files_fail_cleanly() {
        // Hostile shapes around the refusal (issue #466): every
        // truncation and single-byte mutation of the AV1-in-mif1
        // file is a decode error, never a panic and never pixels.
        let bytes = crate::engine::test_formats::av1_in_mif1();
        for end in 0..bytes.len() {
            assert!(decode_heic(&bytes[..end]).is_err(), "truncated at {end}");
        }
        let mut mutated = bytes.clone();
        for i in 0..bytes.len() {
            for value in [0x00, 0xff, bytes[i] ^ 0x01, bytes[i] ^ 0x80] {
                mutated[i] = value;
                assert!(decode_heic(&mutated).is_err(), "byte {i} = {value:#04x}");
            }
            mutated[i] = bytes[i];
        }
    }

    #[test]
    fn hevc_fixtures_pass_the_codec_check() {
        // Every HEVC fixture, grid tiles and 10-bit included,
        // still decodes: the refusal is for other codecs only.
        for bytes in [GRID, MAIN10, VUI_LIMITED, P3, ORIENT6] {
            validate_decode_budget(bytes).unwrap();
        }
    }

    #[test]
    fn coded_geometry_rule_bounds() {
        // Coded size equal to the declaration, and padding out
        // to the tolerance, are what every encoder produces.
        check_coded_geometry(7, 64, 64, Some((64, 64))).unwrap();
        check_coded_geometry(7, 64 + SPS_SIZE_TOLERANCE, 64, Some((64, 64))).unwrap();
        // One sample past the tolerance is the mismatch shape.
        let err =
            check_coded_geometry(7, 64 + SPS_SIZE_TOLERANCE + 1, 64, Some((64, 64))).unwrap_err();
        assert!(format!("{err}").contains("codes"), "{err}");
        // The audit's numbers, verbatim: a small declaration in
        // front of a near-cap coded picture.
        assert!(check_coded_geometry(7, 16_384, 16_384, Some((64, 64))).is_err());
        // The per-axis cap binds even when the declaration agrees.
        check_coded_geometry(7, MAX_DECODE_DIMENSION, 8, Some((MAX_DECODE_DIMENSION, 8))).unwrap();
        assert!(
            check_coded_geometry(
                7,
                MAX_DECODE_DIMENSION + 1,
                8,
                Some((MAX_DECODE_DIMENSION, 8))
            )
            .is_err()
        );
        // No declared size: the pixel budget binds the coded
        // picture directly, at the budget exactly and past it.
        check_coded_geometry(7, 8192, 8192, None).unwrap();
        assert!(check_coded_geometry(7, 8192, 8193, None).is_err());
    }

    #[test]
    fn announced_size_rule_bounds() {
        check_announced_size(7, MAX_DECODE_DIMENSION, 8).unwrap();
        check_announced_size(7, 8192, 8192).unwrap(); // exactly the pixel budget
        assert!(check_announced_size(7, MAX_DECODE_DIMENSION + 1, 8).is_err());
        assert!(check_announced_size(7, 8192, 8193).is_err());
    }

    #[test]
    fn deflated_metadata_is_never_inflated() {
        // R2-11: with the crate's deflate feature off, a deflated
        // metadata payload (here 1 MiB of zeros, a few KB on the
        // wire) comes back as no metadata at all instead of
        // being inflated toward the crate's 1 GiB cap.
        use std::io::Write;
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&vec![0u8; 1 << 20]).unwrap();
        let deflated = encoder.finish().unwrap();
        assert!(deflated.len() < 8192, "deflate test payload sanity");
        assert_eq!(
            oxideav_heif::decode::inflate_metadata(&deflated).unwrap(),
            None
        );
    }

    #[test]
    fn metadata_over_the_cap_is_dropped() {
        let at_cap = vec![0u8; crate::color::MAX_META_BYTES];
        assert_eq!(
            cap_metadata(Some(at_cap.clone()), crate::color::MAX_META_BYTES),
            Some(at_cap)
        );
        let over_cap = vec![0u8; crate::color::MAX_META_BYTES + 1];
        assert_eq!(
            cap_metadata(Some(over_cap), crate::color::MAX_META_BYTES),
            None
        );
        assert_eq!(cap_metadata(None, crate::color::MAX_META_BYTES), None);
    }

    #[test]
    fn an_alternative_s_gain_map_is_held_to_the_decode_limits() {
        // Should the primary (item 1) fail to decode, the decoder
        // falls back to its `altr` alternative (item 2) and
        // decodes that item's gain map (item 3, the second input
        // of the `tmap` item 4) along with it. The gain map
        // announces a size past the decode limit, so validation
        // must refuse the file, though the gain map belongs to no
        // graph the primary's or the alternative's walk reaches.
        let infe = |id: u16, kind: &[u8; 4]| {
            let mut body = id.to_be_bytes().to_vec();
            body.extend_from_slice(&[0, 0]);
            body.extend_from_slice(kind);
            body.push(0);
            full_box(b"infe", 2, &body)
        };
        let ispe = |width: u32, height: u32| {
            let mut body = width.to_be_bytes().to_vec();
            body.extend_from_slice(&height.to_be_bytes());
            full_box(b"ispe", 0, &body)
        };
        let mut meta = full_box(b"hdlr", 0, b"\0\0\0\0pict\0\0\0\0\0\0\0\0\0\0\0\0\0");
        meta.extend(full_box(b"pitm", 0, &1u16.to_be_bytes()));
        let mut iinf = 4u16.to_be_bytes().to_vec();
        for (id, kind) in [(1, b"hvc1"), (2, b"hvc1"), (3, b"hvc1"), (4, b"tmap")] {
            iinf.extend(infe(id, kind));
        }
        meta.extend(full_box(b"iinf", 0, &iinf));
        // The `tmap` item 4 derives from item 2 and gain map 3.
        let dimg = [4u16, 2, 2, 3].map(u16::to_be_bytes).concat();
        meta.extend(full_box(b"iref", 0, &bmff_box(b"dimg", &dimg)));
        let mut iprp = bmff_box(b"ipco", &[ispe(64, 64), ispe(20_000, 48)].concat());
        let mut ipma = 4u32.to_be_bytes().to_vec();
        for (id, property) in [(1u16, 1u8), (2, 1), (3, 2), (4, 1)] {
            ipma.extend_from_slice(&id.to_be_bytes());
            ipma.extend_from_slice(&[1, property]);
        }
        iprp.extend(full_box(b"ipma", 0, &ipma));
        meta.extend(bmff_box(b"iprp", &iprp));
        // One `altr` group: group 100, items 1 and 2.
        let altr = [100u32, 2, 1, 2].map(u32::to_be_bytes).concat();
        meta.extend(bmff_box(b"grpl", &full_box(b"altr", 0, &altr)));
        let mut bytes = bmff_box(b"ftyp", b"heic\0\0\0\0mif1heic");
        bytes.extend(full_box(b"meta", 0, &meta));

        let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
        let meta = file.meta().unwrap();
        assert_eq!(meta.groups_containing(1, b"altr").len(), 1);
        assert!(gain_map_node(&file, 1).is_none());
        assert_eq!(gain_map_node(&file, 2).map(|node| node.item.id), Some(3));
        match validate_decode_budget(&bytes) {
            Err(EngineError::Decode(msg)) => {
                assert!(msg.contains("item 3 announces 20000x48"), "{msg}")
            }
            other => panic!("expected the gain map refusal, got {other:?}"),
        }
    }

    /// One ISO-BMFF box.
    fn bmff_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((8 + body.len()) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    /// One full box: version 0, flags 0, unless `version` says.
    fn full_box(kind: &[u8; 4], version: u8, body: &[u8]) -> Vec<u8> {
        let mut inner = vec![version, 0, 0, 0];
        inner.extend_from_slice(body);
        bmff_box(kind, &inner)
    }

    /// A HEIF file whose primary item (1) is a `rows` x `columns`
    /// grid whose every tile is the same image item (2), with
    /// `described` metadata items, each one `cdsc` reference to
    /// item 2. The container's structure is all there is: item
    /// 2 carries no coded data, so only the graph walks run.
    #[derive(Clone, Copy, Default)]
    struct SharedTile {
        rows: u8,
        columns: u8,
        described: u16,
        /// The length of the tile item's name.
        tile_name: usize,
        /// The length of each describing item's name.
        describer_name: usize,
        /// Property entries the tile carries besides its `ispe`,
        /// each naming the same empty property of a type the
        /// parser does not know.
        extra_entries: usize,
        /// Makes the tile a `tili` item whose `tilC` lists 255
        /// tile properties.
        tile_properties: Option<TileProperties>,
        /// Gives the tile an alpha auxiliary image, and a `prem`
        /// reference that names it this many times.
        premultiplied: Option<u16>,
        /// Makes the primary item a `tmap` whose base image is
        /// the tile and whose gain map image is the grid.
        tone_mapped: bool,
    }

    /// What the 255 tile properties of a [`SharedTile`]'s `tilC`
    /// name.
    #[derive(Clone, Copy)]
    enum TileProperties {
        /// Each the same property, of this many bytes.
        Named(usize),
        /// No property at all: each entry's index is 0.
        Unnamed,
    }

    impl SharedTile {
        fn new(rows: u8, columns: u8) -> Self {
            SharedTile {
                rows,
                columns,
                ..SharedTile::default()
            }
        }

        fn build(self) -> Vec<u8> {
            let tiles = u16::from(self.rows) * u16::from(self.columns);
            // Items 1 (the grid) and 2 (the tile), the describing
            // items, then the alpha image and the `tmap` if any.
            let mut next_id = 3 + self.described;
            let mut take_id = || {
                next_id += 1;
                next_id - 1
            };
            let alpha_id = self.premultiplied.map(|_| take_id());
            let tmap_id = self.tone_mapped.then(&mut take_id);
            let mut out = bmff_box(b"ftyp", b"heic\0\0\0\0mif1heic");
            let mut meta = full_box(b"hdlr", 0, b"\0\0\0\0pict\0\0\0\0\0\0\0\0\0\0\0\0\0");
            meta.extend(full_box(b"pitm", 0, &tmap_id.unwrap_or(1).to_be_bytes()));
            // iloc v1: the grid descriptor, 8 bytes at idat offset
            // 0, which a `tmap` item reads as its payload too.
            let located: Vec<u16> = std::iter::once(1).chain(tmap_id).collect();
            let mut iloc = vec![0x44, 0x00];
            iloc.extend_from_slice(&(located.len() as u16).to_be_bytes());
            for id in located {
                iloc.extend_from_slice(&id.to_be_bytes());
                iloc.extend_from_slice(&[0, 1, 0, 0, 0, 1]);
                iloc.extend_from_slice(&0u32.to_be_bytes());
                iloc.extend_from_slice(&8u32.to_be_bytes());
            }
            meta.extend(full_box(b"iloc", 1, &iloc));
            let infe = |id: u16, kind: &[u8; 4], name: usize| {
                let mut body = id.to_be_bytes().to_vec();
                body.extend_from_slice(&[0, 0]);
                body.extend_from_slice(kind);
                body.resize(body.len() + name, b'n');
                body.push(0);
                full_box(b"infe", 2, &body)
            };
            let tile_type = if self.tile_properties.is_some() {
                b"tili"
            } else {
                b"hvc1"
            };
            let mut entries = vec![infe(1, b"grid", 0), infe(2, tile_type, self.tile_name)];
            for id in 0..self.described {
                entries.push(infe(3 + id, b"Exif", self.describer_name));
            }
            entries.extend(alpha_id.map(|id| infe(id, b"hvc1", 0)));
            entries.extend(tmap_id.map(|id| infe(id, b"tmap", 0)));
            let mut iinf = (entries.len() as u16).to_be_bytes().to_vec();
            iinf.extend(entries.concat());
            meta.extend(full_box(b"iinf", 0, &iinf));
            // One reference box: its type, its source, its targets.
            let reference = |kind: &[u8; 4], from: u16, to: &[u16]| {
                let mut body = from.to_be_bytes().to_vec();
                body.extend_from_slice(&(to.len() as u16).to_be_bytes());
                for id in to {
                    body.extend_from_slice(&id.to_be_bytes());
                }
                bmff_box(kind, &body)
            };
            let mut iref = reference(b"dimg", 1, &vec![2; usize::from(tiles)]);
            for id in 0..self.described {
                iref.extend(reference(b"cdsc", 3 + id, &[2]));
            }
            if let (Some(alpha), Some(count)) = (alpha_id, self.premultiplied) {
                iref.extend(reference(b"auxl", alpha, &[2]));
                iref.extend(reference(b"prem", 2, &vec![alpha; usize::from(count)]));
            }
            if let Some(tmap) = tmap_id {
                iref.extend(reference(b"dimg", tmap, &[2, 1]));
            }
            meta.extend(full_box(b"iref", 0, &iref));
            // The properties, numbered from 1 in `ipco` order.
            let mut ispe = 64u32.to_be_bytes().to_vec();
            ispe.extend_from_slice(&64u32.to_be_bytes());
            let mut ipco = full_box(b"ispe", 0, &ispe);
            let mut tile_entries = vec![0x01];
            let mut next = 2u8;
            let empty = next;
            if self.extra_entries > 0 {
                ipco.extend(bmff_box(b"zzzz", b""));
                next += 1;
            }
            if let Some(names) = self.tile_properties {
                let mut tilc = 64u32.to_be_bytes().to_vec();
                tilc.extend_from_slice(&64u32.to_be_bytes());
                tilc.push(0);
                tilc.extend_from_slice(b"hvc1");
                let tipa = match names {
                    TileProperties::Named(bytes) => {
                        let named = u16::from(next);
                        ipco.extend(bmff_box(b"yyyy", &vec![7; bytes]));
                        next += 1;
                        // Flags 1: 16-bit property indices.
                        let mut tipa = vec![0, 0, 0, 1, 255];
                        for _ in 0..255 {
                            tipa.extend_from_slice(&named.to_be_bytes());
                        }
                        tipa
                    }
                    // Flags 0: 8-bit property indices.
                    TileProperties::Unnamed => [vec![0, 0, 0, 0, 255], vec![0; 255]].concat(),
                };
                tilc.extend(bmff_box(b"tipa", &tipa));
                ipco.extend(full_box(b"tilC", 0, &tilc));
                tile_entries.push(next);
                next += 1;
            }
            let mut rows = vec![(2u16, tile_entries)];
            if let Some(alpha) = alpha_id {
                let urn = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0";
                ipco.extend(full_box(b"auxC", 0, urn));
                rows.push((alpha, vec![0x01, next]));
            }
            let mut iprp = bmff_box(b"ipco", &ipco);
            let mut ipma = (rows.len() as u32).to_be_bytes().to_vec();
            for (id, row) in rows {
                ipma.extend_from_slice(&id.to_be_bytes());
                ipma.push(row.len() as u8);
                ipma.extend_from_slice(&row);
            }
            iprp.extend(full_box(b"ipma", 0, &ipma));
            // One more `ipma` box (each with its own flags) for
            // every 255 further entries.
            let mut left = self.extra_entries;
            let mut flags = 2u32;
            while left > 0 {
                let n = left.min(255);
                left -= n;
                let mut body = flags.to_be_bytes().to_vec();
                body.extend_from_slice(&1u32.to_be_bytes());
                body.extend_from_slice(&[0, 2, n as u8]);
                body.resize(body.len() + n, empty);
                iprp.extend(bmff_box(b"ipma", &body));
                flags += 2;
            }
            meta.extend(bmff_box(b"iprp", &iprp));
            let width = 64 * u16::from(self.columns);
            let height = 64 * u16::from(self.rows);
            let mut grid = vec![0, 0, self.rows - 1, self.columns - 1];
            grid.extend_from_slice(&width.to_be_bytes());
            grid.extend_from_slice(&height.to_be_bytes());
            meta.extend(bmff_box(b"idat", &grid));
            out.extend(full_box(b"meta", 0, &meta));
            out
        }
    }

    fn shared_tile_heif(rows: u8, columns: u8, described: u16) -> Vec<u8> {
        SharedTile {
            described,
            ..SharedTile::new(rows, columns)
        }
        .build()
    }

    /// The cost [`check_item_graphs`] works out for the primary
    /// item's graph.
    fn primary_graph_cost(bytes: &[u8]) -> Cost {
        let file = oxideav_heif::HeifFile::parse_borrowed(bytes).unwrap();
        let meta = file.meta().unwrap();
        let mut cost = GraphCost::new(meta, bytes.len() as u64);
        let primary = file.primary_item().unwrap().id;
        cost.walk(primary, 0, Cost::LIMIT)
            .plus(Cost::held(NODE_BYTES))
    }

    #[test]
    fn an_item_graph_that_revisits_a_described_tile_is_refused_before_it_is_built() {
        // A 255x255 grid whose 65,025 tiles all name one item,
        // which 64 metadata items describe. The graph builder
        // visits the tile once per reference and looks up every
        // describing item on each visit. The bound refuses it
        // from the tables, before the builder runs, in the probe
        // and the decode alike.
        let bytes = shared_tile_heif(255, 255, 64);
        let started = std::time::Instant::now();
        assert_eq!(probe_dimensions(&bytes), None);
        match decode_heic(&bytes) {
            Err(EngineError::Decode(msg)) => assert!(msg.contains("item graph"), "{msg}"),
            other => panic!("expected the item graph refusal, got {other:?}"),
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "the refusals took {:?}",
            started.elapsed()
        );
        let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
        assert!(check_item_graphs(&file, bytes.len()).is_err());
    }

    #[test]
    fn an_item_graph_too_large_to_hold_is_refused_before_it_is_built() {
        // The builder copies an item into the graph on every
        // visit, with its name, its parsed properties, and the
        // entries of the items that describe it, and a `tili`
        // visit also resolves each tile property its `tilC`
        // names. Each of these graphs would hold more than the
        // bound once built, and the check refuses it from the
        // tables first. The grids are kept small enough that,
        // were a check missing, the build would still end and
        // the assertion fail.
        let cases = [
            (
                "a long tile name",
                SharedTile {
                    tile_name: 8 << 10,
                    ..SharedTile::new(128, 128)
                },
            ),
            (
                "a long name on the describing item",
                SharedTile {
                    described: 1,
                    describer_name: 8 << 10,
                    ..SharedTile::new(128, 128)
                },
            ),
            (
                "many property entries",
                SharedTile {
                    extra_entries: 1_000,
                    ..SharedTile::new(32, 32)
                },
            ),
            (
                "tile properties",
                SharedTile {
                    tile_properties: Some(TileProperties::Named(64 << 10)),
                    ..SharedTile::new(4, 4)
                },
            ),
        ];
        for (name, spec) in cases {
            let bytes = spec.build();
            let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
            match check_item_graphs(&file, bytes.len()) {
                Err(EngineError::Decode(msg)) => {
                    assert!(msg.contains("item graph"), "{name}: {msg}")
                }
                other => panic!("{name}: expected the item graph refusal, got {other:?}"),
            }
            assert_eq!(probe_dimensions(&bytes), None, "{name}");
            assert!(decode_heic(&bytes).is_err(), "{name}");
        }
        // The property entries take little work to walk: what
        // the graph would hold is what refuses them.
        let entries = SharedTile {
            extra_entries: 1_000,
            ..SharedTile::new(32, 32)
        }
        .build();
        let cost = primary_graph_cost(&entries);
        assert!(cost.work < MAX_GRAPH_WORK / 10, "{cost:?}");
        assert!(cost.held > MAX_GRAPH_BYTES, "{cost:?}");
    }

    #[test]
    fn a_decode_is_charged_for_both_copies_of_the_graph_it_holds() {
        // The decode builds the primary item's graph, then builds
        // it again while it still holds the first copy. One copy
        // of this graph is within the byte bound; the two are not,
        // and the check refuses the file for what it would hold.
        let bytes = SharedTile {
            extra_entries: 1_000,
            ..SharedTile::new(18, 18)
        }
        .build();
        let one = primary_graph_cost(&bytes);
        assert!(one.held < MAX_GRAPH_BYTES, "{one:?}");
        assert!(one.work < MAX_GRAPH_WORK / 20, "{one:?}");
        let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
        match check_item_graphs(&file, bytes.len()) {
            Err(EngineError::Decode(msg)) => assert!(msg.contains("would hold"), "{msg}"),
            other => panic!("expected the item graph refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_tone_mapped_primary_is_charged_for_its_gain_map_graph_too() {
        // A `tmap` primary's gain map image is its second input,
        // already in its graph, and the decode builds the gain
        // map's graph once more while it holds both copies of the
        // primary's. Here the gain map image is the grid: the two
        // copies of the primary's graph fit the byte bound, and
        // the gain map's on top of them does not.
        let bytes = SharedTile {
            extra_entries: 1_000,
            tone_mapped: true,
            ..SharedTile::new(13, 13)
        }
        .build();
        let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
        let meta = file.meta().unwrap();
        let primary = file.primary_item().unwrap();
        assert_eq!(primary.item_type, oxideav_heif::meta::ITEM_TYPE_TMAP);
        assert_eq!(meta.derivation_inputs(primary.id), [2, 1]);
        let mut cost = GraphCost::new(meta, bytes.len() as u64);
        assert_eq!(cost.gain_map(primary.id), Some((primary.id, Some(1))));
        let one = primary_graph_cost(&bytes);
        let gain = cost.walk(1, 0, Cost::LIMIT).plus(Cost::held(NODE_BYTES));
        assert!(2 * one.held < MAX_GRAPH_BYTES, "{one:?}");
        assert!(
            2 * one.held + gain.held > MAX_GRAPH_BYTES,
            "{one:?} {gain:?}"
        );
        match check_item_graphs(&file, bytes.len()) {
            Err(EngineError::Decode(msg)) => assert!(msg.contains("would hold"), "{msg}"),
            other => panic!("expected the item graph refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_tili_visit_is_charged_a_slot_for_every_tile_entry() {
        // A `tili` visit resolves its tile properties into a list
        // with room for every entry its `tilC` lists, an entry
        // that names no property included. Here none of the
        // entries names a property, so the room is all the list
        // holds.
        let bytes = SharedTile {
            tile_properties: Some(TileProperties::Unnamed),
            ..SharedTile::new(40, 40)
        }
        .build();
        let one = primary_graph_cost(&bytes);
        assert!(one.held >= 40 * 40 * 255 * ENTRY_BYTES, "{one:?}");
        let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
        match check_item_graphs(&file, bytes.len()) {
            Err(EngineError::Decode(msg)) => assert!(msg.contains("would hold"), "{msg}"),
            other => panic!("expected the item graph refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_visit_that_finds_an_alpha_image_is_charged_for_the_prem_lookup() {
        // A visit that finds an item's alpha image reads the
        // item's `prem` references, copying their targets, to
        // learn whether its samples are premultiplied. Here the
        // tile's `prem` reference is a long one, and the grid
        // visits the tile once for each of its cells.
        let bytes = SharedTile {
            premultiplied: Some(20_000),
            ..SharedTile::new(64, 64)
        }
        .build();
        let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
        // The builder takes that path on a visit of the tile.
        let tile = oxideav_heif::build_graph(&file, 2).unwrap();
        assert!(tile.alpha.is_some() && tile.premultiplied_alpha);
        match check_item_graphs(&file, bytes.len()) {
            Err(EngineError::Decode(msg)) => assert!(msg.contains("steps to walk"), "{msg}"),
            other => panic!("expected the item graph refusal, got {other:?}"),
        }
        // The same graph with a `prem` reference of one target is
        // well within both bounds.
        let short = SharedTile {
            premultiplied: Some(1),
            ..SharedTile::new(64, 64)
        }
        .build();
        let file = oxideav_heif::HeifFile::parse_borrowed(&short).unwrap();
        let total = decode_graph_cost(&file, short.len());
        assert!(total.work < MAX_GRAPH_WORK / 20, "{total:?}");
        assert!(total.held < MAX_GRAPH_BYTES / 2, "{total:?}");
    }

    #[test]
    fn a_property_with_a_body_is_charged_at_least_one_allocation() {
        // A parse that keeps any of its body takes an allocation,
        // and the allocator hands out no block smaller than 32
        // bytes on a 64-bit target, however few bytes it holds:
        // one byte of a box the parser does not know, or of an
        // ICC profile.
        const SMALLEST_BLOCK: u64 = 32;
        for (box_type, body) in [(*b"zzzz", vec![7]), (*b"colr", b"prof\x07".to_vec())] {
            let len = body.len() as u64;
            let raw = oxideav_heif::meta::RawProperty::new(box_type, None, body, 8 + len as usize);
            let cost = PropertyCost::of(&raw);
            assert!(
                cost.parse.held >= SMALLEST_BLOCK.max(len),
                "{}: {:?}",
                box_type.escape_ascii(),
                cost.parse
            );
        }
        // An empty body is kept without an allocation.
        let empty = oxideav_heif::meta::RawProperty::new(*b"zzzz", None, Vec::new(), 8);
        assert_eq!(PropertyCost::of(&empty).parse.held, 0);
    }

    #[test]
    fn item_graphs_within_the_walk_budget_still_build() {
        // The same shared tile with no metadata items costs a
        // few cheap steps per visit and stays within the bound
        // in a grid of as many 64 px tiles as the decode budget
        // holds, as does a grid whose tiles are few, and tiles
        // and describing items with names of a usual length: the
        // probe sizes them all.
        let named = SharedTile {
            described: 2,
            tile_name: 16,
            describer_name: 16,
            ..SharedTile::new(64, 64)
        };
        for spec in [
            SharedTile::new(128, 128),
            SharedTile {
                described: 4,
                ..SharedTile::new(16, 16)
            },
            named,
        ] {
            let bytes = spec.build();
            let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
            check_item_graphs(&file, bytes.len()).unwrap();
            let expected = (64 * u32::from(spec.columns), 64 * u32::from(spec.rows));
            assert_eq!(probe_dimensions(&bytes), Some(expected));
        }
        // Every fixture passes, with room to spare, counted as
        // the decode holds its graphs.
        for bytes in [GRID, MAIN10, VUI_LIMITED, P3, ORIENT6] {
            let file = oxideav_heif::HeifFile::parse_borrowed(bytes).unwrap();
            let cost = decode_graph_cost(&file, bytes.len());
            assert!(cost.work < MAX_GRAPH_WORK / 1000, "{cost:?}");
            assert!(cost.held < MAX_GRAPH_BYTES / 1000, "{cost:?}");
        }
    }

    #[test]
    fn the_walk_estimate_stops_where_the_builder_stops() {
        // A reference cycle: the builder refuses it at the
        // repeat, and the estimate counts the walk up to there
        // without looping.
        let mut bytes = shared_tile_heif(1, 1, 0);
        // Point the grid's one tile back at the grid itself.
        let at = bytes.windows(4).position(|w| w == b"dimg").unwrap();
        bytes[at + 8..at + 10].copy_from_slice(&1u16.to_be_bytes());
        let file = oxideav_heif::HeifFile::parse_borrowed(&bytes).unwrap();
        assert!(oxideav_heif::build_primary_graph(&file).is_err());
        check_item_graphs(&file, bytes.len()).unwrap();
        assert_eq!(probe_dimensions(&bytes), None);
    }
}
