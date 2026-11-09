//! The expansion bound (issue #443): an upper bound on what usvg
//! builds for a document, computed from the parsed XML before
//! usvg builds anything.
//!
//! usvg converts several kinds of reference afresh each time it
//! follows one instead of sharing one conversion: an `feImage`
//! naming an element converts that element again for every
//! filter instance, filters, masks, and clip paths convert per
//! referencing element (unless every unit is user space), a
//! marker converts once per vertex, a pattern is copied for
//! every path it paints, and `<use>` copies its target. Chained,
//! these multiply: K references per level across N levels build
//! K^N nodes from a document of a few kilobytes, all allocated
//! before the post-parse count in `prepare` runs. The recursion
//! has no guard beyond direct self-references either, so a
//! three-link reference cycle, or a long enough chain, overflows
//! the stack.
//!
//! The walk below follows the same references over the parsed
//! XML and charges each one what usvg converts when it follows
//! it: one node per element instance and per character of text,
//! plus the `data:` payload bytes of every image instance, and
//! the depth of the recursion. Results are memoized per element
//! and role, and it stops as soon as a partial sum passes a
//! limit.
//!
//! The walk is linear in the document because every step it
//! takes spends a unit of fuel: every evaluation of an element
//! in a role, a memoized result reused included, and every
//! declared value examined for a reference, a value shared by
//! many elements through an id or an `inherit`, `context-*`, or
//! `<use>` list included. The fuel is a small multiple of the
//! node limit, so the total work the walk does is bounded by it
//! however the references fan out, and a document that would
//! make the walk do more is refused (as `Nodes`) instead. The
//! facts the walk reads are gathered once, under the same kind
//! of bound: matching the style sheet's kept rules against the
//! elements is capped at [`MAX_STYLE_MATCHES`] comparisons.
//!
//! It never undercounts those conversions. A reference whose
//! source is ambiguous without usvg's full style resolution
//! (`inherit`, a `<use>` site's inherited properties,
//! `context-fill`, an id carried by several elements) is
//! charged at the costliest value it could take, and an element
//! is charged wherever it is drawn, never shared. The one
//! approximation is the vertex count markers multiply by: an
//! elliptical arc counts as [`ARC_SEGMENTS`] vertices (see
//! there). One conversion builds at most a few tree nodes (a
//! group around a path, a second path for a split paint order),
//! so the bound tracks the post-parse count in `prepare` to
//! within a small factor while refusing a K^N document at K^N.
//! Cycles are refused, except marker cycles, which usvg breaks
//! itself (a marker never draws inside its own content) and
//! which therefore cost nothing here either.

use std::collections::HashMap;

use roxmltree::Node;

/// What one conversion builds: tree nodes (elements, plus one
/// per character of text) and the bytes of the `data:` image
/// payloads usvg decodes for it, and how deep it recurses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Cost {
    pub nodes: u64,
    pub bytes: u64,
    /// The levels of conversion recursion below and including
    /// this one: what a memoized result adds to the stack each
    /// time it is reached again, from wherever it is reached.
    pub height: usize,
}

impl Cost {
    const ZERO: Cost = Cost {
        nodes: 0,
        bytes: 0,
        height: 0,
    };

    fn nodes(nodes: u64) -> Cost {
        Cost {
            nodes,
            ..Cost::ZERO
        }
    }

    fn plus(self, other: Cost) -> Cost {
        Cost {
            nodes: self.nodes.saturating_add(other.nodes),
            bytes: self.bytes.saturating_add(other.bytes),
            height: self.height.max(other.height),
        }
    }

    fn times(self, k: u64) -> Cost {
        Cost {
            nodes: self.nodes.saturating_mul(k),
            bytes: self.bytes.saturating_mul(k),
            height: self.height,
        }
    }

    fn max(self, other: Cost) -> Cost {
        Cost {
            nodes: self.nodes.max(other.nodes),
            bytes: self.bytes.max(other.bytes),
            height: self.height.max(other.height),
        }
    }
}

/// The limits a walk enforces. `depth` caps the conversion
/// recursion: element nesting plus reference hops, the same
/// recursion usvg performs on the thread's stack. The bound's
/// `height` is the depth the document reached.
pub(super) struct Limits {
    pub nodes: u64,
    pub bytes: u64,
    pub depth: usize,
}

/// Why a document was refused.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Refusal {
    Nodes,
    Bytes,
    Cycle,
    Depth,
    /// Bounding the document needs more pre-parse work than
    /// its structure is allowed: the style sheet's kept rules
    /// against the elements passed [`MAX_STYLE_MATCHES`]. Kept
    /// separate from `Nodes` so the message names the cause.
    Work,
}

/// The most element-by-rule comparisons [`Facts::gather`] makes
/// when it matches the kept style rules against the elements. A
/// document with `R` kept rules and `E` elements costs `R * E`
/// comparisons; both are bounded only by the node cap, so their
/// product is quadratic, and a crafted document reaches it in a
/// few kilobytes. The cap is well above any real style sheet
/// (even a few hundred rules against the whole 10,000-node cap
/// stays under it) and refuses the quadratic case before it
/// runs.
const MAX_STYLE_MATCHES: u64 = 4_000_000;

/// The segments usvg may cut one elliptical arc into. kurbo
/// subdivides an arc into up to `(11.163 * r)^(1/6)` cubics per
/// full turn, so 64 covers any arc whose radius, after kurbo
/// scales it to reach its end point, stays below about 6e9
/// units. A rect's rounded corners and a circle or ellipse are
/// four quarter arcs, 64 segments between them. A larger arc
/// can make more, and an extreme one hangs usvg's subdivision
/// whether or not a marker draws on it: that needs its own
/// bound on the path data, not this one.
const ARC_SEGMENTS: u64 = 64;

/// The presentation properties that reference other elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prop {
    Filter,
    Mask,
    ClipPath,
    MarkerStart,
    MarkerMid,
    MarkerEnd,
    Fill,
    Stroke,
}

/// The properties that inherit: a `<use>` passes its values
/// for these down to the copy of its target.
const INHERITED: [Prop; 5] = [
    Prop::MarkerStart,
    Prop::MarkerMid,
    Prop::MarkerEnd,
    Prop::Fill,
    Prop::Stroke,
];

impl Prop {
    fn index(self) -> usize {
        self as usize
    }

    /// The properties an attribute or declaration named `name`
    /// sets; the `marker` shorthand sets all three markers. Names
    /// compare ASCII case-insensitively, a superset of usvg's
    /// exact match.
    fn named(name: &str) -> &'static [Prop] {
        const TABLE: [(&str, &[Prop]); 9] = [
            ("filter", &[Prop::Filter]),
            ("mask", &[Prop::Mask]),
            ("clip-path", &[Prop::ClipPath]),
            ("marker-start", &[Prop::MarkerStart]),
            ("marker-mid", &[Prop::MarkerMid]),
            ("marker-end", &[Prop::MarkerEnd]),
            (
                "marker",
                &[Prop::MarkerStart, Prop::MarkerMid, Prop::MarkerEnd],
            ),
            ("fill", &[Prop::Fill]),
            ("stroke", &[Prop::Stroke]),
        ];
        TABLE
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map_or(&[], |(_, props)| props)
    }
}

/// The role a reference converts an element in; each role
/// converts differently, so memoization and cycle detection key
/// on the element and the role together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Role {
    /// `convert_element`: tree traversal, `<use>` content, and
    /// `feImage` targets.
    Element,
    /// The target of a `<use>`: a symbol draws its children.
    UseTarget,
    Filter,
    Mask,
    Clip,
    Marker,
    Pattern,
}

/// The kinds of element usvg resolves by walking a chain of
/// `href` links of the same kind (`HrefIter`, and the
/// `find_*`/`resolve_*_attr` helpers in usvg's parser).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum HrefKind {
    Gradient,
    Pattern,
    Filter,
}

fn href_kind(node: Node) -> Option<HrefKind> {
    match node.tag_name().name() {
        "linearGradient" | "radialGradient" => Some(HrefKind::Gradient),
        "pattern" => Some(HrefKind::Pattern),
        "filter" => Some(HrefKind::Filter),
        _ => None,
    }
}

/// Refuse a document in which usvg would follow an `href` chain
/// for ever (issue #443). usvg inherits a gradient's stops and
/// coordinates, a pattern's children and tiling, and a filter's
/// region and primitives from the element its `href` names, of
/// the same kind, and repeats that for the next `href`, and so
/// on. Its only guard (`HrefIter`) stops the walk when a link
/// points at the element being resolved or back at the one the
/// walk began at; a cycle that the starting element is not part
/// of, such as `#a`→`#b`→`#c`→`#b`, is followed with no end,
/// and the blocking thread never returns.
///
/// This decides what that walk does from every gradient,
/// pattern, and filter, with usvg's stop rule: the walk from an
/// element ends when its chain ends or comes back to the
/// element itself, and loops when the chain runs into a cycle
/// the element is not on. So a chain that ends, or a pure ring
/// of links, is not refused (no legitimate `href` inheritance
/// chain is), and a chain that leads into a ring from outside
/// it is.
///
/// Each element has at most one same-kind successor, so the
/// pass is linear: every element is walked onto once, and its
/// fate is recorded when the walk that reached it settles (the
/// chain ends, or it is a member of a ring). A later walk that
/// reaches a settled element stops there: at one whose chain
/// ends, it ends too; at a ring member, it came from outside
/// the ring, so usvg would loop from where it began. A shared
/// id costs nothing extra: the successor lookup reads at most
/// two candidates per id and kind.
fn check_href_cycles<'a, 'input>(
    doc: &'a roxmltree::Document<'input>,
    facts: &Facts<'a, 'input>,
) -> Result<(), Refusal> {
    // usvg's walk continues at the first element of the same
    // kind that an `href` names, other than the element it
    // stands on (it stops at a kind change). Of the elements
    // sharing an id, only the first two of each kind in
    // document order can be that element, so they are kept
    // instead of scanning every element sharing the id.
    let mut candidates: HashMap<(&str, HrefKind), [Option<Node<'a, 'input>>; 2]> = HashMap::new();
    for (&id, nodes) in &facts.ids {
        for &node in nodes {
            let Some(kind) = href_kind(node) else {
                continue;
            };
            let pair = candidates.entry((id, kind)).or_default();
            if let Some(slot) = pair.iter_mut().find(|slot| slot.is_none()) {
                *slot = Some(node);
            }
        }
    }
    let successor = |node: Node<'a, 'input>| -> Option<Node<'a, 'input>> {
        let kind = href_kind(node)?;
        node.attributes()
            .filter(|a| a.name().eq_ignore_ascii_case("href"))
            .filter_map(|a| iri_id(a.value()))
            .filter_map(|id| candidates.get(&(id, kind)))
            .flat_map(|pair| pair.iter().flatten().copied())
            .find(|target| *target != node)
    };
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fate {
        Unseen,
        /// On the chain being walked, at this position.
        OnPath(usize),
        /// Its chain ends: usvg stops from it, and from
        /// anything leading to it.
        Ends,
        /// A member of a pure ring: usvg stops from it when
        /// the walk comes back to it.
        Ring,
    }
    let count = doc.descendants().len() + 1;
    let mut fate = vec![Fate::Unseen; count];
    // A backstop on the pass's total work: every element is
    // walked onto at most once, so the steps across the whole
    // pass never pass the node count.
    let mut steps = count;
    let mut path: Vec<usize> = Vec::new();
    for node in doc.descendants().filter(|n| href_kind(*n).is_some()) {
        if fate[node.id().get_usize()] != Fate::Unseen {
            continue;
        }
        path.clear();
        let mut curr = node;
        let settled = loop {
            let curr_id = curr.id().get_usize();
            steps = steps.checked_sub(1).ok_or(Refusal::Nodes)?;
            fate[curr_id] = Fate::OnPath(path.len());
            path.push(curr_id);
            let Some(next) = successor(curr) else {
                break Fate::Ends;
            };
            match fate[next.id().get_usize()] {
                Fate::Unseen => curr = next,
                Fate::Ends => break Fate::Ends,
                // Back at the element the walk began at: every
                // element on the path is on the ring, and the
                // walk from each comes back to it.
                Fate::OnPath(0) => break Fate::Ring,
                // Into a ring the walk's origin is not on, met
                // on this walk or an earlier one: usvg follows
                // it with no end.
                Fate::OnPath(_) | Fate::Ring => return Err(Refusal::Cycle),
            }
        };
        for &id in &path {
            fate[id] = settled;
        }
    }
    Ok(())
}

/// Bound what usvg builds for `doc`, or say why it is refused.
pub(super) fn bound(doc: &roxmltree::Document, limits: Limits) -> Result<Cost, Refusal> {
    let facts = Facts::gather(doc)?;
    // usvg follows `href` on gradients, patterns, and filters
    // with a guard that misses cycles not passing through the
    // element it started from, so such a cycle hangs it before
    // it builds anything the walk below would count. Refuse it
    // first (issue #443).
    check_href_cycles(doc, &facts)?;
    let fuel = limits
        .nodes
        .saturating_mul(4)
        .saturating_add(4 * doc.descendants().len() as u64);
    let mut walk = Walk {
        facts: &facts,
        limits,
        memo: HashMap::new(),
        depth: 0,
        guard: usize::MAX,
        fuel,
    };
    walk.eval(doc.root_element(), Role::Element)
}

/// What the walk reads about the document, gathered once.
struct Facts<'a, 'input> {
    /// Every element carrying each id. usvg resolves a duplicate
    /// id to one of them (which one differs between `<use>` and
    /// other references); the walk charges the costliest.
    ids: HashMap<&'a str, Vec<Node<'a, 'input>>>,
    /// The values each element declares for each property, by
    /// node index: attributes, matching style sheet rules, and
    /// the `style` attribute. usvg's cascade keeps one of them;
    /// the walk charges the costliest.
    decls: Vec<Vec<(Prop, &'a str)>>,
    /// Whether a node sits inside the target of some `<use>`,
    /// where it inherits from the `<use>` site too.
    in_use: Vec<bool>,
    /// Every value declared anywhere, per property: what
    /// `inherit` on a non-inherited property can resolve to.
    anywhere: [Vec<&'a str>; 8],
    /// The inherited property values in force at any `<use>`:
    /// what a node inside a `<use>` target can inherit from the
    /// site that copies it.
    at_use: [Vec<&'a str>; 8],
}

impl<'a, 'input> Facts<'a, 'input> {
    fn gather(doc: &'a roxmltree::Document<'input>) -> Result<Self, Refusal> {
        let count = doc.descendants().len() + 1;
        let mut facts = Facts {
            ids: HashMap::new(),
            decls: vec![Vec::new(); count],
            in_use: vec![false; count],
            anywhere: Default::default(),
            at_use: Default::default(),
        };
        for node in doc.descendants().filter(Node::is_element) {
            for attr in node.attributes() {
                if attr.name().eq_ignore_ascii_case("id") {
                    facts.ids.entry(attr.value()).or_default().push(node);
                }
            }
        }
        let rules = facts.style_rules(doc);
        // Matching every kept rule against every element is
        // quadratic in the document, so it is capped: a
        // document whose rules and elements together would make
        // more than `MAX_STYLE_MATCHES` comparisons is refused
        // here, before the matching runs long.
        let mut matches_left = MAX_STYLE_MATCHES;
        for node in doc.descendants().filter(Node::is_element) {
            let mut decls = Vec::new();
            for attr in node.attributes() {
                for &prop in Prop::named(attr.name()) {
                    decls.push((prop, attr.value()));
                }
            }
            for rule in &rules {
                matches_left = matches_left.checked_sub(1).ok_or(Refusal::Work)?;
                if rule.selector.matches(&XmlNode(node)) {
                    for decl in &rule.declarations {
                        for &prop in Prop::named(decl.name) {
                            decls.push((prop, decl.value));
                        }
                    }
                }
            }
            for attr in node.attributes() {
                if attr.name().eq_ignore_ascii_case("style") {
                    for decl in simplecss::DeclarationTokenizer::from(attr.value()) {
                        for &prop in Prop::named(decl.name) {
                            decls.push((prop, decl.value));
                        }
                    }
                }
            }
            for &(prop, value) in &decls {
                if !is_inherit(value) {
                    facts.anywhere[prop.index()].push(value);
                }
            }
            facts.decls[node.id().get_usize()] = decls;
        }
        let uses: Vec<Node> = doc
            .descendants()
            .filter(|n| n.is_element() && n.tag_name().name() == "use")
            .collect();
        for &use_node in &uses {
            for target in facts.href_targets(use_node) {
                for node in target.descendants() {
                    facts.in_use[node.id().get_usize()] = true;
                }
            }
        }
        for &use_node in &uses {
            for prop in INHERITED {
                let values = facts.declared_up(use_node, prop);
                facts.at_use[prop.index()].extend(values);
            }
        }
        Ok(facts)
    }

    fn values(&self, node: Node<'a, 'input>, prop: Prop) -> impl Iterator<Item = &'a str> + '_ {
        self.decls[node.id().get_usize()]
            .iter()
            .filter(move |(p, _)| *p == prop)
            .map(|(_, v)| *v)
    }

    /// The values `prop` can take at `node` from the node and
    /// its document ancestors: the nearest one declaring it
    /// wins, and `inherit` defers to the next one up.
    fn declared_up(&self, node: Node<'a, 'input>, prop: Prop) -> Vec<&'a str> {
        let mut out = Vec::new();
        let mut current = Some(node);
        while let Some(n) = current {
            let mut declared = false;
            let mut inherit = false;
            for value in self.values(n, prop) {
                declared = true;
                if is_inherit(value) {
                    inherit = true;
                } else {
                    out.push(value);
                }
            }
            if declared && !inherit {
                break;
            }
            current = n.parent_element();
        }
        out
    }

    /// The values a non-inherited property can take at `node`:
    /// its own declarations, or anything declared anywhere when
    /// one of them is `inherit` (the parent it inherits from may
    /// be a `<use>` site, not the document parent).
    fn own(&self, node: Node<'a, 'input>, prop: Prop) -> Vec<&'a str> {
        let mut out: Vec<&'a str> = self.values(node, prop).collect();
        if out.iter().any(|v| is_inherit(v)) {
            out.extend(&self.anywhere[prop.index()]);
        }
        out
    }

    /// The values an inherited property can take at `node`,
    /// including what a `<use>` site passes down and what a
    /// `context-fill` or `context-stroke` resolves to.
    fn inherited(&self, node: Node<'a, 'input>, prop: Prop) -> Vec<&'a str> {
        let mut out = self.declared_up(node, prop);
        if self.in_use[node.id().get_usize()] {
            out.extend(&self.at_use[prop.index()]);
        }
        if out
            .iter()
            .any(|v| starts_with_ci(v.trim_start(), "context-"))
        {
            out.extend(&self.anywhere[Prop::Fill.index()]);
            out.extend(&self.anywhere[Prop::Stroke.index()]);
        }
        out
    }

    /// The elements a `url(#id)` in `value` can name.
    fn url_targets(&self, value: &str) -> Vec<Vec<Node<'a, 'input>>> {
        url_ids(value)
            .into_iter()
            .filter_map(|id| self.ids.get(id).cloned())
            .collect()
    }

    /// The elements the `href` attributes of `node` name.
    fn href_targets(&self, node: Node<'a, 'input>) -> Vec<Node<'a, 'input>> {
        node.attributes()
            .filter(|a| a.name().eq_ignore_ascii_case("href"))
            .filter_map(|a| iri_id(a.value()))
            .filter_map(|id| self.ids.get(id))
            .flatten()
            .copied()
            .collect()
    }

    /// The style sheet usvg applies (the text of every `<style>`
    /// element whose type is absent or `text/css`), keeping only
    /// the declarations that can cost something: a reference to
    /// an element that converts content, `inherit`, or a context
    /// paint. Matching every rule against every element is the
    /// expensive part of the walk, and a rule that sets only a
    /// color or a gradient cannot add to the bound; dropping its
    /// declarations only lets an inherited value from further up
    /// through, which overestimates.
    fn style_rules(&self, doc: &'a roxmltree::Document) -> Vec<simplecss::Rule<'a>> {
        let mut sheet = simplecss::StyleSheet::new();
        for node in doc.descendants() {
            if !(node.is_element() && node.tag_name().name() == "style") {
                continue;
            }
            if !matches!(node.attribute("type"), None | Some("text/css")) {
                continue;
            }
            if let Some(text) = node.text() {
                sheet.parse_more(text);
            }
        }
        let costly = |value: &str| {
            is_inherit(value)
                || starts_with_ci(value.trim_start(), "context-")
                || url_ids(value).into_iter().any(|id| {
                    self.ids.get(id).is_some_and(|nodes| {
                        nodes.iter().any(|n| {
                            matches!(
                                n.tag_name().name(),
                                "filter" | "mask" | "clipPath" | "marker" | "pattern"
                            )
                        })
                    })
                })
        };
        let mut rules = sheet.rules;
        for rule in &mut rules {
            rule.declarations
                .retain(|d| !Prop::named(d.name).is_empty() && costly(d.value));
        }
        rules.retain(|rule| !rule.declarations.is_empty());
        rules
    }
}

enum Slot {
    /// Being evaluated, at this stack depth.
    Active(usize),
    Done(Cost),
}

struct Walk<'f, 'a, 'input> {
    facts: &'f Facts<'a, 'input>,
    limits: Limits,
    memo: HashMap<(usize, Role), Slot>,
    depth: usize,
    /// The shallowest active frame a marker cycle cut short.
    /// Frames above it computed a cost that holds only inside
    /// that marker, so they are not memoized.
    guard: usize,
    /// Evaluations left: a backstop for the frames the marker
    /// guard keeps out of the memo.
    fuel: u64,
}

impl<'a, 'input> Walk<'_, 'a, 'input> {
    fn eval(&mut self, node: Node<'a, 'input>, role: Role) -> Result<Cost, Refusal> {
        // Every visit spends a unit, a memoized result reused
        // included, so the number of visits across the whole
        // walk is bounded however references fan out onto an
        // element shared by many sites (issue #443).
        self.fuel = self.fuel.checked_sub(1).ok_or(Refusal::Nodes)?;
        let key = (node.id().get_usize(), role);
        match self.memo.get(&key) {
            // A memoized conversion repeats its whole recursion
            // below wherever it is reached again.
            Some(Slot::Done(cost)) if self.depth + cost.height > self.limits.depth => {
                return Err(Refusal::Depth);
            }
            Some(Slot::Done(cost)) => return Ok(*cost),
            Some(Slot::Active(depth)) if role == Role::Marker => {
                self.guard = self.guard.min(*depth);
                return Ok(Cost::ZERO);
            }
            Some(Slot::Active(_)) => return Err(Refusal::Cycle),
            None => {}
        }
        if self.depth >= self.limits.depth {
            return Err(Refusal::Depth);
        }
        let depth = self.depth;
        self.memo.insert(key, Slot::Active(depth));
        self.depth += 1;
        let result = self.convert(node, role);
        self.depth -= 1;
        let mut cost = result?;
        cost.height += 1;
        if self.guard >= depth {
            self.guard = usize::MAX;
            self.memo.insert(key, Slot::Done(cost));
        } else {
            self.memo.remove(&key);
        }
        Ok(cost)
    }

    /// Add `more` to `sum`, refusing as soon as a limit is
    /// passed: a partial sum past it means the whole is too.
    fn add(&self, sum: Cost, more: Cost) -> Result<Cost, Refusal> {
        let sum = sum.plus(more);
        if sum.nodes > self.limits.nodes {
            return Err(Refusal::Nodes);
        }
        if sum.bytes > self.limits.bytes {
            return Err(Refusal::Bytes);
        }
        Ok(sum)
    }

    fn children(&mut self, node: Node<'a, 'input>, mut sum: Cost) -> Result<Cost, Refusal> {
        for child in node.children().filter(Node::is_element) {
            let cost = self.eval(child, Role::Element)?;
            sum = self.add(sum, cost)?;
        }
        Ok(sum)
    }

    /// The costliest element any `url(#id)` in `values` names,
    /// in `role`; with `sum`, the `url()`s in one value add up
    /// (a filter list applies every filter it names).
    fn refs(&mut self, values: &[&str], role: Role, sum: bool) -> Result<Cost, Refusal> {
        let mut best = Cost::ZERO;
        for value in values {
            // Every value examined spends a unit, so a list
            // grown by an `inherit`, `context-*`, or `<use>`
            // fan-out (which can repeat a value once per
            // declaration) cannot make the walk do unbounded
            // work before it finds the references in it.
            self.fuel = self.fuel.checked_sub(1).ok_or(Refusal::Nodes)?;
            let mut total = Cost::ZERO;
            for targets in self.facts.url_targets(value) {
                let mut cost = Cost::ZERO;
                for target in targets {
                    cost = cost.max(self.eval(target, role)?);
                }
                total = if sum {
                    self.add(total, cost)?
                } else {
                    total.max(cost)
                };
            }
            best = best.max(total);
        }
        Ok(best)
    }

    fn convert(&mut self, node: Node<'a, 'input>, role: Role) -> Result<Cost, Refusal> {
        let name = node.tag_name().name();
        match role {
            Role::Element => self.element(node, name),
            Role::UseTarget if name == "symbol" => self.children(node, Cost::nodes(2)),
            Role::UseTarget => self.eval(node, Role::Element),
            Role::Filter if name == "filter" => self.filter(node),
            Role::Mask if name == "mask" => {
                let sum = self.children(node, Cost::nodes(1))?;
                let linked = self.refs(&self.facts.own(node, Prop::Mask), Role::Mask, false)?;
                self.add(sum, linked)
            }
            Role::Clip if name == "clipPath" => {
                let sum = self.children(node, Cost::nodes(1))?;
                let linked = self.refs(&self.facts.own(node, Prop::ClipPath), Role::Clip, false)?;
                self.add(sum, linked)
            }
            // A group and a clip path per instance.
            Role::Marker if name == "marker" => self.children(node, Cost::nodes(2)),
            Role::Pattern if name == "pattern" => {
                if node.children().any(|c| c.is_element()) {
                    self.children(node, Cost::nodes(1))
                } else {
                    self.linked(node, Role::Pattern)
                }
            }
            // A reference to an element of the wrong kind
            // converts nothing.
            _ => Ok(Cost::ZERO),
        }
    }

    /// The costliest element the `href` of `node` names, in
    /// `role`: a filter or pattern without children of its own
    /// takes them from the one it links to.
    fn linked(&mut self, node: Node<'a, 'input>, role: Role) -> Result<Cost, Refusal> {
        let mut best = Cost::ZERO;
        for target in self.facts.href_targets(node) {
            best = best.max(self.eval(target, role)?);
        }
        Ok(best)
    }

    fn filter(&mut self, node: Node<'a, 'input>) -> Result<Cost, Refusal> {
        if !node.children().any(|c| c.is_element()) {
            return self.linked(node, Role::Filter);
        }
        let mut sum = Cost::ZERO;
        for primitive in node.children().filter(|c| c.tag_name().name() == "feImage") {
            // An feImage naming an element converts it into a
            // fresh group; one holding a data: payload decodes
            // it. With both href spellings present, usvg uses
            // one of them.
            let mut best = Cost::ZERO;
            for attr in primitive.attributes() {
                if !attr.name().eq_ignore_ascii_case("href") {
                    continue;
                }
                let cost = if let Some(id) = iri_id(attr.value()) {
                    let mut target_cost = Cost::ZERO;
                    for &target in self.facts.ids.get(id).into_iter().flatten() {
                        target_cost = target_cost.max(self.eval(target, Role::Element)?);
                    }
                    Cost::nodes(1).plus(target_cost)
                } else {
                    data_payload(attr.value())
                };
                best = best.max(cost);
            }
            sum = self.add(sum, best)?;
        }
        Ok(sum)
    }

    fn element(&mut self, node: Node<'a, 'input>, name: &str) -> Result<Cost, Refusal> {
        const SHAPES: [&str; 7] = [
            "rect", "circle", "ellipse", "line", "polyline", "polygon", "path",
        ];
        // usvg's convert_element draws only these; anything else
        // (defs, symbol, mask, filter, unknown elements) draws
        // nothing where it stands, and is charged where a
        // reference converts it.
        let container = matches!(name, "g" | "a" | "switch" | "svg");
        let shape = SHAPES.contains(&name);
        if !(container || shape || matches!(name, "image" | "text" | "use")) {
            return Ok(Cost::ZERO);
        }
        let facts = self.facts;
        let mut sum = Cost::nodes(1);
        let filters = self.refs(&facts.own(node, Prop::Filter), Role::Filter, true)?;
        sum = self.add(sum, filters)?;
        let mask = self.refs(&facts.own(node, Prop::Mask), Role::Mask, false)?;
        sum = self.add(sum, mask)?;
        let clip = self.refs(&facts.own(node, Prop::ClipPath), Role::Clip, false)?;
        sum = self.add(sum, clip)?;
        if container {
            // A nested <svg> adds a clip path for its viewport.
            let extra = u64::from(name == "svg" && node.parent_element().is_some());
            return self.children(node, sum.plus(Cost::nodes(extra)));
        }
        if shape {
            let paint = self.paint(node)?;
            sum = self.add(sum, paint)?;
            let markers = self.markers(node)?;
            return self.add(sum, markers);
        }
        match name {
            "image" => {
                let mut payload = Cost::ZERO;
                for attr in node.attributes() {
                    if attr.name().eq_ignore_ascii_case("href") {
                        payload = payload.max(data_payload(attr.value()));
                    }
                }
                // The payload's node is the image itself.
                self.add(
                    sum,
                    Cost {
                        nodes: 0,
                        ..payload
                    },
                )
            }
            "text" => {
                let text = self.text(node)?;
                self.add(sum, text)
            }
            _ => {
                // <use>: a group (and a clip path for a symbol
                // or svg target) around a copy of the target.
                let mut target = Cost::ZERO;
                for t in facts.href_targets(node) {
                    target = target.max(self.eval(t, Role::UseTarget)?);
                }
                self.add(sum, target.plus(Cost::nodes(1)))
            }
        }
    }

    /// The patterns a shape's fill and stroke copy.
    fn paint(&mut self, node: Node<'a, 'input>) -> Result<Cost, Refusal> {
        let facts = self.facts;
        let fill = self.refs(&facts.inherited(node, Prop::Fill), Role::Pattern, false)?;
        let stroke = self.refs(&facts.inherited(node, Prop::Stroke), Role::Pattern, false)?;
        self.add(fill, stroke)
    }

    /// The marker instances on a shape: one per vertex.
    fn markers(&mut self, node: Node<'a, 'input>) -> Result<Cost, Refusal> {
        let facts = self.facts;
        let start = facts.inherited(node, Prop::MarkerStart);
        let mid = facts.inherited(node, Prop::MarkerMid);
        let end = facts.inherited(node, Prop::MarkerEnd);
        if start.is_empty() && mid.is_empty() && end.is_empty() {
            return Ok(Cost::ZERO);
        }
        let start = self.refs(&start, Role::Marker, false)?;
        let end = self.refs(&end, Role::Marker, false)?;
        let mid = self.refs(&mid, Role::Marker, false)?;
        let sum = self.add(start, end)?;
        self.add(sum, mid.times(segments(node)))
    }

    /// A text element: a node per character and per element of
    /// its content (glyph clusters, spans, and the flattened
    /// paths), plus the patterns every span copies.
    fn text(&mut self, node: Node<'a, 'input>) -> Result<Cost, Refusal> {
        let facts = self.facts;
        let mut units = 1u64;
        let mut paint = Cost::ZERO;
        for d in node.descendants() {
            if d.is_text() {
                units = units.saturating_add(char_count(d));
            } else if d.is_element() {
                units = units.saturating_add(1);
                paint = paint.max(self.paint(d)?);
                if d.tag_name().name() == "tref" {
                    for target in facts.href_targets(d) {
                        for t in target.descendants().filter(Node::is_text) {
                            units = units.saturating_add(char_count(t));
                        }
                    }
                }
            }
        }
        // Each span's fill and stroke are processed for the
        // span, its layout, and its flattened path.
        self.add(Cost::nodes(units), paint.times(units.saturating_mul(4)))
    }
}

fn char_count(node: Node) -> u64 {
    node.text().map_or(0, |t| t.chars().count() as u64)
}

/// An upper bound on the segments usvg makes of a shape: each
/// path segment carries at least one number of its own (a run
/// of digits), except a closepath, and an arc carries at least
/// five and becomes up to [`ARC_SEGMENTS`] cubics.
fn segments(node: Node) -> u64 {
    let attr_max = |name: &str, f: fn(&str) -> u64| {
        node.attributes()
            .filter(|a| a.name().eq_ignore_ascii_case(name))
            .map(|a| f(a.value()))
            .max()
            .unwrap_or(0)
    };
    match node.tag_name().name() {
        "path" => attr_max("d", path_segments),
        "polyline" | "polygon" => attr_max("points", path_segments),
        "line" => 3,
        _ => ARC_SEGMENTS + 10,
    }
}

fn path_segments(d: &str) -> u64 {
    let mut in_arc = false;
    let mut in_digits = false;
    let (mut runs, mut arc_runs, mut closes) = (0u64, 0u64, 0u64);
    for c in d.bytes() {
        if c.is_ascii_digit() {
            if !in_digits {
                if in_arc {
                    arc_runs += 1;
                } else {
                    runs += 1;
                }
            }
            in_digits = true;
            continue;
        }
        in_digits = false;
        match c {
            b'A' | b'a' => in_arc = true,
            b'Z' | b'z' => {
                in_arc = false;
                closes += 1;
            }
            b'M' | b'm' | b'L' | b'l' | b'H' | b'h' | b'V' | b'v' | b'C' | b'c' | b'S' | b's'
            | b'Q' | b'q' | b'T' | b't' => in_arc = false,
            _ => {}
        }
    }
    2 + runs + closes + arc_runs / 5 * ARC_SEGMENTS
}

fn is_inherit(value: &str) -> bool {
    value.trim() == "inherit"
}

fn starts_with_ci(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

/// A `data:` payload's cost: one image node and its encoded
/// length, which bounds the bytes usvg decodes from it.
fn data_payload(value: &str) -> Cost {
    if starts_with_ci(value.trim_start(), "data:") {
        Cost {
            nodes: 1,
            bytes: value.len() as u64,
            height: 0,
        }
    } else {
        Cost::ZERO
    }
}

/// The id an `href` names, parsed as usvg's IRI parser would
/// (leading spaces, `#`, up to a space), or a superset of it.
fn iri_id(value: &str) -> Option<&str> {
    let rest = value.trim_start_matches([' ', '\t', '\n', '\r']);
    let link = rest.strip_prefix('#')?;
    let id = link.split(' ').next().unwrap_or("");
    (!id.is_empty()).then_some(id)
}

/// The ids every `url(#id)` in `value` names. This accepts what
/// both of usvg's parsers accept (the FuncIRI one, with spaces
/// and quotes inside the parentheses, and the filter list one,
/// with spaces before them), in any letter case: a superset.
fn url_ids(value: &str) -> Vec<&str> {
    const SPACE: [char; 4] = [' ', '\t', '\n', '\r'];
    let mut ids = Vec::new();
    let mut rest = value;
    while let Some(i) = super::find_ascii_ci(rest, "url") {
        rest = &rest[i + 3..];
        let Some(inner) = rest.trim_start_matches(SPACE).strip_prefix('(') else {
            continue;
        };
        let inner = inner.trim_start_matches(SPACE);
        let (quote, inner) = match inner.chars().next() {
            Some(q @ ('\'' | '"')) => (Some(q), inner[1..].trim_start_matches(SPACE)),
            _ => (None, inner),
        };
        let Some(link) = inner.strip_prefix('#') else {
            continue;
        };
        let id = match quote {
            Some(q) => link.split(q).next().unwrap_or("").trim_end(),
            None => link.split([' ', ')']).next().unwrap_or(""),
        };
        if !id.is_empty() {
            ids.push(id);
        }
    }
    ids
}

/// usvg's selector-matching view of an element, reproduced so
/// the walk matches the same rules usvg's cascade does.
struct XmlNode<'a, 'input>(Node<'a, 'input>);

impl simplecss::Element for XmlNode<'_, '_> {
    fn parent_element(&self) -> Option<Self> {
        self.0.parent_element().map(XmlNode)
    }

    fn prev_sibling_element(&self) -> Option<Self> {
        self.0.prev_sibling_element().map(XmlNode)
    }

    fn has_local_name(&self, local_name: &str) -> bool {
        self.0.tag_name().name() == local_name
    }

    fn attribute_matches(&self, local_name: &str, operator: simplecss::AttributeOperator) -> bool {
        self.0
            .attribute(local_name)
            .is_some_and(|value| operator.matches(value))
    }

    fn pseudo_class_matches(&self, class: simplecss::PseudoClass) -> bool {
        match class {
            simplecss::PseudoClass::FirstChild => self.prev_sibling_element().is_none(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_ids_accept_what_usvg_accepts() {
        assert_eq!(url_ids("url(#a)"), ["a"]);
        assert_eq!(url_ids("  url(  #a  )  "), ["a"]);
        assert_eq!(url_ids("url('#a b')"), ["a b"]);
        assert_eq!(url_ids("url( \"#a\" )"), ["a"]);
        assert_eq!(url_ids("url (#a) blur(2) URL(#b)"), ["a", "b"]);
        assert_eq!(url_ids("url(#a) red"), ["a"]);
        assert!(url_ids("none").is_empty());
        assert!(url_ids("url()").is_empty());
        assert!(url_ids("url(a)").is_empty());
    }

    #[test]
    fn iri_ids_accept_what_usvg_accepts() {
        assert_eq!(iri_id("#a"), Some("a"));
        assert_eq!(iri_id("  #a"), Some("a"));
        assert_eq!(iri_id("#a b"), Some("a"));
        assert_eq!(iri_id("data:image/png;base64,AA=="), None);
        assert_eq!(iri_id("#"), None);
    }

    #[test]
    fn path_segments_cover_every_command() {
        // A run of digits per number, one per closepath, and
        // the arc allowance per five arc numbers.
        assert_eq!(path_segments("M0 0 L10 10"), 2 + 4);
        assert_eq!(path_segments("M0,0h5v5z"), 2 + 4 + 1);
        assert_eq!(path_segments("M0 0 A5 5 0 0 1 10 0"), 2 + 2 + ARC_SEGMENTS);
        // Packed arc flags share a run with the coordinate.
        assert_eq!(path_segments("M0 0a5 5 0 015 5"), 2 + 2 + ARC_SEGMENTS);
        assert_eq!(path_segments(""), 2);
    }

    fn bound_of(svg: &str) -> Result<Cost, Refusal> {
        let doc = roxmltree::Document::parse(svg).unwrap();
        bound(
            &doc,
            Limits {
                nodes: 100_000,
                bytes: 1 << 28,
                depth: 128,
            },
        )
    }

    fn svg(body: &str) -> String {
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"10\" height=\"10\">{body}</svg>"
        )
    }

    #[test]
    fn plain_documents_cost_their_drawn_elements() {
        let cost = bound_of(&svg(
            "<defs><rect id=\"r\" width=\"1\" height=\"1\"/></defs><g><rect width=\"1\" height=\"1\"/><circle r=\"1\"/></g>",
        ))
        .unwrap();
        // svg, g, rect, circle: the rect in defs draws nothing.
        assert_eq!(cost.nodes, 4);
        // svg > g > rect.
        assert_eq!(cost.height, 3);
    }

    #[test]
    fn feimage_charges_its_target_per_filter_instance() {
        let cost = bound_of(&svg(
            "<defs><g id=\"t\"><rect width=\"1\" height=\"1\"/><rect width=\"1\" height=\"1\"/></g>\
             <filter id=\"f\"><feImage href=\"#t\"/><feImage href=\"#t\"/></filter></defs>\
             <rect width=\"1\" height=\"1\" filter=\"url(#f)\"/><rect width=\"1\" height=\"1\" filter=\"url(#f)\"/>",
        ))
        .unwrap();
        // svg + 2 * (rect + 2 * (feImage group + g + 2 rects)).
        assert_eq!(cost.nodes, 1 + 2 * (1 + 2 * (1 + 3)));
    }

    #[test]
    fn css_and_style_declarations_are_followed() {
        let body = "<defs><g id=\"t\"><rect width=\"1\" height=\"1\"/></g>\
                    <filter id=\"f\"><feImage href=\"#t\"/></filter></defs>";
        let css = svg(&format!(
            "<style>.x {{ filter: url(#f) }}</style>{body}<rect class=\"x\" width=\"1\" height=\"1\"/>"
        ));
        let inline = svg(&format!(
            "{body}<rect style=\"filter:url(#f)\" width=\"1\" height=\"1\"/>"
        ));
        let none = svg(&format!("{body}<rect width=\"1\" height=\"1\"/>"));
        let base = bound_of(&none).unwrap().nodes;
        assert_eq!(bound_of(&css).unwrap().nodes, base + 3);
        assert_eq!(bound_of(&inline).unwrap().nodes, base + 3);
    }

    #[test]
    fn markers_are_charged_per_vertex_through_inheritance() {
        let cost = bound_of(&svg(
            "<defs><marker id=\"m\"><rect width=\"1\" height=\"1\"/></marker></defs>\
             <g marker-mid=\"url(#m)\"><polyline points=\"0,0 1,1 2,2\"/></g>",
        ))
        .unwrap();
        // svg, g, polyline, then 6 runs + 2 vertices of markers
        // at 3 nodes each (group, clip path, rect).
        assert_eq!(cost.nodes, 3 + 8 * 3);
    }

    #[test]
    fn a_marker_inside_its_own_content_costs_nothing_more() {
        // usvg skips a marker nested in itself; the walk does
        // too instead of calling it a cycle.
        let cost = bound_of(&svg(
            "<g marker-end=\"url(#m)\"><marker id=\"m\"><path d=\"M0 0 L1 1\"/></marker>\
             <path d=\"M0 0 L1 1\"/></g>",
        ))
        .unwrap();
        assert!(cost.nodes < 20, "{cost:?}");
    }

    #[test]
    fn cycles_and_deep_chains_are_refused() {
        let cycle = svg(
            "<defs><mask id=\"a\"><rect width=\"1\" height=\"1\" mask=\"url(#b)\"/></mask>\
             <mask id=\"b\"><rect width=\"1\" height=\"1\" mask=\"url(#c)\"/></mask>\
             <mask id=\"c\"><rect width=\"1\" height=\"1\" mask=\"url(#a)\"/></mask></defs>\
             <rect width=\"1\" height=\"1\" mask=\"url(#a)\"/>",
        );
        assert_eq!(bound_of(&cycle), Err(Refusal::Cycle));
        let mut chain =
            String::from("<defs><mask id=\"m0\"><rect width=\"1\" height=\"1\"/></mask>");
        for i in 1..100 {
            chain.push_str(&format!(
                "<mask id=\"m{i}\"><rect width=\"1\" height=\"1\" mask=\"url(#m{})\"/></mask>",
                i - 1
            ));
        }
        chain.push_str("</defs><rect width=\"1\" height=\"1\" mask=\"url(#m99)\"/>");
        assert_eq!(bound_of(&svg(&chain)), Err(Refusal::Depth));
        // A memoized subtree reached again deeper down repeats
        // its recursion there: a 40-level group reached through
        // a 60-hop chain is as deep as the two together.
        let mut deep = String::from("<defs>");
        deep.push_str(&"<g>".repeat(40));
        deep.push_str("<rect width=\"1\" height=\"1\"/>");
        deep.push_str(&"</g>".repeat(40));
        deep = deep.replacen("<g>", "<g id=\"m0\">", 1);
        for i in 1..60 {
            deep.push_str(&format!(
                "<mask id=\"k{i}\"><rect width=\"1\" height=\"1\" mask=\"url(#k{})\"/></mask>",
                i - 1
            ));
        }
        deep = deep.replace("url(#k0)", "url(#k0m)");
        deep.push_str("<mask id=\"k0m\"><use href=\"#m0\"/></mask></defs>");
        // First reached shallow (memoized), then at the bottom
        // of the chain.
        deep.push_str("<use href=\"#m0\"/><rect width=\"1\" height=\"1\" mask=\"url(#k59)\"/>");
        assert_eq!(bound_of(&svg(&deep)), Err(Refusal::Depth));
    }

    #[test]
    fn href_chain_cycles_are_refused() {
        // `#a` → `#b` → `#c` → `#b`: usvg's HrefIter guard
        // (self-link or return to origin) misses this, so it
        // loops with no end. Refused for each kind that walks
        // an href chain, including gradients and elements that
        // carry their own children (usvg still walks the chain
        // to resolve absent attributes).
        for body in [
            // Gradient: the issue's named case.
            "<defs><linearGradient id=\"a\" href=\"#b\"/><linearGradient id=\"b\" href=\"#c\"/>\
             <linearGradient id=\"c\" href=\"#b\"/></defs>\
             <rect width=\"1\" height=\"1\" fill=\"url(#a)\"/>",
            // Radial gradient, xlink spelling.
            "<defs><radialGradient id=\"a\" xlink:href=\"#b\"/><radialGradient id=\"b\" xlink:href=\"#c\"/>\
             <radialGradient id=\"c\" xlink:href=\"#b\"/></defs>\
             <rect width=\"1\" height=\"1\" fill=\"url(#a)\"/>",
            // Pattern whose entry carries children: the walk
            // still resolves tiling attributes down the chain.
            "<defs><pattern id=\"a\" href=\"#b\" width=\"1\" height=\"1\"><rect width=\"1\" height=\"1\"/></pattern>\
             <pattern id=\"b\" href=\"#c\"/><pattern id=\"c\" href=\"#b\"/></defs>\
             <rect width=\"1\" height=\"1\" fill=\"url(#a)\"/>",
            // Filter whose entry carries a primitive.
            "<defs><filter id=\"a\" href=\"#b\"><feFlood/></filter>\
             <filter id=\"b\" href=\"#c\"/><filter id=\"c\" href=\"#b\"/></defs>\
             <rect width=\"1\" height=\"1\" filter=\"url(#a)\"/>",
        ] {
            assert_eq!(bound_of(&svg(body)), Err(Refusal::Cycle), "{body}");
        }
    }

    #[test]
    fn legitimate_href_chains_are_not_refused() {
        // Chains usvg terminates must render: a plain
        // inheritance chain, a two-link chain through the
        // origin, and a self-link.
        for body in [
            "<defs><linearGradient id=\"base\"><stop offset=\"0\" stop-color=\"#f00\"/></linearGradient>\
             <linearGradient id=\"mid\" href=\"#base\"/><linearGradient id=\"a\" href=\"#mid\"/></defs>\
             <rect width=\"1\" height=\"1\" fill=\"url(#a)\"/>",
            "<defs><linearGradient id=\"a\" href=\"#b\"/><linearGradient id=\"b\" href=\"#a\"/></defs>\
             <rect width=\"1\" height=\"1\" fill=\"url(#a)\"/>",
            "<defs><pattern id=\"p\" href=\"#p\" width=\"1\" height=\"1\"><rect width=\"1\" height=\"1\"/></pattern></defs>\
             <rect width=\"1\" height=\"1\" fill=\"url(#p)\"/>",
        ] {
            assert!(bound_of(&svg(body)).is_ok(), "{body}");
        }
    }

    #[test]
    fn data_payloads_are_charged_per_instance() {
        let payload = format!("data:image/png;base64,{}", "A".repeat(1000));
        let cost = bound_of(&svg(&format!(
            "<defs><image id=\"i\" href=\"{payload}\" width=\"1\" height=\"1\"/></defs>\
             <use href=\"#i\"/><use href=\"#i\"/>"
        )))
        .unwrap();
        assert_eq!(cost.bytes, 2 * payload.len() as u64);
    }

    /// An id carried by many elements, each referenced from
    /// many sites, makes the walk re-reach every one of them
    /// from every site. Each visit (a memoized one included)
    /// spends fuel, so the document is refused instead of
    /// walked a quadratic number of times. Without the fuel
    /// charge on memoized visits, the walk runs
    /// `gradients * fills` times and is not refused at all.
    #[test]
    fn a_shared_id_referenced_many_times_is_refused_not_walked() {
        let mut body = String::from("<defs>");
        for _ in 0..500 {
            // All share one id, so each `url(#g)` resolves to
            // every one of them.
            body.push_str("<linearGradient id=\"g\"/>");
        }
        body.push_str("</defs>");
        for _ in 0..1000 {
            body.push_str("<rect width=\"1\" height=\"1\" fill=\"url(#g)\"/>");
        }
        assert_eq!(bound_of(&svg(&body)), Err(Refusal::Nodes));
        // A handful of the same references is walked and
        // accepted, so it is the fan-out, not the shape, that
        // is refused.
        let mut small = String::from("<defs><linearGradient id=\"g\"/></defs>");
        for _ in 0..4 {
            small.push_str("<rect width=\"1\" height=\"1\" fill=\"url(#g)\"/>");
        }
        assert!(bound_of(&svg(&small)).is_ok());
    }

    /// An `inherit` pulls every value a property takes anywhere
    /// in the document into the list examined for references.
    /// With one such list per element, the lists examined grow
    /// to the square of the document. Each value examined
    /// spends fuel, so the document is refused; without that
    /// charge the walk examines `declarations * inheritors`
    /// values and is not refused.
    #[test]
    fn an_inherit_value_list_fanned_out_is_refused() {
        let mut body = String::new();
        // Each contributes a `filter` value to what an
        // `inherit` resolves to.
        for _ in 0..750 {
            body.push_str("<rect width=\"1\" height=\"1\" filter=\"none\"/>");
        }
        // Each examines all of them.
        for _ in 0..750 {
            body.push_str("<rect width=\"1\" height=\"1\" filter=\"inherit\"/>");
        }
        assert_eq!(bound_of(&svg(&body)), Err(Refusal::Nodes));
    }

    /// Matching the kept style rules against the elements is
    /// capped: a `<style>` with many referencing rules and many
    /// elements for them to match is refused before the
    /// quadratic matching runs. Without the cap it runs
    /// `rules * elements` comparisons and is accepted.
    #[test]
    fn style_rules_matching_is_capped() {
        let mut style = String::from("<style>");
        for i in 0..2100 {
            // Each rule references a filter, so it is kept, and
            // each is a distinct selector.
            style.push_str(&format!(".c{i} {{ filter: url(#f) }}"));
        }
        style.push_str("</style>");
        let mut body = format!("<defs><filter id=\"f\"><feFlood/></filter></defs>{style}");
        for _ in 0..2100 {
            body.push_str("<rect width=\"1\" height=\"1\"/>");
        }
        assert_eq!(bound_of(&svg(&body)), Err(Refusal::Work));
        // A small style sheet matched against few elements is
        // under the cap and is bounded as before.
        let ok = svg("<style>.c { filter: url(#f) }</style>\
             <defs><filter id=\"f\"><feFlood/></filter></defs>\
             <rect class=\"c\" width=\"1\" height=\"1\"/>");
        assert!(bound_of(&ok).is_ok());
    }

    /// The fuel and the style cap do not refuse a document
    /// whose reuse is ordinary: an element reused a modest
    /// number of times, with an `inherit`, is bounded, and to
    /// the same cost as the reuse written out by hand.
    #[test]
    fn ordinary_reuse_is_bounded_unchanged() {
        let reused = bound_of(&svg(
            "<defs><g id=\"t\"><rect width=\"1\" height=\"1\"/><rect width=\"1\" height=\"1\"/></g></defs>\
             <use href=\"#t\"/><use href=\"#t\"/><use href=\"#t\"/>",
        ))
        .unwrap();
        // svg + 3 * (use + its group node + (g + 2 rects)).
        assert_eq!(reused.nodes, 1 + 3 * (1 + 1 + (1 + 2)));
        // An `inherit` on a non-inherited property still lands
        // under the caps for a normal document.
        assert!(
            bound_of(&svg("<defs><filter id=\"f\"><feFlood/></filter></defs>\
             <rect width=\"1\" height=\"1\" filter=\"url(#f)\"/>\
             <rect width=\"1\" height=\"1\" filter=\"inherit\"/>",))
            .is_ok()
        );
    }

    /// A ring of same-kind `href` links, a tail leading into a
    /// ring, and a chain that ends keep the results they had
    /// before the check was made linear: usvg terminates on a
    /// pure ring (every member returns to where it began) and
    /// on a chain that ends, and loops when a chain leads into
    /// a ring it is not part of, so the first two are accepted
    /// and the third is refused.
    #[test]
    fn rings_and_chains_keep_their_results() {
        // A pure ring: a0 -> a1 -> a2 -> a0.
        let pure_ring = "<defs><filter id=\"a0\" href=\"#a1\"/><filter id=\"a1\" href=\"#a2\"/>\
             <filter id=\"a2\" href=\"#a0\"/></defs><rect width=\"1\" height=\"1\"/>";
        assert!(bound_of(&svg(pure_ring)).is_ok());
        // A tail into a ring: t -> a0, and a0 -> a1 -> a2 -> a0.
        let tail = "<defs><filter id=\"t\" href=\"#a0\"/>\
             <filter id=\"a0\" href=\"#a1\"/><filter id=\"a1\" href=\"#a2\"/>\
             <filter id=\"a2\" href=\"#a0\"/></defs><rect width=\"1\" height=\"1\"/>";
        assert_eq!(bound_of(&svg(tail)), Err(Refusal::Cycle));
        // A chain that ends at a missing id.
        let chain = "<defs><filter id=\"a0\" href=\"#a1\"/><filter id=\"a1\" href=\"#a2\"/>\
             <filter id=\"a2\" href=\"#end\"/></defs><rect width=\"1\" height=\"1\"/>";
        assert!(bound_of(&svg(chain)).is_ok());
    }

    /// A pure ring of 10,000 same-kind `href` links is checked
    /// in linear time: before the check was made linear it
    /// walked the ring from every member, so a ring this size
    /// took seconds. The bound here is far above the linear
    /// cost and far below the quadratic one.
    #[test]
    fn a_large_href_ring_is_checked_quickly() {
        // svg + defs + 9997 filters + rect = the 10,000-node
        // document cap.
        const N: usize = 9997;
        let mut body = String::from("<defs>");
        for i in 0..N {
            body.push_str(&format!("<filter id=\"a{i}\" href=\"#a{}\"/>", (i + 1) % N));
        }
        body.push_str("</defs><rect width=\"1\" height=\"1\"/>");
        let text = svg(&body);
        let doc = roxmltree::Document::parse(&text).unwrap();
        let start = std::time::Instant::now();
        let result = bound(
            &doc,
            Limits {
                nodes: 100_000,
                bytes: 1 << 28,
                depth: 128,
            },
        );
        let elapsed = start.elapsed();
        // A pure ring is accepted, as it was before.
        assert!(result.is_ok(), "{result:?}");
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "checking a {N}-element ring took {elapsed:?}; the check is not linear"
        );
    }
}
