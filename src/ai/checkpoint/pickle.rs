//! A restricted pickle machine that reads a head's state
//! dict.

/// A tensor reference out of the pickle machine: the
/// storage it reads from, the element offset into it,
/// and its shape. Only the flat element sequence is
/// consumed (in storage order), so the stride never
/// matters here.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PickleTensor {
    pub(super) storage: String,
    pub(super) offset: usize,
    pub(super) size: Vec<usize>,
}

/// A value on the restricted pickle machine's stack.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// A global that has not been called: (module,
    /// name). Only a small allowlist is callable, and
    /// only through REDUCE.
    Global(String, String),
    Tuple(Vec<PickleValue>),
    /// A tensor storage, by the archive member key its
    /// bytes live under (`archive/data/<key>`).
    Storage(String),
    Tensor(PickleTensor),
    /// An insertion-ordered string-keyed map: the
    /// state dict itself, and plain dicts (the BUILD
    /// state) share the shape.
    Map(Vec<(String, PickleValue)>),
}

/// The deepest a value may nest, counted in tuples and
/// dicts. A head save nests a few levels. The bound keeps
/// the recursive drop, clone and compare of `PickleValue`
/// shallow (issue #609).
const MAX_DEPTH: usize = 32;

/// The most the machine may build: one per value plus
/// the bytes of its strings. A memo copy counts at the
/// size of the tree it copies. A head save builds a few
/// hundred (issue #609).
const MAX_BUILT: usize = 1 << 16;

// The opcodes the machine reads, by their names in
// CPython's pickle module. `PickleMachine::run` refuses
// every other byte.
const PROTO: u8 = 0x80;
const STOP: u8 = b'.';
const BINUNICODE: u8 = b'X';
const SHORT_BINUNICODE: u8 = 0x8c;
const SHORT_BINSTRING: u8 = b'U';
const BININT1: u8 = b'K';
const BININT2: u8 = b'M';
const BININT: u8 = b'J';
const LONG: u8 = b'L';
const LONG1: u8 = 0x8a;
const LONG4: u8 = 0x8b;
const BINFLOAT: u8 = b'G';
const NONE: u8 = b'N';
const NEWTRUE: u8 = 0x88;
const NEWFALSE: u8 = 0x89;
const MARK: u8 = b'(';
const EMPTY_TUPLE: u8 = b')';
const TUPLE: u8 = b't';
const TUPLE1: u8 = 0x85;
const TUPLE2: u8 = 0x86;
const TUPLE3: u8 = 0x87;
const EMPTY_DICT: u8 = b'}';
const SETITEM: u8 = b's';
const SETITEMS: u8 = b'u';
const BUILD: u8 = b'b';
const BINPUT: u8 = b'q';
const LONG_BINPUT: u8 = b'r';
const BINGET: u8 = b'h';
const LONG_BINGET: u8 = b'j';
const GLOBAL: u8 = b'c';
const REDUCE: u8 = b'R';
const BINPERSID: u8 = b'Q';

/// A restricted pickle interpreter, sized for the
/// state dicts `torch.save` writes for a linear head:
/// an ordered dict of tensor rebuilds over float
/// storages, with memo references. The opcode set is
/// an allowlist; anything else (a code-carrying
/// global, a reduce of an unknown callable, an
/// unexpected structure) is an error, so a hostile or
/// merely different save fails at load instead of
/// being half-parsed. This is a data reader, not a
/// general unpickler: it cannot construct arbitrary
/// objects and never touches the filesystem. What it
/// builds is bounded too: every value counts against
/// `MAX_BUILT`, and no value nests past `MAX_DEPTH`.
///
/// An open MARK fences the stack, as in CPython's
/// unpickler (`Modules/_pickle.c`, where `load_mark` sets
/// the fence of the `Pdata` stack, `marker` lowers it to
/// the MARK before when an opcode closes one, and
/// `Pdata_pop`, `Pdata_poptuple`, `load_build`,
/// `do_setitems` and the memo puts refuse to reach below
/// it). No opcode here pops or reads a value from below
/// the latest open MARK (see `fenced`), so no MARK stands
/// past the top of the stack, and TUPLE and SETITEMS
/// always find theirs within it (issue #760).
struct PickleMachine<'a> {
    bytes: &'a [u8],
    at: usize,
    stack: Vec<PickleValue>,
    /// The stack length each open MARK recorded, the
    /// latest last.
    marks: Vec<usize>,
    memo: Vec<Option<PickleValue>>,
    /// The running count against `MAX_BUILT`.
    built: usize,
}

impl<'a> PickleMachine<'a> {
    /// Read opcodes until STOP. MARK and the opcodes that
    /// push a fixed value run here; every other opcode
    /// runs in the handler for its family below.
    fn run(mut self) -> Result<PickleValue, String> {
        loop {
            let op = self.byte().ok_or("the pickle ends without a STOP")?;
            match op {
                PROTO => self.proto()?,
                STOP => return self.stop(),
                BINUNICODE | SHORT_BINUNICODE | SHORT_BINSTRING => self.string(op)?,
                BININT1 | BININT2 | BININT | LONG | LONG1 | LONG4 => self.integer(op)?,
                BINFLOAT => self.float()?,
                NONE => self.push(PickleValue::None)?,
                NEWTRUE => self.push(PickleValue::Bool(true))?,
                NEWFALSE => self.push(PickleValue::Bool(false))?,
                MARK => self.marks.push(self.stack.len()),
                EMPTY_TUPLE => self.push(PickleValue::Tuple(Vec::new()))?,
                TUPLE | TUPLE1 | TUPLE2 | TUPLE3 => self.tuple(op)?,
                EMPTY_DICT => self.push(PickleValue::Map(Vec::new()))?,
                SETITEM => self.set_item()?,
                SETITEMS => self.set_items()?,
                BUILD => self.build()?,
                BINPUT | LONG_BINPUT | BINGET | LONG_BINGET => self.memo(op)?,
                GLOBAL => self.global()?,
                REDUCE => self.call()?,
                BINPERSID => self.persistent_id()?,
                _ => return Err(unsupported(op)),
            }
        }
    }

    /// PROTO: the version byte is advisory; the opcode
    /// allowlist in `run` is the real gate.
    fn proto(&mut self) -> Result<(), String> {
        self.byte().ok_or("the pickle ends inside PROTO")?;
        Ok(())
    }

    /// STOP: the one value left on the stack is the
    /// pickle's. CPython pops it like any other value, so
    /// it refuses one below an open MARK.
    fn stop(mut self) -> Result<PickleValue, String> {
        self.fenced("STOP", 1)?;
        if self.stack.len() != 1 {
            return Err(format!(
                "the pickle ends with {} values on the stack, not one",
                self.stack.len()
            ));
        }
        Ok(self.stack.pop().unwrap())
    }

    /// Strings: BINUNICODE and SHORT_BINUNICODE (UTF-8
    /// behind a four-byte and a one-byte length) and
    /// SHORT_BINSTRING (bytes behind a one-byte length,
    /// read lossily). They are the state dict's keys and
    /// the storage arguments.
    fn string(&mut self, op: u8) -> Result<(), String> {
        let len = match op {
            BINUNICODE => self.u32()? as usize,
            SHORT_BINUNICODE | SHORT_BINSTRING => {
                self.byte().ok_or("the pickle ends inside a short string")? as usize
            }
            _ => return Err(unsupported(op)),
        };
        let raw = self.take(len)?;
        let s = if op == SHORT_BINSTRING {
            String::from_utf8_lossy(raw).into_owned()
        } else {
            String::from_utf8(raw.to_vec())
                .map_err(|_| "a pickle string is not valid UTF-8".to_string())?
        };
        self.push(PickleValue::Str(s))
    }

    /// Integers: BININT1, BININT2 and BININT, LONG (a
    /// decimal line), and LONG1 and LONG4 (little-endian
    /// two's complement behind a one-byte and a four-byte
    /// length).
    fn integer(&mut self, op: u8) -> Result<(), String> {
        let v = match op {
            BININT1 => self.byte().ok_or("the pickle ends inside BININT1")? as i64,
            BININT2 => self.u16()? as i64,
            BININT => self.u32()? as i32 as i64,
            LONG => {
                let raw = self.line()?;
                let digits = raw.strip_suffix('L').unwrap_or(&raw);
                digits
                    .parse::<i64>()
                    .map_err(|_| format!("a pickle LONG is not a number: {raw:?}"))?
            }
            LONG1 => {
                let len = self.byte().ok_or("the pickle ends inside LONG1")? as usize;
                self.long(len)?
            }
            LONG4 => {
                let len = self.u32()? as usize;
                self.long(len)?
            }
            _ => return Err(unsupported(op)),
        };
        self.push(PickleValue::Int(v))
    }

    /// The body of a LONG1 or LONG4: `len` bytes of
    /// little-endian two's complement, at most the eight
    /// an `i64` holds.
    fn long(&mut self, len: usize) -> Result<i64, String> {
        if len > 8 {
            return Err("a pickle long is too large for a head file".to_string());
        }
        let raw = self.take(len)?;
        let mut v: i64 = 0;
        for (i, b) in raw.iter().enumerate() {
            v |= (*b as i64) << (8 * i);
        }
        if len > 0 && len < 8 && raw[len - 1] & 0x80 != 0 {
            v -= 1i64 << (8 * len);
        }
        Ok(v)
    }

    /// BINFLOAT: a big-endian f64.
    fn float(&mut self) -> Result<(), String> {
        let raw = self.take(8)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(raw);
        self.push(PickleValue::Float(f64::from_be_bytes(buf)))
    }

    /// TUPLE folds the values since the mark into a tuple,
    /// TUPLE1 to TUPLE3 the top one to three. TUPLE closes
    /// its MARK and takes only what stands above it; CPython
    /// refuses TUPLE1 to TUPLE3 a value below an open MARK.
    fn tuple(&mut self, op: u8) -> Result<(), String> {
        let from = match op {
            TUPLE => self.mark()?,
            TUPLE1..=TUPLE3 => {
                let n = usize::from(op - TUPLE1) + 1;
                self.fenced(["TUPLE1", "TUPLE2", "TUPLE3"][n - 1], n)?;
                self.stack
                    .len()
                    .checked_sub(n)
                    .ok_or("the pickle's stack underflows at a tuple")?
            }
            _ => return Err(unsupported(op)),
        };
        let items: Vec<PickleValue> = self.stack.drain(from..).collect();
        self.push_tuple(items)
    }

    /// SETITEM: value, then key, onto the map below them.
    /// CPython refuses it unless all three stand above the
    /// latest open MARK.
    fn set_item(&mut self) -> Result<(), String> {
        self.fenced("SETITEM", 3)?;
        let value = self.pop()?;
        let key = self.pop()?;
        let PickleValue::Str(key) = key else {
            return Err("a pickle dict key is not a string".to_string());
        };
        check_nesting(&value)?;
        let Some(PickleValue::Map(pairs)) = self.stack.last_mut() else {
            return Err("a pickle SETITEM has no dict under it".to_string());
        };
        set_map_entry(pairs, key, value);
        Ok(())
    }

    /// SETITEMS: key/value pairs since the mark, onto the
    /// map below it. CPython reads the map below a MARK on
    /// purpose: it sits under the MARK that SETITEMS
    /// closes. It must still stand above the MARK open
    /// before that one, or CPython refuses it.
    fn set_items(&mut self) -> Result<(), String> {
        let from = self.mark()?;
        // The pairs since the MARK, and the map under them.
        self.fenced("SETITEMS", self.stack.len() - from + 1)?;
        let items: Vec<PickleValue> = self.stack.drain(from..).collect();
        if !items.len().is_multiple_of(2) {
            return Err("a pickle SETITEMS holds an odd item count".to_string());
        }
        for value in items.iter().skip(1).step_by(2) {
            check_nesting(value)?;
        }
        let Some(PickleValue::Map(pairs)) = self.stack.last_mut() else {
            return Err("a pickle SETITEMS has no dict under it".to_string());
        };
        let mut iter = items.into_iter();
        while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
            let PickleValue::Str(key) = k else {
                return Err("a pickle dict key is not a string".to_string());
            };
            set_map_entry(pairs, key, v);
        }
        Ok(())
    }

    /// BUILD: the state (a metadata dict in torch saves)
    /// is validated as data and discarded; the built
    /// object stays. CPython refuses it unless both stand
    /// above the latest open MARK.
    fn build(&mut self) -> Result<(), String> {
        self.fenced("BUILD", 2)?;
        let state = self.pop()?;
        if !matches!(state, PickleValue::Map(_) | PickleValue::None) {
            return Err("a pickle BUILD state is not a dict".to_string());
        }
        if self.stack.is_empty() {
            return Err("a pickle BUILD has no object under it".to_string());
        }
        Ok(())
    }

    /// The memo: BINPUT and LONG_BINPUT store the stack
    /// top under a one-byte or a four-byte index, BINGET
    /// and LONG_BINGET push what an index holds.
    fn memo(&mut self, op: u8) -> Result<(), String> {
        match op {
            BINPUT => {
                let at = self.byte().ok_or("the pickle ends inside BINPUT")? as usize;
                self.put_memo("BINPUT", at)
            }
            LONG_BINPUT => {
                let at = self.u32()? as usize;
                self.put_memo("LONG_BINPUT", at)
            }
            BINGET => {
                let at = self.byte().ok_or("the pickle ends inside BINGET")? as usize;
                self.get_memo(at)
            }
            LONG_BINGET => {
                let at = self.u32()? as usize;
                self.get_memo(at)
            }
            _ => Err(unsupported(op)),
        }
    }

    /// GLOBAL: 'c' module '\n' name '\n'.
    fn global(&mut self) -> Result<(), String> {
        let module = self.line()?;
        let name = self.line()?;
        self.push(PickleValue::Global(module, name))
    }

    /// REDUCE: call an allowlisted global (see `reduce`)
    /// with an argument tuple. CPython pops both, so it
    /// refuses either below an open MARK.
    fn call(&mut self) -> Result<(), String> {
        self.fenced("REDUCE", 2)?;
        let args = self.pop()?;
        let callable = self.pop()?;
        let PickleValue::Global(module, name) = callable else {
            return Err("a pickle REDUCE target is not a global".to_string());
        };
        let PickleValue::Tuple(args) = args else {
            return Err("a pickle REDUCE argument is not a tuple".to_string());
        };
        let value = reduce(&module, &name, args)?;
        self.push(value)
    }

    /// BINPERSID: torch's persistent id for a storage, a
    /// tuple of (the word "storage", the storage class,
    /// the archive key, the location, the element count).
    /// Only the key is used. CPython pops the id, so it
    /// refuses one below an open MARK.
    fn persistent_id(&mut self) -> Result<(), String> {
        self.fenced("BINPERSID", 1)?;
        let value = self.pop()?;
        let PickleValue::Tuple(parts) = &value else {
            return Err("a pickle persistent id is not a tuple".to_string());
        };
        let [
            PickleValue::Str(kind),
            PickleValue::Global(_, class),
            PickleValue::Str(key),
            ..,
        ] = parts.as_slice()
        else {
            return Err("a pickle persistent id does not name a storage".to_string());
        };
        if kind.as_str() != "storage" || !class.contains("Storage") {
            return Err(format!(
                "a pickle persistent id names {kind} {class}, not a storage"
            ));
        }
        self.push(PickleValue::Storage(key.clone()))
    }

    fn byte(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.at)?;
        self.at += 1;
        Some(b)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let raw = self
            .bytes
            .get(self.at..self.at + len)
            .ok_or("the pickle is truncated")?;
        self.at += len;
        Ok(raw)
    }

    fn u16(&mut self) -> Result<u16, String> {
        let raw = self.take(2)?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let raw = self.take(4)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn line(&mut self) -> Result<String, String> {
        let start = self.at;
        let end = self
            .bytes
            .iter()
            .skip(start)
            .position(|b| *b == b'\n')
            .map(|p| start + p)
            .ok_or("the pickle ends inside a line")?;
        let raw = &self.bytes[start..end];
        self.at = end + 1;
        String::from_utf8(raw.to_vec()).map_err(|_| "a pickle line is not valid UTF-8".to_string())
    }

    fn pop(&mut self) -> Result<PickleValue, String> {
        self.stack
            .pop()
            .ok_or_else(|| "the pickle's stack underflows".to_string())
    }

    /// Close the latest open MARK, for TUPLE and SETITEMS:
    /// the stack length it recorded. No opcode since has
    /// popped below it, so it is within the stack.
    fn mark(&mut self) -> Result<usize, String> {
        self.marks
            .pop()
            .ok_or_else(|| "the pickle has a tuple or items opcode with no mark".to_string())
    }

    /// Refuse `name`, an opcode about to pop or read the
    /// top `n` values of the stack, when one of them stands
    /// below the latest open MARK, as CPython refuses it
    /// ("unexpected MARK found"). Each opcode checks before
    /// it takes or looks at a value, as in CPython. A MARK
    /// at the bottom of the stack fences nothing: there, a
    /// stack shorter than `n` is the opcode's own
    /// underflow, which it reports in its own words.
    fn fenced(&self, name: &str, n: usize) -> Result<(), String> {
        let fence = self.marks.last().copied().unwrap_or(0);
        if fence > 0 && self.stack.len() < fence + n {
            return Err(fence_error(name));
        }
        Ok(())
    }

    /// Push a value built from the file's bytes. It
    /// counts at its size.
    fn push(&mut self, value: PickleValue) -> Result<(), String> {
        charge(&mut self.built, measure(&value).0)?;
        self.stack.push(value);
        Ok(())
    }

    /// Push a tuple of values that were on the stack. They
    /// were counted when they were built, so the tuple
    /// counts as one.
    fn push_tuple(&mut self, items: Vec<PickleValue>) -> Result<(), String> {
        let depth = 1 + items.iter().map(|v| measure(v).1).max().unwrap_or(0);
        if depth > MAX_DEPTH {
            return Err(nesting_error());
        }
        charge(&mut self.built, 1)?;
        self.stack.push(PickleValue::Tuple(items));
        Ok(())
    }

    /// Store the stack top in the memo, for `name`. A memo
    /// copy counts at the size of the tree it copies, before
    /// the copy is made. CPython refuses to store a top that
    /// stands below an open MARK; it checks once it has read
    /// the index, as this does.
    fn put_memo(&mut self, name: &str, at: usize) -> Result<(), String> {
        self.fenced(name, 1)?;
        if at > 4096 {
            return Err("a pickle memo index is implausibly large".to_string());
        }
        let top = self
            .stack
            .last()
            .ok_or("a pickle memo stores an empty stack")?;
        charge(&mut self.built, measure(top).0)?;
        let value = top.clone();
        if self.memo.len() <= at {
            self.memo.resize(at + 1, None);
        }
        self.memo[at] = Some(value);
        Ok(())
    }

    fn get_memo(&mut self, at: usize) -> Result<(), String> {
        let stored = self
            .memo
            .get(at)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("a pickle memo reference to {at} was never stored"))?;
        charge(&mut self.built, measure(stored).0)?;
        let value = stored.clone();
        self.stack.push(value);
        Ok(())
    }
}

/// A value's size (one per value, plus the bytes of its
/// strings) and its depth in tuples and dicts. The
/// recursion is shallow: every value the machine holds
/// was checked against `MAX_DEPTH` when it was built.
fn measure(value: &PickleValue) -> (usize, usize) {
    match value {
        PickleValue::None | PickleValue::Bool(_) | PickleValue::Int(_) | PickleValue::Float(_) => {
            (1, 0)
        }
        PickleValue::Str(s) | PickleValue::Storage(s) => (1 + s.len(), 0),
        PickleValue::Global(module, name) => (1 + module.len() + name.len(), 0),
        PickleValue::Tensor(t) => (1 + t.storage.len() + t.size.len(), 0),
        PickleValue::Tuple(items) => items.iter().fold((1, 1), |(size, depth), item| {
            let (s, d) = measure(item);
            (size + s, depth.max(d + 1))
        }),
        PickleValue::Map(pairs) => pairs.iter().fold((1, 1), |(size, depth), (key, item)| {
            let (s, d) = measure(item);
            (size + key.len() + s, depth.max(d + 1))
        }),
    }
}

/// Add `size` to the running count, or refuse once it
/// passes `MAX_BUILT`.
fn charge(built: &mut usize, size: usize) -> Result<(), String> {
    *built = built.saturating_add(size);
    if *built > MAX_BUILT {
        return Err(format!(
            "the pickle builds too much for a head state dict: over {MAX_BUILT} values and string bytes"
        ));
    }
    Ok(())
}

/// Refuse a value that would nest too deep once a dict
/// holds it.
fn check_nesting(value: &PickleValue) -> Result<(), String> {
    if measure(value).1 + 1 > MAX_DEPTH {
        return Err(nesting_error());
    }
    Ok(())
}

fn nesting_error() -> String {
    format!(
        "the pickle nests tuples or dicts more than {MAX_DEPTH} levels deep: a head state dict nests a few"
    )
}

/// The refusal of an opcode that would pop or read a
/// value from below an open MARK (see
/// `PickleMachine::fenced`).
fn fence_error(name: &str) -> String {
    format!("a pickle {name} uses a value from below an open MARK")
}

/// The refusal of a byte outside the opcode allowlist.
fn unsupported(op: u8) -> String {
    format!("unsupported pickle opcode 0x{op:02x}: not a head state dict shape this loader reads")
}

/// The REDUCE allowlist: build an ordered dict, or
/// rebuild a tensor over a storage. Anything else is
/// refused by name, so the failure says what the save
/// tried to do.
fn reduce(module: &str, name: &str, args: Vec<PickleValue>) -> Result<PickleValue, String> {
    match (module, name) {
        ("collections", "OrderedDict") => {
            if args.is_empty() {
                Ok(PickleValue::Map(Vec::new()))
            } else {
                Err("an OrderedDict reduce with arguments is not a state dict".to_string())
            }
        }
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
            let mut args = args.into_iter();
            let storage = match args.next() {
                Some(PickleValue::Storage(key)) => key,
                _ => return Err("a tensor rebuild does not start from a storage".to_string()),
            };
            let offset = match args.next() {
                Some(PickleValue::Int(v)) if v >= 0 => v as usize,
                _ => return Err("a tensor rebuild has no valid offset".to_string()),
            };
            let size = match args.next() {
                Some(PickleValue::Tuple(items)) => {
                    let mut size = Vec::with_capacity(items.len());
                    for item in items {
                        match item {
                            PickleValue::Int(v) if v >= 0 => size.push(v as usize),
                            _ => return Err("a tensor rebuild has an invalid size".to_string()),
                        }
                    }
                    size
                }
                _ => return Err("a tensor rebuild has no size".to_string()),
            };
            // The stride, the requires-grad flag, and
            // any later arguments are validated only
            // as data: the weights are read flat, in
            // storage order.
            Ok(PickleValue::Tensor(PickleTensor {
                storage,
                offset,
                size,
            }))
        }
        _ => Err(format!(
            "unsupported pickle callable {module}.{name}: not a head state dict shape this loader reads"
        )),
    }
}

/// Insert or replace one entry of a pickle map.
fn set_map_entry(pairs: &mut Vec<(String, PickleValue)>, key: String, value: PickleValue) {
    if let Some(slot) = pairs.iter_mut().find(|(k, _)| *k == key) {
        slot.1 = value;
    } else {
        pairs.push((key, value));
    }
}

/// Run the restricted pickle machine over a
/// `data.pkl` payload.
pub(super) fn unpickle(bytes: &[u8]) -> Result<PickleValue, String> {
    if bytes.len() > 1024 * 1024 {
        return Err("the data.pkl is implausibly large for a head state dict".to_string());
    }
    PickleMachine {
        bytes,
        at: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: Vec::new(),
        built: 0,
    }
    .run()
}

#[cfg(test)]
mod tests;

// The opcode handlers, pinned against a frozen copy of the
// single loop they replaced (issue #741), apart from the
// two changes issue #760 made after it.
#[cfg(test)]
mod handler_proofs;
