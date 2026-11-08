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
/// objects and never touches the filesystem.
struct PickleMachine<'a> {
    bytes: &'a [u8],
    at: usize,
    stack: Vec<PickleValue>,
    marks: Vec<usize>,
    memo: Vec<Option<PickleValue>>,
}

impl<'a> PickleMachine<'a> {
    fn run(mut self) -> Result<PickleValue, String> {
        loop {
            let op = self.byte().ok_or("the pickle ends without a STOP")?;
            match op {
                // PROTO: the version byte is advisory;
                // the opcode allowlist below is the
                // real gate.
                0x80 => {
                    self.byte().ok_or("the pickle ends inside PROTO")?;
                }
                // GLOBAL: 'c' module '\n' name '\n'
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(PickleValue::Global(module, name));
                }
                // BINUNICODE / SHORT_BINUNICODE /
                // SHORT_BINSTRING: length-prefixed
                // strings (the state dict's keys and the
                // storage arguments).
                b'X' => {
                    let len = self.u32()? as usize;
                    let raw = self.take(len)?;
                    self.stack
                        .push(PickleValue::Str(String::from_utf8(raw.to_vec()).map_err(
                            |_| "a pickle string is not valid UTF-8".to_string(),
                        )?));
                }
                0x8c => {
                    let len = self.byte().ok_or("the pickle ends inside a short string")? as usize;
                    let raw = self.take(len)?;
                    self.stack
                        .push(PickleValue::Str(String::from_utf8(raw.to_vec()).map_err(
                            |_| "a pickle string is not valid UTF-8".to_string(),
                        )?));
                }
                b'U' => {
                    let len = self.byte().ok_or("the pickle ends inside a short string")? as usize;
                    let raw = self.take(len)?;
                    self.stack
                        .push(PickleValue::Str(String::from_utf8_lossy(raw).into_owned()));
                }
                // Integers: BININT1, BININT2, BININT,
                // LONG (a decimal line), LONG1/LONG4
                // (little-endian two's complement).
                b'K' => {
                    let v = self.byte().ok_or("the pickle ends inside BININT1")?;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                b'M' => {
                    let v = self.u16()?;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                b'J' => {
                    let v = self.u32()? as i32;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                b'L' => {
                    let raw = self.line()?;
                    let digits = raw.strip_suffix('L').unwrap_or(&raw);
                    let v: i64 = digits
                        .parse()
                        .map_err(|_| format!("a pickle LONG is not a number: {raw:?}"))?;
                    self.stack.push(PickleValue::Int(v));
                }
                0x8a | 0x8b => {
                    let len = if op == 0x8a {
                        self.byte().ok_or("the pickle ends inside LONG1")? as usize
                    } else {
                        self.u32()? as usize
                    };
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
                    self.stack.push(PickleValue::Int(v));
                }
                // BINFLOAT: big-endian f64.
                b'G' => {
                    let raw = self.take(8)?;
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(raw);
                    self.stack.push(PickleValue::Float(f64::from_be_bytes(buf)));
                }
                b'N' => self.stack.push(PickleValue::None),
                0x88 => self.stack.push(PickleValue::Bool(false)),
                0x89 => self.stack.push(PickleValue::Bool(true)),
                b')' => self.stack.push(PickleValue::Tuple(Vec::new())),
                b'(' => self.marks.push(self.stack.len()),
                b't' => {
                    let from = self.mark()?;
                    let items: Vec<PickleValue> = self.stack.drain(from..).collect();
                    self.stack.push(PickleValue::Tuple(items));
                }
                0x85..=0x87 => {
                    let n = (op - 0x84) as usize;
                    if self.stack.len() < n {
                        return Err("the pickle's stack underflows at a tuple".to_string());
                    }
                    let items: Vec<PickleValue> =
                        self.stack.drain(self.stack.len() - n..).collect();
                    self.stack.push(PickleValue::Tuple(items));
                }
                // BINPERSID: torch's persistent id for a
                // storage, a tuple of (the word
                // "storage", the storage class, the
                // archive key, the location, the element
                // count). Only the key is used.
                b'Q' => {
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
                    self.stack.push(PickleValue::Storage(key.clone()));
                }
                // REDUCE: call an allowlisted global.
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let PickleValue::Global(module, name) = callable else {
                        return Err("a pickle REDUCE target is not a global".to_string());
                    };
                    let PickleValue::Tuple(args) = args else {
                        return Err("a pickle REDUCE argument is not a tuple".to_string());
                    };
                    let value = reduce(&module, &name, args)?;
                    self.stack.push(value);
                }
                b'}' => self.stack.push(PickleValue::Map(Vec::new())),
                // SETITEM: value, then key, onto the map
                // below them.
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let PickleValue::Str(key) = key else {
                        return Err("a pickle dict key is not a string".to_string());
                    };
                    let Some(PickleValue::Map(pairs)) = self.stack.last_mut() else {
                        return Err("a pickle SETITEM has no dict under it".to_string());
                    };
                    set_map_entry(pairs, key, value);
                }
                // SETITEMS: key/value pairs since the
                // mark, onto the map below it.
                b'u' => {
                    let from = self.mark()?;
                    let items: Vec<PickleValue> = self.stack.drain(from..).collect();
                    if !items.len().is_multiple_of(2) {
                        return Err("a pickle SETITEMS holds an odd item count".to_string());
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
                }
                // BUILD: the state (a metadata dict in
                // torch saves) is validated as data and
                // discarded; the built object stays.
                b'b' => {
                    let state = self.pop()?;
                    if !matches!(state, PickleValue::Map(_) | PickleValue::None) {
                        return Err("a pickle BUILD state is not a dict".to_string());
                    }
                    if self.stack.is_empty() {
                        return Err("a pickle BUILD has no object under it".to_string());
                    }
                }
                // BINPUT / LONG_BINPUT: memoize the
                // stack top.
                b'q' => {
                    let at = self.byte().ok_or("the pickle ends inside BINPUT")? as usize;
                    self.put_memo(at)?;
                }
                b'r' => {
                    let at = self.u32()? as usize;
                    self.put_memo(at)?;
                }
                // BINGET / LONG_BINGET.
                b'h' => {
                    let at = self.byte().ok_or("the pickle ends inside BINGET")? as usize;
                    self.get_memo(at)?;
                }
                b'j' => {
                    let at = self.u32()? as usize;
                    self.get_memo(at)?;
                }
                b'.' => {
                    if self.stack.len() != 1 {
                        return Err(format!(
                            "the pickle ends with {} values on the stack, not one",
                            self.stack.len()
                        ));
                    }
                    return Ok(self.stack.pop().unwrap());
                }
                _ => {
                    return Err(format!(
                        "unsupported pickle opcode 0x{op:02x}: not a head state dict shape this loader reads"
                    ));
                }
            }
        }
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

    fn mark(&mut self) -> Result<usize, String> {
        self.marks
            .pop()
            .ok_or_else(|| "the pickle has a tuple or items opcode with no mark".to_string())
    }

    fn put_memo(&mut self, at: usize) -> Result<(), String> {
        if at > 4096 {
            return Err("a pickle memo index is implausibly large".to_string());
        }
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or("a pickle memo stores an empty stack")?;
        if self.memo.len() <= at {
            self.memo.resize(at + 1, None);
        }
        self.memo[at] = Some(value);
        Ok(())
    }

    fn get_memo(&mut self, at: usize) -> Result<(), String> {
        let value = self
            .memo
            .get(at)
            .cloned()
            .flatten()
            .ok_or_else(|| format!("a pickle memo reference to {at} was never stored"))?;
        self.stack.push(value);
        Ok(())
    }
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
    }
    .run()
}

#[cfg(test)]
mod tests;
