//! A minimal in-memory zip reader, sized for head saves.

/// One entry of a zip archive, extracted.
pub(super) struct ZipEntry {
    pub(super) name: String,
    pub(super) data: Vec<u8>,
}

/// Read every entry of a zip archive held in memory.
/// Deliberately minimal, sized for the few-kilobyte
/// saves `torch.save` writes for a linear head:
/// stored and deflated entries, no encryption, no
/// zip64, no multi-disk archives; anything else is
/// an error naming the limitation. Entry payloads are
/// capped as they inflate, so a hostile archive
/// cannot expand without bound.
pub(super) fn read_zip_entries(bytes: &[u8]) -> Result<Vec<ZipEntry>, String> {
    let u16_at = |at: usize| -> Result<u16, String> {
        bytes
            .get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .ok_or_else(|| "the archive is truncated".to_string())
    };
    let u32_at = |at: usize| -> Result<u32, String> {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| "the archive is truncated".to_string())
    };
    // The end-of-central-directory record closes the
    // file; scan the largest window a comment allows,
    // from the end.
    let window = bytes.len().saturating_sub(22 + u16::MAX as usize);
    let mut eocd = None;
    let mut at = bytes.len();
    while at > window {
        at -= 1;
        if bytes[at..].starts_with(b"PK\x05\x06") {
            eocd = Some(at);
            break;
        }
    }
    let eocd = eocd.ok_or("no end-of-central-directory record: not a zip archive")?;
    if u16_at(eocd + 4)? != 0 || u16_at(eocd + 6)? != 0 {
        return Err("multi-disk archives are not supported".to_string());
    }
    let count = u16_at(eocd + 10)? as usize;
    if count > 64 {
        return Err(format!(
            "the archive holds {count} entries: a head save holds a handful"
        ));
    }
    let mut dir = u32_at(eocd + 16)? as usize;
    let mut entries = Vec::new();
    for _ in 0..count {
        if !bytes
            .get(dir..)
            .is_some_and(|b| b.starts_with(b"PK\x01\x02"))
        {
            return Err("the central directory is malformed".to_string());
        }
        let method = u16_at(dir + 10)?;
        let compressed_len = u32_at(dir + 20)? as usize;
        let plain_len = u32_at(dir + 24)? as usize;
        let name_len = u16_at(dir + 28)? as usize;
        let extra_len = u16_at(dir + 30)? as usize;
        let comment_len = u16_at(dir + 32)? as usize;
        let local_at = u32_at(dir + 42)? as usize;
        if compressed_len == u32::MAX as usize || plain_len == u32::MAX as usize {
            return Err("zip64 archives are not supported".to_string());
        }
        let name_bytes = bytes
            .get(dir + 46..dir + 46 + name_len)
            .ok_or("the central directory is truncated")?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        if !bytes
            .get(local_at..)
            .is_some_and(|b| b.starts_with(b"PK\x03\x04"))
        {
            return Err(format!("the local header for {name} is malformed"));
        }
        let data_at =
            local_at + 30 + u16_at(local_at + 26)? as usize + u16_at(local_at + 28)? as usize;
        let raw = bytes
            .get(data_at..data_at + compressed_len)
            .ok_or("the archive is truncated")?;
        let data = match method {
            0 => raw.to_vec(),
            8 => {
                use std::io::Read;
                let mut out = Vec::new();
                flate2::read::DeflateDecoder::new(raw)
                    .take((plain_len + 1) as u64)
                    .read_to_end(&mut out)
                    .map_err(|e| format!("the entry {name} does not inflate: {e}"))?;
                out
            }
            _ => {
                return Err(format!(
                    "the entry {name} uses unsupported compression method {method}"
                ));
            }
        };
        if data.len() != plain_len {
            return Err(format!(
                "the entry {name} holds {} bytes against the directory's {plain_len}",
                data.len()
            ));
        }
        entries.push(ZipEntry { name, data });
        dir += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}
