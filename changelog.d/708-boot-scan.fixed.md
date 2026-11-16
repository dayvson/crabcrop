The disk tier's boot scan no longer keeps the server from listening (issue
#708). Boot read and verified every entry file in `CRABCROP_RESULT_STORAGE_DIR`
before it bound the port, with no deadline and no log line, so a mount that hung
at boot kept `/health` from answering, and a large tier held the boot for as
long as its scan took. The scan, MAC check included, now runs on a thread of its
own, and the boot waits for it for 5 seconds at most from the store's start. A
scan that ends in time leaves every entry indexed before the server listens, as
before, and the boot line `result storage at <dir>: <n> entries, <bytes> bytes,
budget <m> MB` now also says how many files the scan read, in how long, and how
many it skipped; a line before it says the scan has begun. A scan that takes
longer goes on in the background: a warning on stderr says it is still running
and how many files it has read, again every 30 seconds, and the boot line comes
once it ends. Meanwhile an entry it has not indexed is a miss, which renders and
is written again, writes evict as always, and what the scan then indexes counts
as older than every entry used since the boot. A purge sent meanwhile answers at
once, the scan deletes the files the purge covers instead of indexing them, and
with a shared tier the purge's tombstone stays in force until they are gone. The
directory check before the scan (creating the directory, writing the probe file,
opening it) has the same 5 seconds: a failure within them stops the boot as
before, and a directory that has not answered by then leaves a disk tier that
serves and stores nothing until it does. A shutdown during the scan stops it,
and the next boot scans again. An entry file stored under another key's name,
which the store never writes, is now deleted at boot like a file that does not
verify.
