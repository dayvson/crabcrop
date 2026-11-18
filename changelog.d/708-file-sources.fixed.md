A `file://` source that names a FIFO under `CRABCROP_FILE_ROOT` answers at once
(issue #708). The file was opened with a plain open, which waits for a FIFO's
writer, so the request waited out the source fetch timeout, answered `504`, and
left a thread of the blocking pool stuck in open(2) for good: enough such
requests would take every thread the pool has. File sources now open the way the
disk tier opens its entries (issue #545): the open does not wait, and what it
opened must be a regular file. A FIFO, socket or device under the root answers
`502 source_fetch_failed`, the answer a directory under the root already got; a
device was read until the size cap. The canonicalization and the root rules are
unchanged, and a symlink that resolves inside the root reads its target as
before.
