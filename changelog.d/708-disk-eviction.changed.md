The disk tier finds the entry to evict without reading every entry (issue #708),
as the memory tier does since issue #645. To evict, the tier read the last-use
stamp of every indexed entry, once for each entry evicted, under the index lock,
so a full tier walked its whole index on every write while lookups waited. The
keys are now kept in order of last use, and eviction takes the first. Which
entry is evicted and when, what an entry is charged against
`CRABCROP_RESULT_STORAGE_MAX_MB`, and every response are unchanged.
