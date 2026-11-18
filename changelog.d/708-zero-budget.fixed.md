`CRABCROP_RESULT_STORAGE_MAX_MB=0` turns the disk tier off, as
docs/configuration.md said it did (issue #708). The directory was still created
and probed and an empty store opened over it, so a directory the server could
not write stopped the boot. Now nothing touches the directory, the boot line
says `result storage: off, CRABCROP_RESULT_STORAGE_MAX_MB is 0, so <dir> is not
used`, and `/metrics` has no disk tier series, as with no directory set.
