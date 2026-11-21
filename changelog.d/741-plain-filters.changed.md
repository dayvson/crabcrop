In the library, the filters that `crabcrop::filters` applies from the image
alone have an enum of their own (issue #741). The 36 such variants of
`crabcrop::url::Filter` move under `Filter::Plain(crabcrop::url::PlainFilter)`,
and `crabcrop::filters::apply_filters_filled` and `apply_filters_within` take
`&[PlainFilter]`, so a filter the engine applies itself can no longer reach
that match and be skipped there. The URL grammar, `/explain`, the cache keys
and every render are unchanged.
