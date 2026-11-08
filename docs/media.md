# Media sources

## Format negotiation

When the URL names no `format(...)`, crabcrop picks the output format from the request's `Accept` header. Among the negotiable formats (AVIF, JPEG XL, WebP), the one with the highest q-value wins; a missing q counts as 1. Ties break by server preference, in this order:

1. `image/avif` → AVIF
2. `image/jxl` → JPEG XL
3. `image/webp` → WebP

So `Accept: image/avif;q=0.5, image/webp` produces WebP: the client's stronger preference beats the server's. If none of the three is offered, the source image's format is kept (JPEG if the format cannot be detected). A format refused with `q=0` is never picked, and wildcards such as `*/*` do not count as an offer. An explicit `format(...)` in the URL always wins over negotiation. Animated GIF and WebP sources are the one exception: negotiation never drops their frames. A negotiated GIF or WebP keeps the frames as an animation in that format, and a negotiated AVIF or JPEG XL falls back to the source format instead (GIF stays GIF, WebP stays WebP), since neither is served animated here. Video is the other exception: `mp4` and `webm` are never negotiated at all, and only ever come from an explicit `format(...)` (see [Video output](#video-output)). Responses include a `Vary` covering `Accept` and the client hints so downstream caches key this correctly (the hints likewise change sizing and quality; see [Client hints](filters.md#client-hints)).

## Video output

An animated GIF or WebP can also be served as real video: H.264 MP4 or VP9 WebM, the gifv pattern that plays inline everywhere and weighs a fraction of the GIF. Ask with an explicit format filter:

```
/<signature>/filters:format(mp4)/https://example.com/animation.gif
```

Crabcrop ships no codec. The encode runs in an external program you configure, `CRABCROP_VIDEO_ENCODER`: the path to an encoder executable on the server, with FFmpeg as the reference. The path is validated at startup (it must exist and be executable, or the server refuses to boot). With the variable unset, video is off and a video URL answers `501`, exactly like an AI filter with no model. And there is no silent fallback: a failed encode is an error response, never the GIF.

### The encoder contract

Crabcrop runs the program directly, with no shell and an argv it constructs; only the program path is yours. Frames go in on stdin as raw RGBA and the video comes out on stdout:

```
<encoder> -nostdin -v error -f rawvideo -pix_fmt rgba -s <W>x<H> -r <fps> -i pipe:0 -an -vf format=yuv420p <codec args> pipe:1
```

For MP4 the codec args are `-c:v libx264 -preset veryfast -crf 23 -movflags frag_keyframe+empty_moov+default_base_moof -f mp4`. The fragmented-MP4 flags are what make MP4 writable to a pipe at all; a plain MP4 needs a seekable file to finish its index. For WebM they are `-c:v libvpx-vp9 -b:v 0 -crf 30 -f webm`. A wrapper script that passes its arguments through to a real encoder (one ending in `exec ffmpeg "$@"`, say) satisfies the contract, which is how you bring an encoder build of your own. Because the codec lives outside crabcrop, so does codec licensing: H.264 patent pools and friends are the operator's concern, and that separation is the point. Output bytes vary with the encoder build, so byte-for-byte reproducibility across deployments is not promised; the URL grammar and the errors are.

Details an encoder setup has to match:

- Frames are composited over black and padded up to even dimensions (H.264 and VP9 need them) by replicating the edge pixels. The response describes the padded dimensions.
- Frame timing: the pipe contract carries one constant rate, so crabcrop picks it from the fastest frame, `ceil(1000 / shortest delay)` clamped to 1-100 fps (10 fps when no frame carries a delay), and repeats each frame `round(delay × fps / 1000)` times, at least once. Total duration is preserved to within one frame period per source frame. A source whose delays total more than 600 s is refused with a `400`.
- At most two encodes run at a time, each under a hard wall-clock timeout (`CRABCROP_VIDEO_TIMEOUT_SECS`, default 30 s) after which the encoder is killed, and a hard output cap (`CRABCROP_VIDEO_MAX_MB`, default 64 MB) that aborts the encode the moment it is crossed. The encoder runs with an empty environment; nothing is inherited from the server's.
- The program must not daemonize or detach (no `setsid`, no double-fork). A timeout kills the program's whole process group; a descendant that detaches escapes that kill and is the operator's responsibility. The video decoder and the PDF renderer are held to the same rule.
- On Linux the encoder also runs under kernel resource limits, a backstop behind the wall clock: 8 GiB of address space, and CPU time of the timeout times one more than the cores the server may use, plus 5 s (155 CPU seconds at the default 30 s on a 4-core host). The kernel counts CPU time across all of the program's threads, and FFmpeg works on several, so the limit scales with the cores: a run that fits the wall clock never reaches it.

### Errors

| Situation | Answer |
| --- | --- |
| No encoder configured | `501`: `video output is not configured on this server` |
| Still source (a still image, or a one-frame GIF or WebP) | `400`: the source is a still image, and video needs an animated source |
| Animation over the 600 s cap | `400` |
| Encoder exits nonzero, produces nothing, exceeds the cap, or times out | `502`: `video encoding failed` (the cause, including the encoder's stderr excerpt, is in the server log) |
| Encode queue full (both slots busy past a 10 s wait) | `503` |

Video results cache like any other response, under the video content type (`video/mp4` or `video/webm`), and every operation in the URL (resize, crop, filters) applies per frame before the encode. A meta request (`Accept: application/json`) on a video URL answers `400`: meta describes images, and a video result is not one.

## Video frames

A video the server can fetch is also an image source: a `frame:` segment in the URL selects one frame (see [Video frame](operations.md#video-frame) in the operations), and from there the request is an ordinary image request. The frame is the poster's first frame, the frame at a timestamp, or the frame at a share of the duration; out-of-range selections clamp to the last frame. Extraction happens once per unique URL, on a cache miss, and the result caches (and purges) like any other render, under the same key rules as every spelling of the URL.

Crabcrop ships no video decoder either. Extraction runs in an external program you configure, `CRABCROP_VIDEO_DECODER`: the path to a decoder executable, with FFmpeg as the reference. The path is validated at startup (it must exist and be executable, or the server refuses to boot). With the variable unset, frame selection is off and a `frame:` URL answers `501`. The contract between crabcrop and the program is fixed, and crabcrop builds every argument itself; the program never sees a shell:

- **Input**: the source bytes arrive on stdin as a seekable file, not a pipe. Crabcrop stages them once per extraction in a private temporary file (in `TMPDIR`, at most one fetch-capped source per running extraction) and unlinks it at once, so only the open descriptor reaches the bytes and nothing is left behind. The program reads them through FFmpeg's `fd` protocol, which seeks on a regular file; that is what makes an MP4 with its index (the `moov` box) at the end, FFmpeg's own default layout, readable, because the demuxer has to seek back to the samples after reading the index. Faststart MP4s and every other container work the same way. The `fd` protocol arrived in FFmpeg 6.0, so frame extraction needs FFmpeg 6.0 or newer; an older build refuses the input, and the request answers `502` with the cause named in the server log.
- **Probe**: argv is `ffmpeg -nostdin -v info -protocol_whitelist fd -format_whitelist <containers> -dump_separator <token> -i fd:`, with the source on stdin and no output target. The program reports the input on stderr (duration, stream list); crabcrop reads the duration, the first video stream's dimensions, and its frame rate from that report. A source whose report holds no video stream (a non-video, an audio-only file, a corrupt container) is a bad source. The report also prints the source's own metadata (title, comment, chapter and stream tags, language tags), raw, so crabcrop reads only what the source cannot write. The duration comes from the input's single `Duration:` line at the report's two-space indent, in plain digits (never `NaN` or `inf`). Stream facts come only from codec descriptions that carry `<token>`, a fresh random field separator for each probe (FFmpeg's `dump_separator` option): FFmpeg prints it inside each stream's codec description, after the codec name, and the source cannot know it. A report with a second `Duration:` line, or longer than 256 KiB, is a bad source too, rather than a source to take at its word. A program that ignores `-dump_separator` reports no stream crabcrop can read, so every frame request through it fails as a bad source.
- **Extract**: argv is `ffmpeg -nostdin -v info -protocol_whitelist fd -format_whitelist <containers> -i fd: -ss <seconds> -frames:v 1 -f image2pipe -vcodec png pipe:1`, again with the source on stdin. One PNG frame comes back on stdout and becomes the source image, so every decode and output budget applies to it unchanged.

The whitelists pin the program to the bytes the loader already validated: the protocol whitelist leaves stdin as the only input channel (`fd` reads a descriptor the program already holds and refuses one named in a URL, and no `file` or network protocol is allowed), and the format whitelist names the container formats (the MP4/MOV family, Matroska and WebM, AVI, MPEG, FLV, ASF, Ogg), so a manifest inside the source that points at a `file://` or network target, DASH or HLS, is refused by the program instead of followed.

The bounds mirror the encoder's, because the work is the same class:

- At most two extractions run at a time behind the video gate (shared with encodes), on top of the slow lane; a request that waits more than 10 s for a slot sheds with `503`.
- Each run (probe and extraction alike) is bounded by the same wall-clock timeout as encodes (`CRABCROP_VIDEO_TIMEOUT_SECS`, default 30 s), after which the program's whole process group is killed; the program runs with an empty environment, and its output is capped (a frame whose PNG passes 256 MiB, the decode allocation budget, is refused).
- Each run also gets the encoder's kernel limits (address space, and CPU time scaled by the cores the server may use; see [the encoder details](#the-encoder-contract)), so a multi-threaded decode that fits the wall clock, such as a far seek in a long 4K clip, is never cut short by the CPU limit.
- The source itself flows through the ordinary loader first, with the same allowlist, size cap (`CRABCROP_SOURCE_MAX_MB`, 10 MiB by default: raise it for longer clips), fetch timeout, and redirect rules as any source. Only a URL carrying a `frame:` segment may fetch a `video/*` Content-Type; every other request keeps the image-only rule.

### Errors

| Situation | Answer |
| --- | --- |
| No decoder configured | `501`: `frame extraction is not configured on this server` |
| Source is an image (still or animated) | `502`: frame selection needs a video source |
| Source has no video stream, is not a readable video, or its metadata plants a second `Duration:` line in the probe report | `502`: the bad-source failure every undecodable source gets |
| Percentage selection on a video that reports no duration | `400` |
| Malformed `frame:` selector | `400` with the `invalid_frame` code, before any fetch |
| Decoder exits nonzero, produces no frame, exceeds the cap, or times out; decoder older than FFmpeg 6.0 (no `fd` protocol); source cannot be staged in the temporary directory | `502`: `video frame extraction failed` (the cause, including the decoder's stderr excerpt, is in the server log) |
| Video queue full (both slots busy past a 10 s wait) | `503` |

A frame render's [meta response](meta.md#meta-responses) carries a `video` object: the source's `width` and `height`, its `duration` in seconds when the container reports one, and the `frame_timestamp` the frame was selected at, after clamping. The explain endpoint labels a video source's kind as `video`; its dimensions stay unknown there, because explain never runs the decoder.

## PDF pages

A PDF the server can fetch is also an image source: a `page:` segment in the URL selects one page (see [PDF page](operations.md#pdf-page) in the operations), and from there the request is an ordinary image request. The page renders at a fixed scale of 2 (a 144 dpi raster), clamped down for very large pages so neither raster axis passes the 16384 px decode cap; the rendered PNG re-enters the pipeline, so every decode budget, filter, and output rule applies to it unchanged. Rendering happens once per unique URL, on a cache miss, and the result caches (and purges) like any other render, under the same key rules as every spelling of the URL: `page:7` and `page:07` are one entry.

Crabcrop ships no PDF renderer. Rendering runs in an external program you configure, `CRABCROP_PDF_RENDERER`: the path to a renderer executable, with Poppler's `pdftocairo` as the reference. The renderer does not come alone: crabcrop derives the `pdfinfo` program next to it (same directory, same extension) and uses it to probe documents. Both paths are validated at startup (they must exist and be executable, or the server refuses to boot). With the variable unset, page selection is off and a `page:` URL answers `501`. The contract between crabcrop and the programs is fixed, and crabcrop builds every argument itself; the programs never see a shell, a file path, or the source URL:

- **Probe**: argv is `pdfinfo -f <page> -l <page> -`, with the PDF bytes on stdin. Crabcrop reads the page count, the encryption state, and the selected page's size in points from the report. When the probe refuses the range itself, which it does past the end of the document, a whole-document probe confirms the count for the refusal.
- **Render**: argv is `pdftocairo -png -singlefile -f <page> -l <page> -r <dpi> - -`, again with the PDF bytes on stdin. One PNG page comes back on stdout and becomes the source image.

Because the tools receive only bytes over stdin, nothing inside a document has anything to resolve: external references, embedded files, and document actions or scripts have no target and no effect. The source itself flows through the ordinary loader first, with the same allowlist and redirect rules as any source and the same fetch cap (`CRABCROP_SOURCE_MAX_MB`, 10 MiB by default: raise it for larger documents); only a URL carrying a `page:` segment may fetch an `application/pdf` Content-Type. Documents over 1000 pages are refused.

The bounds mirror the frame extractor's, because the work is the same class:

- Page work takes the video gate (at most two runs at a time, shared with encodes and frame extractions), on top of the slow lane; a request that waits more than 10 s for a slot sheds with `503`.
- Each run is bounded by `CRABCROP_PDF_TIMEOUT_SECS` (default 30 s), after which the program's whole process group is killed; the program runs with an empty environment, and its output is capped (a page whose PNG passes 256 MiB, the decode allocation budget, is refused).
- Each run also gets the same kernel limits as the video tools (address space, and CPU time scaled by the cores the server may use; see [the encoder details](#the-encoder-contract)), here against `CRABCROP_PDF_TIMEOUT_SECS`.

### Errors

| Situation | Answer |
| --- | --- |
| No renderer configured | `501`: `page rendering is not configured on this server` |
| Source is an image, a video, or anything else that is not a PDF | `502`: page selection needs a PDF source |
| Source is not a readable PDF (junk bytes, a corrupt or truncated document) | `502`: the bad-source failure every undecodable source gets |
| Document over the 1000-page cap | `502`, naming the count and the cap |
| Page past the end of the document | `400`, naming the page and the document's page count |
| Document needs a password to open | `400`: the document is password-protected |
| Malformed `page:` selector | `400` with the `invalid_page` code, before any fetch |
| Renderer or probe exits nonzero, produces no page, exceeds the cap, or times out | `502`: `PDF rendering failed` (the cause, including the tool's stderr excerpt, is in the server log) |
| Video queue full (both slots busy past a 10 s wait) | `503` |

A page render's [meta response](meta.md#meta-responses) carries a `pdf` object: the document's `pages` count, the `page` rendered, and the page's `page_width` and `page_height` in points. The explain endpoint labels a PDF source's kind as `pdf`; its dimensions stay unknown there, because explain never runs the renderer.
