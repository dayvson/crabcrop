//! One decode of the render for all the annotation sections
//! of a meta document (issue #676), on the blocking pool and
//! with a decoder's panic contained there (issue #736).

use super::*;
use crate::ai::{
    DetectionProvider, EmbeddingProvider, ModerationProvider, OcrProvider, TagProvider,
};
use std::sync::atomic::AtomicBool;

/// One provider call: who was called, where the frame was,
/// and its size.
type Seen = (&'static str, usize, (u32, u32));

/// The frames the providers were handed.
#[derive(Debug, Clone, Default)]
struct FrameLog(Arc<std::sync::Mutex<Vec<Seen>>>);

impl FrameLog {
    fn note(&self, who: &'static str, img: &DynamicImage) {
        let at = std::ptr::from_ref(img) as usize;
        self.0
            .lock()
            .unwrap()
            .push((who, at, (img.width(), img.height())));
    }

    fn take(&self) -> Vec<Seen> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// The fakes of each annotation kind, noting the frame they
/// are handed before they answer.
#[derive(Debug)]
struct LoggingTagger(FrameLog);

impl TagProvider for LoggingTagger {
    fn scores(&self, img: &DynamicImage) -> Result<Vec<crate::ai::TagScore>, crate::ai::AiError> {
        self.0.note("tags", img);
        FakeTagger.scores(img)
    }
}

#[derive(Debug)]
struct LoggingDetector(FrameLog, &'static str);

impl DetectionProvider for LoggingDetector {
    fn knows_class(&self, name: &str) -> bool {
        FakeDetector.knows_class(name)
    }
    fn describe(&self) -> Vec<String> {
        FakeDetector.describe()
    }
    fn detect(&self, img: &DynamicImage) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
        self.0.note(self.1, img);
        FakeDetector.detect(img)
    }
}

#[derive(Debug)]
struct LoggingModeration(FrameLog);

impl ModerationProvider for LoggingModeration {
    fn moderate(
        &self,
        img: &DynamicImage,
    ) -> Result<crate::ai::ModerationScores, crate::ai::AiError> {
        self.0.note("moderation", img);
        FakeModeration.moderate(img)
    }
}

#[derive(Debug)]
struct LoggingEmbedding(FrameLog);

impl EmbeddingProvider for LoggingEmbedding {
    fn embed(&self, img: &DynamicImage) -> Result<crate::ai::ImageEmbedding, crate::ai::AiError> {
        self.0.note("embedding", img);
        FakeEmbedding.embed(img)
    }
}

#[derive(Debug)]
struct LoggingOcr(FrameLog);

impl OcrProvider for LoggingOcr {
    fn recognize(&self, img: &DynamicImage) -> Result<crate::ai::OcrReport, crate::ai::AiError> {
        self.0.note("ocr", img);
        FakeOcr.recognize(img)
    }
}

/// A state with a provider for every section, and two
/// detectors, all noting their frames in `log`.
fn logging_state(log: &FrameLog) -> AppState {
    let mut state = test_state();
    let providers = providers_mut(&mut state);
    providers.tagger = Some(Arc::new(LoggingTagger(log.clone())));
    providers.detect = Some(Arc::new(LoggingDetector(log.clone(), "builtin")));
    providers.detectors = vec![NamedDetector {
        name: "crates".to_string(),
        provider: Arc::new(LoggingDetector(log.clone(), "crates")),
    }];
    providers.moderation = Some(Arc::new(LoggingModeration(log.clone())));
    providers.embeddings = Some(Arc::new(LoggingEmbedding(log.clone())));
    providers.ocr = Some(Arc::new(LoggingOcr(log.clone())));
    state
}

/// A request for every section a client can opt into.
const EVERY_SECTION: &str = "application/json; detections=1; moderation=1; embeddings=1; ocr=1";

/// A response's status, headers and body, to compare two.
async fn parts(response: Response) -> (StatusCode, String, String) {
    let status = response.status();
    let headers = format!("{:?}", response.headers());
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, headers, String::from_utf8(body.to_vec()).unwrap())
}

/// The engine's decoder, counting its calls. One counter
/// for each test: the tests run side by side.
macro_rules! counting_decoder {
    ($decode:ident, $count:ident) => {
        static $count: AtomicUsize = AtomicUsize::new(0);
        fn $decode(bytes: &[u8]) -> Result<DynamicImage, crate::engine::EngineError> {
            $count.fetch_add(1, Ordering::SeqCst);
            crate::engine::decode_image(bytes)
        }
    };
}

counting_decoder!(decode_for_the_document, DOCUMENT_DECODES);
counting_decoder!(decode_what_does_not_decode, FAILED_DECODES);
counting_decoder!(decode_behind_the_gate, GATED_DECODES);

#[tokio::test]
async fn a_meta_document_decodes_its_render_once_for_every_section() {
    // Issue #676: tags, detections from two detectors,
    // moderation, the embedding and OCR used to decode the
    // render once each. The document now decodes it once,
    // every provider is handed that one frame, and the
    // response is the one the sections give when each
    // decodes for itself.
    let log = FrameLog::default();
    let state = logging_state(&log);
    let processed = still_processed(test_png(), ImageFormat::Png);
    let frame = DocumentFrame::with_decoder(Arc::new(processed.clone()), decode_for_the_document);
    let shared = meta_document_from(&state, &processed, frame, false, Some(EVERY_SECTION)).await;
    assert_eq!(DOCUMENT_DECODES.load(Ordering::SeqCst), 1);
    let seen = log.take();
    let order: Vec<&str> = seen.iter().map(|s| s.0).collect();
    assert_eq!(
        order,
        [
            "tags",
            "builtin",
            "crates",
            "moderation",
            "embedding",
            "ocr"
        ]
    );
    for (who, at, size) in &seen {
        assert_eq!((*at, *size), (seen[0].1, (100, 60)), "{who}");
    }
    let alone = processed.clone();
    let alone = meta_document_from(&state, &processed, alone, false, Some(EVERY_SECTION)).await;
    assert_eq!(log.take().len(), 6);
    let (shared, alone) = (parts(shared).await, parts(alone).await);
    assert_eq!(shared, alone);
    assert_eq!(shared.0, StatusCode::OK);
    let body: serde_json::Value = serde_json::from_str(&shared.2).unwrap();
    for section in ["tags", "detections", "moderation", "embedding", "ocr"] {
        assert!(body.get(section).is_some(), "{section} is missing: {body}");
    }
    assert_eq!(body["detections"].as_array().unwrap().len(), 2);
    assert_eq!(state.ai_gate.available_permits(), 2);
}

#[tokio::test]
async fn an_undecodable_render_degrades_each_section_on_its_own() {
    // The degrade-not-fail rule on the shared frame: no
    // section reaches its provider, tags and detections
    // are omitted, the sections asked for by name report
    // the failure, and every permit comes back. A failed
    // decode is not remembered: each section asks for the
    // pixels itself, as it always did.
    let log = FrameLog::default();
    let state = logging_state(&log);
    let processed = still_processed(b"these bytes are no image".to_vec(), ImageFormat::Png);
    let frame =
        DocumentFrame::with_decoder(Arc::new(processed.clone()), decode_what_does_not_decode);
    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    assert!(annotate_meta(&state, &frame).await.is_none());
    assert!(detect_meta(&state, &frame).await.is_none());
    assert_eq!(
        moderate_meta(&state, &frame).await,
        ModerationMeta::Unavailable("inference_failed")
    );
    assert_eq!(
        embed_meta(&state, &frame).await,
        EmbeddingMeta::Unavailable("inference_failed")
    );
    assert_eq!(
        ocr_meta(&state, &frame, deadline).await,
        OcrMeta::Unavailable("inference_failed")
    );
    assert!(log.take().is_empty());
    assert_eq!(FAILED_DECODES.load(Ordering::SeqCst), 5);
    assert_eq!(state.ai_gate.available_permits(), 2);
    // The document still answers, as it did when each
    // section decoded for itself.
    let frame =
        DocumentFrame::with_decoder(Arc::new(processed.clone()), decode_what_does_not_decode);
    let shared = meta_document_from(&state, &processed, frame, false, Some(EVERY_SECTION)).await;
    let alone = processed.clone();
    let alone = meta_document_from(&state, &processed, alone, false, Some(EVERY_SECTION)).await;
    let (shared, alone) = (parts(shared).await, parts(alone).await);
    assert_eq!(shared, alone);
    assert_eq!(shared.0, StatusCode::OK);
    let body: serde_json::Value = serde_json::from_str(&shared.2).unwrap();
    assert!(body.get("tags").is_none() && body.get("detections").is_none());
    assert!(shared.2.contains("inference_failed"), "{}", shared.2);
    assert!(log.take().is_empty());
}

#[tokio::test]
async fn a_section_takes_its_permit_before_it_asks_for_the_frame() {
    // Every section still takes its own AI permit, and
    // takes it first: with the gate's only permit held
    // elsewhere the sections wait, and nothing is decoded
    // for a section that may never run. Once the permit
    // returns each section answers, one after the other,
    // from one decode.
    let log = FrameLog::default();
    let mut state = logging_state(&log);
    state.ai_gate = Arc::new(Gate::new(1));
    let processed = still_processed(test_png(), ImageFormat::Png);
    let frame = DocumentFrame::with_decoder(Arc::new(processed.clone()), decode_behind_the_gate);
    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    let held = state.ai_gate.clone().acquire_owned().await.unwrap();
    let waiting = tokio::time::timeout(std::time::Duration::from_millis(200), async {
        tokio::join!(
            annotate_meta(&state, &frame),
            detect_meta(&state, &frame),
            moderate_meta(&state, &frame),
            embed_meta(&state, &frame),
            ocr_meta(&state, &frame, deadline),
        )
    })
    .await;
    assert!(waiting.is_err(), "a section ran without a permit");
    assert_eq!(GATED_DECODES.load(Ordering::SeqCst), 0);
    assert!(log.take().is_empty());
    drop(held);
    assert!(annotate_meta(&state, &frame).await.is_some());
    assert!(detect_meta(&state, &frame).await.is_some());
    assert!(matches!(
        moderate_meta(&state, &frame).await,
        ModerationMeta::Report(_)
    ));
    assert!(matches!(
        embed_meta(&state, &frame).await,
        EmbeddingMeta::Report(_)
    ));
    assert!(matches!(
        ocr_meta(&state, &frame, deadline).await,
        OcrMeta::Report(_)
    ));
    assert_eq!(GATED_DECODES.load(Ordering::SeqCst), 1);
    assert_eq!(log.take().len(), 6);
    assert_eq!(state.ai_gate.available_permits(), 1);
}

#[tokio::test]
async fn the_annotation_deadline_still_ends_a_document_on_a_shared_frame() {
    // The sections still run under the annotation deadline:
    // a tagger that does not answer is left out when it
    // passes, the sections behind it report that they timed
    // out, and the document goes out. The frame the stuck
    // inference holds is the document's own, not a second
    // one.
    let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let log = FrameLog::default();
    let mut state = logging_state(&log);
    state.process_timeout = std::time::Duration::from_millis(300);
    providers_mut(&mut state).tagger = Some(Arc::new(BlockingTagger {
        release: release.clone(),
    }));
    let processed = still_processed(test_png(), ImageFormat::Png);
    let frame =
        DocumentFrame::with_decoder(Arc::new(processed.clone()), crate::engine::decode_image);
    let started = Instant::now();
    let response = meta_document_from(&state, &processed, frame, false, Some(EVERY_SECTION)).await;
    assert!(started.elapsed() < std::time::Duration::from_secs(20));
    let (status, _, body) = parts(response).await;
    assert_eq!(status, StatusCode::OK);
    let body: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(body.get("tags").is_none(), "{body}");
    assert!(body.to_string().contains("timed_out"), "{body}");
    open_release(&release);
    let mut back = false;
    for _ in 0..100 {
        if state.ai_gate.available_permits() == 2 {
            back = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(back, "every permit returns once the inferences do");
}

/// Set while [`decode_beside_the_runtime`] waits.
static DECODING: AtomicBool = AtomicBool::new(false);
/// Set by [`run_beside_the_decode`] once it has run during a
/// decode.
static RAN_BESIDE: AtomicBool = AtomicBool::new(false);
static BESIDE_DECODES: AtomicUsize = AtomicUsize::new(0);

/// The engine's decoder, once another future on the test's
/// runtime has run during the decode. On a runtime of one
/// thread that can only happen when the decode runs off that
/// thread: a decode on it holds the thread, so the other
/// future never runs, and after 5 seconds the decode gives up
/// as a render that does not decode.
fn decode_beside_the_runtime(bytes: &[u8]) -> Result<DynamicImage, crate::engine::EngineError> {
    BESIDE_DECODES.fetch_add(1, Ordering::SeqCst);
    DECODING.store(true, Ordering::SeqCst);
    let started = Instant::now();
    while !RAN_BESIDE.load(Ordering::SeqCst) {
        if started.elapsed() > std::time::Duration::from_secs(5) {
            return Err(crate::engine::EngineError::Decode(
                "nothing else ran on the runtime while the render decoded".to_string(),
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    crate::engine::decode_image(bytes)
}

/// The other future: it waits, giving the runtime's thread
/// up, until a decode is under way, and then lets it finish.
async fn run_beside_the_decode() {
    while !DECODING.load(Ordering::SeqCst) {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    RAN_BESIDE.store(true, Ordering::SeqCst);
}

#[tokio::test(flavor = "current_thread")]
async fn the_render_decodes_off_the_runtime_thread() {
    // Issue #736: each section decoded the shared frame on
    // the runtime's worker, before its blocking task, so a
    // large render's decode held that worker from every other
    // request scheduled on it. This runtime has one thread,
    // and its decoder cannot finish unless another future
    // runs on that thread meanwhile. Each section, the first
    // to ask on a frame of its own, answers from the pixels.
    let log = FrameLog::default();
    let mut state = logging_state(&log);
    // Room for every section at once, for the last part.
    state.ai_gate = Arc::new(Gate::new(5));
    let processed = Arc::new(still_processed(test_png(), ImageFormat::Png));
    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    let fresh = || {
        DECODING.store(false, Ordering::SeqCst);
        RAN_BESIDE.store(false, Ordering::SeqCst);
        DocumentFrame::with_decoder(Arc::clone(&processed), decode_beside_the_runtime)
    };
    let frame = fresh();
    let (tags, ()) = tokio::join!(annotate_meta(&state, &frame), run_beside_the_decode());
    assert!(tags.is_some(), "the tags' decode held the runtime's thread");
    let frame = fresh();
    let (boxes, ()) = tokio::join!(detect_meta(&state, &frame), run_beside_the_decode());
    assert_eq!(
        boxes.map(|boxes| boxes.len()),
        Some(2),
        "the detections' decode held the runtime's thread"
    );
    let frame = fresh();
    let (moderation, ()) = tokio::join!(moderate_meta(&state, &frame), run_beside_the_decode());
    assert!(
        matches!(moderation, ModerationMeta::Report(_)),
        "the moderation's decode held the runtime's thread: {moderation:?}"
    );
    let frame = fresh();
    let (embedding, ()) = tokio::join!(embed_meta(&state, &frame), run_beside_the_decode());
    assert!(
        matches!(embedding, EmbeddingMeta::Report(_)),
        "the embedding's decode held the runtime's thread: {embedding:?}"
    );
    let frame = fresh();
    let (ocr, ()) = tokio::join!(ocr_meta(&state, &frame, deadline), run_beside_the_decode());
    assert!(
        matches!(ocr, OcrMeta::Report(_)),
        "the OCR's decode held the runtime's thread: {ocr:?}"
    );
    assert_eq!(BESIDE_DECODES.load(Ordering::SeqCst), 5);
    assert_eq!(log.take().len(), 6);
    // All five at once on one frame: the section that asks
    // first decodes it, and the others wait for that frame
    // rather than decode their own.
    let frame = fresh();
    let ((tags, boxes, moderation, embedding, ocr), ()) = tokio::join!(
        async {
            tokio::join!(
                annotate_meta(&state, &frame),
                detect_meta(&state, &frame),
                moderate_meta(&state, &frame),
                embed_meta(&state, &frame),
                ocr_meta(&state, &frame, deadline),
            )
        },
        run_beside_the_decode(),
    );
    assert!(tags.is_some() && boxes.is_some());
    assert!(matches!(moderation, ModerationMeta::Report(_)));
    assert!(matches!(embedding, EmbeddingMeta::Report(_)));
    assert!(matches!(ocr, OcrMeta::Report(_)));
    assert_eq!(BESIDE_DECODES.load(Ordering::SeqCst), 6);
    let seen = log.take();
    assert_eq!(seen.len(), 6);
    for (who, at, size) in &seen {
        assert_eq!((*at, *size), (seen[0].1, (100, 60)), "{who}");
    }
    assert_eq!(state.ai_gate.available_permits(), 5);
}

static PANICKED_DECODES: AtomicUsize = AtomicUsize::new(0);

/// A decoder with a bug: the render's bytes make it panic.
fn decode_that_panics(_: &[u8]) -> Result<DynamicImage, crate::engine::EngineError> {
    PANICKED_DECODES.fetch_add(1, Ordering::SeqCst);
    panic!("a decoder bug: these bytes break it");
}

/// A decoder that refuses the render's bytes, as it refuses
/// bytes that are no image.
fn decode_that_refuses(_: &[u8]) -> Result<DynamicImage, crate::engine::EngineError> {
    Err(crate::engine::EngineError::Decode(
        "not an image".to_string(),
    ))
}

#[tokio::test]
async fn a_decoder_panic_fails_its_sections_and_not_the_request() {
    // Issue #736: the decode ran on the request's own task,
    // outside any catch_unwind, so a decoder that panicked
    // unwound that task and the client lost its answer. The
    // panic is now caught in the blocking task that decodes
    // and answers as a render that does not decode: tags and
    // detections are left out, the sections asked for by name
    // report the failure, no provider runs, every permit comes
    // back, and the document is the one a decoder that refuses
    // the bytes gives.
    let log = FrameLog::default();
    let state = Arc::new(logging_state(&log));
    let processed = Arc::new(still_processed(test_png(), ImageFormat::Png));
    // The sections run on a task of their own, as a
    // request's do: a panic that unwound them ends that task
    // with it.
    let sections = tokio::spawn({
        let state = Arc::clone(&state);
        let frame = DocumentFrame::with_decoder(Arc::clone(&processed), decode_that_panics);
        async move {
            let deadline = Instant::now() + std::time::Duration::from_secs(60);
            (
                annotate_meta(&state, &frame).await,
                detect_meta(&state, &frame).await,
                moderate_meta(&state, &frame).await,
                embed_meta(&state, &frame).await,
                ocr_meta(&state, &frame, deadline).await,
            )
        }
    })
    .await;
    let (tags, boxes, moderation, embedding, ocr) =
        sections.unwrap_or_else(|e| panic!("the decoder's panic unwound the sections' task: {e}"));
    assert!(tags.is_none() && boxes.is_none());
    assert_eq!(moderation, ModerationMeta::Unavailable("inference_failed"));
    assert_eq!(embedding, EmbeddingMeta::Unavailable("inference_failed"));
    assert_eq!(ocr, OcrMeta::Unavailable("inference_failed"));
    assert!(log.take().is_empty());
    // As for a render that does not decode, each section
    // tried for itself.
    assert_eq!(PANICKED_DECODES.load(Ordering::SeqCst), 5);
    assert_eq!(state.ai_gate.available_permits(), 2);
    let document = |decode: FrameDecoder| {
        let state = Arc::clone(&state);
        let processed = Arc::clone(&processed);
        tokio::spawn(async move {
            let frame = DocumentFrame::with_decoder(Arc::clone(&processed), decode);
            let response =
                meta_document_from(&state, &processed, frame, false, Some(EVERY_SECTION)).await;
            parts(response).await
        })
    };
    let panicked = document(decode_that_panics)
        .await
        .unwrap_or_else(|e| panic!("the decoder's panic unwound the document's task: {e}"));
    let refused = document(decode_that_refuses).await.unwrap();
    assert_eq!(panicked, refused);
    assert_eq!(panicked.0, StatusCode::OK);
    assert!(panicked.2.contains("inference_failed"), "{}", panicked.2);
    assert!(log.take().is_empty());
}
