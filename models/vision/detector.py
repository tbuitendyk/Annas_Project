"""
models/vision/detector.py — Hybrid video content detector.

Two backends, combined:
  • NudeNet  — explicit nudity, WITH bounding boxes (enables 'mild' region blur)
  • CLIP     — zero-shot scoring of fuzzy/contextual categories (kissing,
               bedroom/suggestive, violence, gore) via text prompts; frame-level

Both backends are lazy-loaded and degrade gracefully: if a library or its
weights are unavailable, that backend simply contributes no hits and the
detector keeps working with whatever loaded. Device is auto-selected (GPU if
present, else CPU) unless overridden.

NOTE: the model glue here targets common library APIs but MUST be validated and
threshold-tuned on a real machine with the models installed — none of the ML
backends can run in CI. The pure decision logic (decide_frame_action) is the
part covered by unit tests.
"""

from dataclasses import dataclass, field
from loguru import logger

from config import SEVERITY_ACTIONS


# ── Detector output ─────────────────────────────────────────────────────────

@dataclass
class CategoryHit:
    key: str                       # category key (matches config.CONTENT_CATEGORIES)
    score: float                   # 0..1 confidence
    regions: list = field(default_factory=list)  # [(x, y, w, h), ...] frame coords


# ── Category mappings / prompts ─────────────────────────────────────────────

# NudeNet detection class -> our category key. Only the explicit "_EXPOSED"
# classes are treated as nudity; covered/suggestive classes are left to CLIP.
NUDENET_TO_CATEGORY = {
    "FEMALE_BREAST_EXPOSED":   "nudity",
    "FEMALE_GENITALIA_EXPOSED":"nudity",
    "MALE_GENITALIA_EXPOSED":  "nudity",
    "BUTTOCKS_EXPOSED":        "nudity",
    "ANUS_EXPOSED":            "nudity",
    "MALE_BREAST_EXPOSED":     "nudity",
}

# CLIP zero-shot prompts. For each category we score the image against its
# positive prompts vs. the shared NEUTRAL prompts; the category fires when the
# summed positive probability exceeds its threshold.
CLIP_PROMPTS = {
    "sexual_content":     ["people having sex", "an explicit sex scene",
                           "sexual intercourse"],
    "bedroom_suggestive": ["a suggestive bedroom scene", "people in underwear in bed",
                           "a person partially undressed"],
    "kissing_intimacy":   ["two people kissing passionately", "a couple making out"],
    "violence":           ["people fighting violently", "a person attacking another",
                           "someone pointing a gun at a person"],
    "gore_injury":        ["a bloody gory scene", "a graphic injury with blood",
                           "a mutilated body"],
}
NEUTRAL_PROMPTS = ["a normal everyday scene", "people talking", "a landscape",
                   "an empty room", "an ordinary photo"]

# Default per-category confidence thresholds (0..1). Tune on real footage.
DEFAULT_THRESHOLDS = {
    "nudity":             0.5,
    "sexual_content":     0.6,
    "bedroom_suggestive": 0.65,
    "kissing_intimacy":   0.6,
    "violence":           0.6,
    "gore_injury":        0.6,
}


# ── Device selection ────────────────────────────────────────────────────────

def resolve_device(pref: str = "auto") -> str:
    """
    Resolve a device preference to a concrete device string ('cuda' or 'cpu').
    'auto' uses the GPU when available; 'gpu' warns and falls back to CPU if
    none is found; 'cpu' forces CPU.
    """
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    gpu = False
    try:
        import torch
        gpu = bool(torch.cuda.is_available())
    except Exception:
        gpu = False
    if pref == "gpu" and not gpu:
        logger.warning("Content filter: GPU requested but none detected — using CPU")
    return "cuda" if gpu else "cpu"


# ── Pure decision logic (unit-tested) ───────────────────────────────────────

# Action precedence — a stronger action on the frame wins over a weaker one.
_ACTION_RANK = {"pass": 0, "blur_region": 1, "blur_frame": 2, "skip": 3}


def decide_frame_action(hits, categories_cfg, thresholds=None):
    """
    Collapse a frame's category hits into a single action, honoring the user's
    per-category severities and the fixed severity->action mapping.

    Returns (action, regions) where action is one of:
        "pass"        — nothing flagged
        "blur_region" — blur only `regions`  (mild, and box(es) available)
        "blur_frame"  — blur the whole frame (moderate, or mild with no box)
        "skip"        — skip/black-out the section (severe)

    The strongest triggered action wins. A 'mild' (blur_region) hit with no
    bounding box is upgraded to 'blur_frame' since we can't localize it.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    best = "pass"
    regions: list = []
    for hit in hits:
        severity = categories_cfg.get(hit.key, "off")
        action = SEVERITY_ACTIONS.get(severity)
        if action is None:            # off, or unknown severity
            continue
        if hit.score < thresholds.get(hit.key, 0.6):
            continue
        if action == "blur_region" and not hit.regions:
            action = "blur_frame"     # can't localize -> whole frame
        if action == "blur_region":
            regions.extend(hit.regions)
        if _ACTION_RANK[action] > _ACTION_RANK[best]:
            best = action
    return best, (regions if best == "blur_region" else [])


# ── Hybrid detector ─────────────────────────────────────────────────────────

class ContentDetector:
    """
    Lazy, defensively-loaded hybrid detector. Call load() once before detect().
    detect(frame_bgr) returns a list[CategoryHit]; an empty list means either
    nothing detected or no backend available.
    """

    def __init__(self, device: str = "auto", thresholds: dict | None = None):
        self.device = resolve_device(device)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._nude = None
        self._clip = None          # (model, preprocess, tokenizer-ish)
        self._clip_text = None     # precomputed text features + index ranges
        self._loaded = False

    @property
    def ready(self) -> bool:
        return self._loaded and (self._nude is not None or self._clip is not None)

    def load(self):
        self._load_nudenet()
        self._load_clip()
        self._loaded = True
        backends = [n for n, on in (("NudeNet", self._nude is not None),
                                    ("CLIP", self._clip is not None)) if on]
        if backends:
            logger.info(f"ContentDetector ready on {self.device} — backends: "
                        f"{', '.join(backends)}")
        else:
            logger.error("ContentDetector: no backends available — video content "
                         "filtering will be a no-op. Install nudenet and/or "
                         "open_clip_torch.")

    # — NudeNet —
    def _load_nudenet(self):
        try:
            from nudenet import NudeDetector
            self._nude = NudeDetector()
            logger.info("ContentDetector: NudeNet loaded")
        except Exception as e:
            # Full traceback at debug; a clear one-liner with the exception type
            # at warning, so a real failure (missing onnxruntime, weight
            # download, API change) is diagnosable rather than swallowed.
            logger.opt(exception=True).debug("NudeNet load traceback")
            logger.warning(f"ContentDetector: NudeNet unavailable: "
                           f"{type(e).__name__}: {e}")
            self._nude = None

    def _detect_nudenet(self, frame_bgr) -> list:
        if self._nude is None:
            return []
        try:
            dets = self._nude.detect(frame_bgr)
        except Exception as e:
            logger.debug(f"NudeNet detect error: {e}")
            return []
        hits = []
        for d in dets or []:
            cls = d.get("class") or d.get("label")
            cat = NUDENET_TO_CATEGORY.get(cls)
            if not cat:
                continue
            score = float(d.get("score", d.get("confidence", 0.0)) or 0.0)
            box = d.get("box") or d.get("bbox")
            regions = []
            if box and len(box) == 4:
                x, y, w, h = (int(v) for v in box)
                regions = [(x, y, w, h)]
            hits.append(CategoryHit(cat, score, regions))
        return hits

    # — CLIP —
    def _load_clip(self):
        try:
            import torch
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model = model.to(self.device).eval()

            # Precompute text features for every prompt (positives + neutral).
            prompts, ranges = [], {}
            for key, plist in CLIP_PROMPTS.items():
                start = len(prompts)
                prompts.extend(plist)
                ranges[key] = (start, len(prompts))
            neutral_start = len(prompts)
            prompts.extend(NEUTRAL_PROMPTS)
            neutral_range = (neutral_start, len(prompts))

            with torch.no_grad():
                toks = tokenizer(prompts).to(self.device)
                tfeat = model.encode_text(toks)
                tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

            self._clip = (model, preprocess)
            self._clip_text = (tfeat, ranges, neutral_range)
            logger.info("ContentDetector: CLIP (ViT-B-32) loaded")
        except Exception as e:
            logger.opt(exception=True).debug("CLIP load traceback")
            logger.warning(f"ContentDetector: CLIP unavailable: "
                           f"{type(e).__name__}: {e}")
            self._clip = None
            self._clip_text = None

    def _detect_clip(self, frame_bgr) -> list:
        if self._clip is None or self._clip_text is None:
            return []
        try:
            import torch
            import cv2
            from PIL import Image
            model, preprocess = self._clip
            tfeat, ranges, neutral_range = self._clip_text

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                ifeat = model.encode_image(img)
                ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
                # Cosine sims scaled like CLIP logits, then softmax over all prompts.
                logits = (100.0 * ifeat @ tfeat.T).softmax(dim=-1)[0]

            hits = []
            ns, ne = neutral_range
            for key, (s, e) in ranges.items():
                pos = float(logits[s:e].sum().item())
                hits.append(CategoryHit(key, pos, []))
            return hits
        except Exception as e:
            logger.debug(f"CLIP detect error: {e}")
            return []

    def detect(self, frame_bgr) -> list:
        """Run both backends on one BGR frame and return combined CategoryHits."""
        if not self._loaded:
            return []
        return self._detect_nudenet(frame_bgr) + self._detect_clip(frame_bgr)
