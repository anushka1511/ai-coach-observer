"""Verification script for digression detection and dead-code cleanup."""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub heavy ML deps so orchestrator can be imported without torch installed.
_torch = MagicMock()
_torch.nn = MagicMock()
sys.modules["torch"] = _torch
sys.modules["torch.nn"] = _torch.nn
sys.modules["torchaudio"] = MagicMock()
sys.modules["transformers"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.schemas.data_models import AudioChunk
from backend.core.orchestrator import CoachingObserverSystem


def _chunk(speaker: str, text: str, ts: float) -> AudioChunk:
    return AudioChunk(
        timestamp=ts,
        duration=1.0,
        speaker=speaker,
        transcript=text,
        audio_data=b"",
        is_final=True,
    )


def test_digression():
    orch = CoachingObserverSystem(assemblyai_key="test", gemini_key="")

    # Build a coaching conversation about career goals
    history = [
        _chunk("coach", "What would you like to achieve in your career this year?", 1.0),
        _chunk("coachee", "I want to move into a leadership role within my department.", 2.0),
        _chunk("coach", "Tell me more about what leadership means to you.", 3.0),
        _chunk("coachee", "I think it means guiding the team and making strategic decisions.", 4.0),
    ]

    # Normal follow-up with different vocabulary — should NOT be flagged as digression
    on_topic = _chunk(
        "coachee",
        "My manager mentioned there might be an opening on the product team soon.",
        5.0,
    )
    score_on_topic = orch._detect_digression(on_topic, history + [on_topic])
    assert score_on_topic < 0.35, f"Expected low digression for related topic shift, got {score_on_topic}"

    # Short acknowledgment — should be zero
    short = _chunk("coachee", "Yes exactly.", 6.0)
    score_short = orch._detect_digression(short, history + [on_topic, short])
    assert score_short == 0.0, f"Short utterance should score 0, got {score_short}"

    # Explicit digression marker — should be high
    explicit = _chunk(
        "coachee",
        "By the way, I also wanted to ask about my vacation schedule next month.",
        7.0,
    )
    score_explicit = orch._detect_digression(explicit, history + [on_topic, short, explicit])
    assert score_explicit >= 0.60, f"Explicit digression should score high, got {score_explicit}"

    # Completely unrelated long statement — should be moderate/high
    unrelated = _chunk(
        "coachee",
        "The weather has been terrible lately and my car broke down on the highway yesterday morning.",
        8.0,
    )
    score_unrelated = orch._detect_digression(
        unrelated, history + [on_topic, short, explicit, unrelated]
    )
    assert score_unrelated >= 0.30, f"Unrelated topic should score moderately, got {score_unrelated}"

    print("PASS: digression detection")


def test_dead_code_removed():
    removed = [
        "backend.models.report_generator",
        "backend.models.sarcasm_detector",
        "backend.models.audio_capture",
        "backend.debug_audio",
    ]
    for mod in removed:
        try:
            importlib.import_module(mod)
            raise AssertionError(f"Module should have been removed: {mod}")
        except ModuleNotFoundError:
            pass

    from backend.models.gemini_analyzer import GeminiAnalyzer
    assert "analyze_real_time" not in GeminiAnalyzer.__dict__

    from backend.models.inference_engine import ModelInferenceEngine
    assert not hasattr(ModelInferenceEngine, "_run_digression_inference")

    from backend.models.storage import ChromaDBStorage
    storage = ChromaDBStorage.__dict__
    assert "store_session_chunk" not in storage
    assert "get_session_context" not in storage

    root = Path(__file__).resolve().parent
    for rel in (
        "backend/models/report_generator.py",
        "backend/models/sarcasm_detector.py",
        "backend/models/audio_capture.py",
        "backend/debug_audio.py",
    ):
        assert not (root / rel).exists(), f"File should have been deleted: {rel}"

    print("PASS: dead code removed")


def test_prior_fixes():
    """Spot-check that earlier fixes remain in place."""
    import asyncio

    from backend.models.enhanced_local_analyzer import EnhancedLocalAnalyzer
    from backend.models.gemini_analyzer import GeminiAnalyzer

    orch = CoachingObserverSystem(assemblyai_key="test", gemini_key="")
    chunk = _chunk("coachee", "Hmm.", 1.0)

    grow = asyncio.run(
        orch._analyze_grow_phase(chunk, type("I", (), {"emotion": {}})())
    )
    assert grow.phase == "Uncertain", f"Expected Uncertain, got {grow.phase}"

    assert orch._analyze_emotion_from_text("okay sure") is None
    assert "gemini-2.5-flash" in GeminiAnalyzer._MODEL_CANDIDATES

    analyzer = EnhancedLocalAnalyzer()
    assert analyzer._analyze_learning_styles({"vak_scores": []}) == {}

    print("PASS: prior fixes verified")


if __name__ == "__main__":
    test_dead_code_removed()
    test_digression()
    test_prior_fixes()
    print("\nAll verification checks passed.")
