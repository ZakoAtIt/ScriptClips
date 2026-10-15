"""Unit tests for pipeline gate functions.

These are cheap, data-independent tests that verify the core gate logic
without needing any video/search/transcript data. They are the primary
guard against someone silently loosening the gates during a refactor.

Run: python -m pytest tests/test_gates.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.scoring import passes_relevance_gate, combine_score
from modules.moment_finder import _corroborates, _reason_refs_action_or_object
from modules.clip_exporter import _pick_key


# ---------------------------------------------------------------------------
# passes_relevance_gate tests
# ---------------------------------------------------------------------------

class TestRelevanceGate:
    """Event sentences require entity AND (action OR object OR visual proof)."""

    def _make_candidate(self, **kw):
        base = {
            "entity_hits": 0, "action_hits": 0, "object_hits": 0,
            "event_visual_verified": False, "vlm_happening": False,
            "combined_score": 0.5, "semantic_score": 0.3, "rerank_score": 0.1,
            "llm_found": False, "llm_moment_score": 0.0,
        }
        base.update(kw)
        return base

    def _event_plan(self):
        return {"sentence_type": "event"}

    def _narration_plan(self):
        return {"sentence_type": "narration"}

    def test_event_entity_only_rejected(self):
        """Entity-only candidate fails gate for event sentences."""
        c = self._make_candidate(entity_hits=5, action_hits=0, object_hits=0)
        ok, reason = passes_relevance_gate(c, self._event_plan())
        assert not ok, f"Expected reject, got ok=True: {reason}"
        assert "action" in reason.lower() or "object" in reason.lower() or "visual" in reason.lower()

    def test_event_entity_plus_action_passed(self):
        """Entity + action passes gate."""
        c = self._make_candidate(entity_hits=1, action_hits=2)
        ok, reason = passes_relevance_gate(c, self._event_plan())
        assert ok, f"Expected pass, got ok=False: {reason}"

    def test_event_entity_plus_object_passed(self):
        """Entity + object passes gate."""
        c = self._make_candidate(entity_hits=1, object_hits=1)
        ok, reason = passes_relevance_gate(c, self._event_plan())
        assert ok, f"Expected pass, got ok=False: {reason}"

    def test_event_entity_plus_visual_proof_passed(self):
        """Entity + event_visual_verified passes gate."""
        c = self._make_candidate(entity_hits=1, event_visual_verified=True)
        ok, reason = passes_relevance_gate(c, self._event_plan())
        assert ok, f"Expected pass, got ok=False: {reason}"

    def test_event_entity_plus_vlm_passed(self):
        """Entity + vlm_happening passes gate."""
        c = self._make_candidate(entity_hits=1, vlm_happening=True)
        ok, reason = passes_relevance_gate(c, self._event_plan())
        assert ok, f"Expected pass, got ok=False: {reason}"

    def test_event_no_entity_rejected(self):
        """No entity at all fails gate even with action."""
        c = self._make_candidate(entity_hits=0, action_hits=3)
        ok, reason = passes_relevance_gate(c, self._event_plan())
        assert not ok

    def test_event_nothing_passes_rejected(self):
        """Zero everything fails gate."""
        c = self._make_candidate()
        ok, reason = passes_relevance_gate(c, self._event_plan())
        assert not ok

    def test_narration_entity_only_passed(self):
        """Narration sentence: entity-only is sufficient."""
        c = self._make_candidate(entity_hits=1, action_hits=0, object_hits=0)
        ok, reason = passes_relevance_gate(c, self._narration_plan())
        assert ok, f"Expected pass for narration, got ok=False: {reason}"

    def test_event_high_combined_but_no_action_rejected(self):
        """High combined_score cannot compensate for zero action/object/visual."""
        c = self._make_candidate(
            entity_hits=1, action_hits=0, object_hits=0,
            combined_score=0.95, semantic_score=0.9, rerank_score=0.8)
        ok, reason = passes_relevance_gate(c, self._event_plan())
        assert not ok, "High combined_score should NOT compensate for missing action"


# ---------------------------------------------------------------------------
# _corroborates tests
# ---------------------------------------------------------------------------

class TestCorroborates:
    """Event moments require action or object evidence, not just entity."""

    def _event_plan(self):
        return {
            "sentence_type": "event",
            "core_entities": ["Speed"],
            "action_terms": ["meet"],
            "object_terms": [],
            "visual_event": False,
        }

    def _visual_event_plan(self):
        return {
            "sentence_type": "event",
            "core_entities": ["Kai"],
            "action_terms": ["fly"],
            "object_terms": ["helicopter"],
            "visual_event": True,
        }

    def _narration_plan(self):
        return {
            "sentence_type": "narration",
            "core_entities": ["Speed"],
            "action_terms": [],
            "object_terms": [],
        }

    def test_event_entity_only_not_corroborated(self):
        """Event moment with only entity evidence is NOT corroborated."""
        evidence = "speed is streaming and talking to chat about his day"
        assert not _corroborates(evidence, self._event_plan(), refs_plan=False)

    def test_event_action_evidence_corroborated(self):
        """Event moment with action evidence IS corroborated."""
        evidence = "oh my god he just met ronaldo in person face to face"
        assert _corroborates(evidence, self._event_plan(), refs_plan=False)

    def test_event_refs_plan_with_action_corroborated(self):
        """Event moment whose reason references the plan action is corroborated."""
        evidence = "speed is streaming and talking to chat"
        assert _corroborates(evidence, self._event_plan(), refs_plan=True)

    def test_event_refs_plan_true_sufficient_corroboration(self):
        """Event moment with refs_plan=True is corroborated (caller verified action/object in reason)."""
        evidence = "speed is streaming and talking to chat about his day"
        assert _corroborates(evidence, self._event_plan(), refs_plan=True)

    def test_visual_event_object_evidence_corroborated(self):
        """Visual event moment with object evidence IS corroborated."""
        evidence = "there is a helicopter flying over the university campus"
        assert _corroborates(evidence, self._visual_event_plan(), refs_plan=False)

    def test_narration_entity_only_corroborated(self):
        """Narration moment with entity evidence IS corroborated."""
        evidence = "speed is streaming and talking to chat"
        assert _corroborates(evidence, self._narration_plan(), refs_plan=False)


# ---------------------------------------------------------------------------
# _pick_key tests
# ---------------------------------------------------------------------------

class TestPickKey:
    """Generic plans: action_coverage outranks llm_found."""

    def _generic_plan(self):
        return {"sentence_type": "narration"}

    def _concrete_visual_plan(self):
        return {
            "sentence_type": "event",
            "visual_event_kind": "helicopter",
            "object_terms": ["helicopter"],
        }

    def test_generic_favors_coverage_over_llm(self):
        """Higher action_coverage outranks higher llm_found in generic plans."""
        clip_a = {
            "action_coverage": 1.0, "entity_coverage": 0.8,
            "llm_found": False, "vlm_happening": False,
            "combined_score": 0.3, "anchor_sim": 0.4,
            "keyword_located": False, "llm_moment_score": 0.0,
            "face_all": False, "face_coverage": 0.0,
            "event_visual_score": 0.0, "vision_score": 0.0,
            "vlm_score": 0.0, "semantic_score": 0.5,
            "proposal_score": 0.4,
        }
        clip_b = {
            "action_coverage": 0.0, "entity_coverage": 0.5,
            "llm_found": True, "vlm_happening": False,
            "combined_score": 0.5, "anchor_sim": 0.3,
            "keyword_located": False, "llm_moment_score": 0.7,
            "face_all": False, "face_coverage": 0.0,
            "event_visual_score": 0.0, "vision_score": 0.0,
            "vlm_score": 0.0, "semantic_score": 0.4,
            "proposal_score": 0.3,
        }
        key_a = _pick_key(clip_a, self._generic_plan())
        key_b = _pick_key(clip_b, self._generic_plan())
        assert key_a > key_b, f"clip_a (coverage=1.0) should rank above clip_b (llm_found=True)"

    def test_concrete_visual_favors_vlm_over_coverage(self):
        """Concrete visual plans: vlm_happening outranks coverage."""
        clip_a = {
            "vlm_happening": True, "vlm_score": 0.9,
            "keyword_located": False, "llm_found": False,
            "llm_moment_score": 0.0, "face_all": False,
            "face_coverage": 0.0, "event_visual_score": 0.5,
            "vision_score": 0.3, "anchor_sim": 0.4,
            "combined_score": 0.3, "action_coverage": 0.0,
            "semantic_score": 0.0, "proposal_score": 0.0,
        }
        clip_b = {
            "vlm_happening": False, "vlm_score": 0.0,
            "keyword_located": False, "llm_found": True,
            "llm_moment_score": 0.8, "face_all": False,
            "face_coverage": 0.0, "event_visual_score": 0.0,
            "vision_score": 0.0, "anchor_sim": 0.3,
            "combined_score": 0.5, "action_coverage": 1.0,
            "semantic_score": 0.0, "proposal_score": 0.0,
        }
        key_a = _pick_key(clip_a, self._concrete_visual_plan())
        key_b = _pick_key(clip_b, self._concrete_visual_plan())
        assert key_a > key_b, "vlm_happening should outrank llm_found for visual plans"

    def test_keyword_located_beats_llm_found_generic(self):
        """Keyword-located outranks llm_found in generic plans."""
        clip_a = {
            "keyword_located": True, "llm_found": False,
            "llm_moment_score": 0.0, "vlm_happening": False,
            "vlm_score": 0.0, "face_all": False, "face_coverage": 0.0,
            "event_visual_score": 0.0, "vision_score": 0.0,
            "anchor_sim": 0.3, "combined_score": 0.4,
            "action_coverage": 0.5, "entity_coverage": 0.5,
            "semantic_score": 0.4, "proposal_score": 0.3,
        }
        clip_b = {
            "keyword_located": False, "llm_found": True,
            "llm_moment_score": 0.8, "vlm_happening": False,
            "vlm_score": 0.0, "face_all": False, "face_coverage": 0.0,
            "event_visual_score": 0.0, "vision_score": 0.0,
            "anchor_sim": 0.3, "combined_score": 0.4,
            "action_coverage": 0.5, "entity_coverage": 0.5,
            "semantic_score": 0.4, "proposal_score": 0.3,
        }
        key_a = _pick_key(clip_a, self._generic_plan())
        key_b = _pick_key(clip_b, self._generic_plan())
        assert key_a > key_b, "keyword_located should outrank llm_found"


# ---------------------------------------------------------------------------
# _reason_refs_action_or_object tests
# ---------------------------------------------------------------------------

class TestReasonRefsActionOrObject:
    """For event sentences, the reason must reference action/object, not just entity."""

    def _event_plan(self):
        return {
            "sentence_type": "event",
            "core_entities": ["Speed"],
            "action_terms": ["meet", "hug"],
            "object_terms": ["dog"],
            "visual_event": False,
        }

    def test_reason_with_action_term_accepted(self):
        moment = {"reason": "Speed meets Kai in the hallway"}
        assert _reason_refs_action_or_object(moment, self._event_plan())

    def test_reason_with_object_term_accepted(self):
        moment = {"reason": "Speed's dog runs across the screen"}
        assert _reason_refs_action_or_object(moment, self._event_plan())

    def test_reason_entity_only_rejected(self):
        moment = {"reason": "Speed is streaming and talking to chat"}
        assert not _reason_refs_action_or_object(moment, self._event_plan())

    def test_reason_empty_rejected(self):
        moment = {"reason": ""}
        assert not _reason_refs_action_or_object(moment, self._event_plan())
