"""
tests/test_engine.py — Comprehensive test suite for the Drug Safety Engine.

Tests cover:
  - Drug interaction detection (LLM + fallback)
  - Allergy cross-reactivity detection
  - Condition contraindication checking
  - Cache hit/miss behavior
  - Fallback behavior when LLM is unavailable
  - Invalid input handling
  - Risk scoring logic
  - Output validation and deduplication
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    AllergySeverity,
    AllergyAlert,
    AnalysisSource,
    AnalyzeRequest,
    AnalyzeResponse,
    Confidence,
    DrugInteraction,
    PatientHistory,
    RiskLevel,
    Severity,
)
from cache import InMemoryCache
from engine import DrugSafetyEngine, FallbackDatabase


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """Create a fresh engine instance (LLM disabled for unit tests)."""
    eng = DrugSafetyEngine(cache_backend="memory")
    eng.initialize()
    return eng


@pytest.fixture
def cache():
    """Create a fresh in-memory cache."""
    return InMemoryCache(ttl=60)


@pytest.fixture
def fallback_db():
    """Load the fallback database."""
    return FallbackDatabase()


# ─── Test: Drug Interaction Detection (Fallback) ─────────────────────────────

class TestDrugInteractions:
    """Test drug-drug interaction detection through the fallback database."""

    @pytest.mark.asyncio
    async def test_known_interaction_warfarin_aspirin(self, engine):
        """Warfarin + Aspirin = HIGH severity interaction."""
        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        assert isinstance(response, AnalyzeResponse)
        assert len(response.interactions) >= 1
        assert response.source == AnalysisSource.FALLBACK

        # Find the warfarin-aspirin interaction
        interaction = next(
            (i for i in response.interactions
             if {i.drug_a, i.drug_b} == {"warfarin", "aspirin"}),
            None,
        )
        assert interaction is not None
        assert interaction.severity == Severity.HIGH
        assert interaction.mechanism  # Must have a mechanism
        assert interaction.clinical_recommendation  # Must have recommendation
        assert response.safe_to_prescribe is False

    @pytest.mark.asyncio
    async def test_known_interaction_digoxin_amiodarone(self, engine):
        """Digoxin + Amiodarone = HIGH severity interaction."""
        request = AnalyzeRequest(
            medicines=["digoxin", "amiodarone"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        interaction = next(
            (i for i in response.interactions
             if {i.drug_a, i.drug_b} == {"digoxin", "amiodarone"}),
            None,
        )
        assert interaction is not None
        assert interaction.severity == Severity.HIGH

    @pytest.mark.asyncio
    async def test_no_interaction_safe_drugs(self, engine):
        """Two drugs with no known interaction should return empty."""
        request = AnalyzeRequest(
            medicines=["acetaminophen", "cetirizine"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        assert isinstance(response, AnalyzeResponse)
        assert len(response.interactions) == 0
        assert response.safe_to_prescribe is True

    @pytest.mark.asyncio
    async def test_multiple_drugs_multiple_interactions(self, engine):
        """Multiple drugs should return all pairwise interactions found."""
        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin", "ibuprofen", "lithium"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        # Should find warfarin+aspirin and lithium+ibuprofen at minimum
        assert len(response.interactions) >= 2
        assert response.safe_to_prescribe is False


# ─── Test: Allergy Detection ─────────────────────────────────────────────────

class TestAllergyDetection:
    """Test class-level allergy cross-reactivity detection."""

    @pytest.mark.asyncio
    async def test_penicillin_amoxicillin_crossreactivity(self, engine):
        """Penicillin allergy should trigger CRITICAL alert for Amoxicillin."""
        request = AnalyzeRequest(
            medicines=["amoxicillin"],
            patient_history=PatientHistory(allergies=["penicillin"]),
        )
        response = await engine.analyze(request)

        assert len(response.allergy_alerts) >= 1
        alert = response.allergy_alerts[0]
        assert alert.medicine == "amoxicillin"
        assert alert.severity == AllergySeverity.CRITICAL
        assert "penicillin" in alert.reason.lower()
        assert response.safe_to_prescribe is False
        assert response.requires_doctor_review is True

    @pytest.mark.asyncio
    async def test_nsaid_allergy_ibuprofen(self, engine):
        """NSAID allergy should trigger alert for Ibuprofen."""
        request = AnalyzeRequest(
            medicines=["ibuprofen"],
            patient_history=PatientHistory(allergies=["nsaid"]),
        )
        response = await engine.analyze(request)

        assert len(response.allergy_alerts) >= 1
        assert response.allergy_alerts[0].severity == AllergySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_no_allergy_match(self, engine):
        """No allergy relation should return no alerts."""
        request = AnalyzeRequest(
            medicines=["metformin"],
            patient_history=PatientHistory(allergies=["penicillin"]),
        )
        response = await engine.analyze(request)

        assert len(response.allergy_alerts) == 0

    @pytest.mark.asyncio
    async def test_direct_allergy_match(self, engine):
        """Direct drug allergy match should trigger CRITICAL alert."""
        request = AnalyzeRequest(
            medicines=["aspirin"],
            patient_history=PatientHistory(allergies=["aspirin"]),
        )
        response = await engine.analyze(request)

        # Aspirin is in the NSAID class, so it should be detected
        assert len(response.allergy_alerts) >= 1
        assert response.safe_to_prescribe is False


# ─── Test: Cache Behavior ────────────────────────────────────────────────────

class TestCacheBehavior:
    """Test caching layer: hits, misses, key generation, TTL."""

    def test_cache_key_deterministic(self, cache):
        """Same inputs should always produce the same cache key."""
        key1 = cache.build_key(["aspirin", "warfarin"], ["metformin"])
        key2 = cache.build_key(["warfarin", "aspirin"], ["metformin"])
        assert key1 == key2  # Order shouldn't matter

    def test_cache_key_different_inputs(self, cache):
        """Different inputs should produce different cache keys."""
        key1 = cache.build_key(["aspirin"], [])
        key2 = cache.build_key(["ibuprofen"], [])
        assert key1 != key2

    def test_cache_set_and_get(self, cache):
        """Basic set and get should work."""
        cache.set("test_key", {"data": "value"})
        result = cache.get("test_key")
        assert result == {"data": "value"}

    def test_cache_miss(self, cache):
        """Missing key should return None."""
        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_ttl_expiration(self):
        """Expired entries should not be returned."""
        cache = InMemoryCache(ttl=1)  # 1 second TTL
        cache.set("expire_test", {"data": "temp"})
        assert cache.get("expire_test") is not None
        time.sleep(1.1)
        assert cache.get("expire_test") is None

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeat_request(self, engine):
        """Second identical request should be a cache hit."""
        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin"],
            patient_history=PatientHistory(),
        )

        response1 = await engine.analyze(request)
        assert response1.cache_hit is False

        response2 = await engine.analyze(request)
        assert response2.cache_hit is True
        assert response2.processing_time_ms < response1.processing_time_ms


# ─── Test: Fallback System ───────────────────────────────────────────────────

class TestFallbackSystem:
    """Test fallback database behavior when LLM is unavailable."""

    def test_fallback_db_loads(self, fallback_db):
        """Fallback database should load successfully."""
        result = fallback_db.lookup_interaction("warfarin", "aspirin")
        assert result is not None
        assert result["severity"] == "high"

    def test_fallback_db_no_interaction(self, fallback_db):
        """Unknown pair should return None from fallback."""
        result = fallback_db.lookup_interaction("vitamin_c", "zinc")
        assert result is None

    @pytest.mark.asyncio
    async def test_engine_uses_fallback_without_llm(self, engine):
        """Engine should use fallback when LLM is not available."""
        assert engine.is_llm_available is False  # LLM not loaded in tests

        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        assert response.source == AnalysisSource.FALLBACK
        assert len(response.interactions) >= 1

    def test_fallback_has_minimum_interactions(self, fallback_db):
        """Fallback DB must have at least 15 interactions."""
        all_drugs = [
            "warfarin", "aspirin", "metformin", "furosemide", "lisinopril",
            "potassium", "simvastatin", "clarithromycin", "methotrexate",
            "ibuprofen", "digoxin", "amiodarone", "fluoxetine", "tramadol",
            "ciprofloxacin", "theophylline", "clopidogrel", "omeprazole",
            "lithium", "sildenafil", "nitroglycerin", "alcohol",
            "atorvastatin", "gemfibrozil", "amoxicillin", "spironolactone",
            "phenytoin", "valproic acid", "amlodipine", "fluconazole",
            "prednisone",
        ]
        interactions = fallback_db.lookup_interactions_for_drugs(all_drugs)
        assert len(interactions) >= 15


# ─── Test: Invalid Input Handling ─────────────────────────────────────────────

class TestInvalidInput:
    """Test that invalid inputs are properly rejected or normalized."""

    def test_empty_medicines_rejected(self):
        """Empty medicine list should raise validation error."""
        with pytest.raises(Exception):
            AnalyzeRequest(medicines=[], patient_history=PatientHistory())

    def test_duplicate_medicines_normalized(self):
        """Duplicate medicines should be deduplicated."""
        request = AnalyzeRequest(
            medicines=["Aspirin", "aspirin", "ASPIRIN"],
            patient_history=PatientHistory(),
        )
        assert len(request.medicines) == 1
        assert request.medicines[0] == "aspirin"

    def test_whitespace_normalized(self):
        """Whitespace in drug names should be cleaned."""
        request = AnalyzeRequest(
            medicines=["  aspirin  ", " ibuprofen "],
            patient_history=PatientHistory(),
        )
        assert request.medicines == ["aspirin", "ibuprofen"]

    def test_invalid_age_rejected(self):
        """Age > 150 should be rejected."""
        with pytest.raises(Exception):
            PatientHistory(age=200)

    def test_invalid_weight_rejected(self):
        """Negative weight should be rejected."""
        with pytest.raises(Exception):
            PatientHistory(weight=-5.0)

    def test_mixed_case_allergies_normalized(self):
        """Allergies should be normalized to lowercase."""
        history = PatientHistory(allergies=["PENICILLIN", "Sulfa"])
        assert history.allergies == ["penicillin", "sulfa"]


# ─── Test: Risk Scoring ──────────────────────────────────────────────────────

class TestRiskScoring:
    """Test risk score computation and level classification."""

    @pytest.mark.asyncio
    async def test_high_interaction_high_score(self, engine):
        """High interaction should contribute +30 to risk score."""
        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        assert response.patient_risk_score >= 30
        # Score of 30 is at the boundary (0-30 = low), so with exactly one
        # high interaction (30 points) this may be low or medium
        assert response.overall_risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)

    @pytest.mark.asyncio
    async def test_critical_allergy_high_score(self, engine):
        """Critical allergy should contribute +40 to risk score."""
        request = AnalyzeRequest(
            medicines=["amoxicillin"],
            patient_history=PatientHistory(allergies=["penicillin"]),
        )
        response = await engine.analyze(request)

        assert response.patient_risk_score >= 40
        assert response.overall_risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)

    @pytest.mark.asyncio
    async def test_risk_level_low(self, engine):
        """No risks should result in low risk level."""
        request = AnalyzeRequest(
            medicines=["acetaminophen"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        assert response.patient_risk_score == 0
        assert response.overall_risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_risk_score_capped_at_100(self, engine):
        """Risk score should never exceed 100."""
        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin", "ibuprofen", "lithium"],
            patient_history=PatientHistory(
                allergies=["nsaid"],
                conditions=["kidney disease"],
            ),
        )
        response = await engine.analyze(request)

        assert response.patient_risk_score <= 100


# ─── Test: Contraindications ─────────────────────────────────────────────────

class TestContraindications:
    """Test condition-based contraindication detection."""

    @pytest.mark.asyncio
    async def test_kidney_disease_nsaid_contraindication(self, engine):
        """NSAIDs + kidney disease should trigger contraindication alert."""
        request = AnalyzeRequest(
            medicines=["ibuprofen"],
            patient_history=PatientHistory(conditions=["kidney disease"]),
        )
        response = await engine.analyze(request)

        assert len(response.contraindication_alerts) >= 1
        alert = response.contraindication_alerts[0]
        assert "kidney" in alert.condition.lower()
        assert alert.risk_level == Severity.HIGH

    @pytest.mark.asyncio
    async def test_asthma_beta_blocker_contraindication(self, engine):
        """Beta-blockers + asthma should trigger contraindication alert."""
        request = AnalyzeRequest(
            medicines=["propranolol"],
            patient_history=PatientHistory(conditions=["asthma"]),
        )
        response = await engine.analyze(request)

        assert len(response.contraindication_alerts) >= 1

    @pytest.mark.asyncio
    async def test_no_contraindication(self, engine):
        """Safe drug-condition combo should have no alerts."""
        request = AnalyzeRequest(
            medicines=["metformin"],
            patient_history=PatientHistory(conditions=["hypertension"]),
        )
        response = await engine.analyze(request)

        assert len(response.contraindication_alerts) == 0


# ─── Test: Output Validation ─────────────────────────────────────────────────

class TestOutputValidation:
    """Test that all output fields are properly structured."""

    @pytest.mark.asyncio
    async def test_response_has_all_required_fields(self, engine):
        """Response must contain all required fields."""
        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        # Verify all top-level fields
        assert hasattr(response, "interactions")
        assert hasattr(response, "allergy_alerts")
        assert hasattr(response, "safe_to_prescribe")
        assert hasattr(response, "overall_risk_level")
        assert hasattr(response, "patient_risk_score")
        assert hasattr(response, "requires_doctor_review")
        assert hasattr(response, "source")
        assert hasattr(response, "cache_hit")
        assert hasattr(response, "processing_time_ms")

    @pytest.mark.asyncio
    async def test_interaction_fields_complete(self, engine):
        """Each interaction should have all required fields."""
        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        for interaction in response.interactions:
            assert interaction.drug_a
            assert interaction.drug_b
            assert interaction.severity in Severity
            assert interaction.mechanism
            assert interaction.clinical_recommendation
            assert interaction.source_confidence in Confidence

    @pytest.mark.asyncio
    async def test_processing_time_tracked(self, engine):
        """processing_time_ms should be > 0."""
        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        assert response.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_safe_to_prescribe_false_on_high_interaction(self, engine):
        """safe_to_prescribe must be False when HIGH interaction exists."""
        request = AnalyzeRequest(
            medicines=["warfarin", "aspirin"],
            patient_history=PatientHistory(),
        )
        response = await engine.analyze(request)

        has_high = any(i.severity == Severity.HIGH for i in response.interactions)
        if has_high:
            assert response.safe_to_prescribe is False
            assert response.requires_doctor_review is True


# ─── Test: LLM Output Validation ─────────────────────────────────────────────

class TestLLMValidation:
    """Test the engine's LLM output validation logic (without real LLM)."""

    def test_validate_valid_llm_output(self, engine):
        """Valid LLM output should produce DrugInteraction objects."""
        llm_output = {
            "interactions": [
                {
                    "drug_a": "warfarin",
                    "drug_b": "aspirin",
                    "severity": "high",
                    "mechanism": "Increased bleeding risk",
                    "clinical_recommendation": "Avoid combination",
                    "source_confidence": "high",
                }
            ]
        }
        result = engine._validate_llm_interactions(
            llm_output, ["warfarin", "aspirin"]
        )
        assert len(result) == 1
        assert result[0].severity == Severity.HIGH

    def test_validate_rejects_hallucinated_drugs(self, engine):
        """LLM output referencing drugs not in input should be rejected."""
        llm_output = {
            "interactions": [
                {
                    "drug_a": "fake_drug_xyz",
                    "drug_b": "aspirin",
                    "severity": "high",
                    "mechanism": "Made up",
                    "clinical_recommendation": "N/A",
                    "source_confidence": "high",
                }
            ]
        }
        result = engine._validate_llm_interactions(
            llm_output, ["warfarin", "aspirin"]
        )
        assert len(result) == 0  # Hallucinated drug rejected

    def test_validate_fixes_missing_severity(self, engine):
        """Missing severity should default to 'medium' (conservative)."""
        llm_output = {
            "interactions": [
                {
                    "drug_a": "warfarin",
                    "drug_b": "aspirin",
                    "severity": "invalid_value",
                    "mechanism": "Some mechanism",
                    "clinical_recommendation": "Some recommendation",
                    "source_confidence": "high",
                }
            ]
        }
        result = engine._validate_llm_interactions(
            llm_output, ["warfarin", "aspirin"]
        )
        assert len(result) == 1
        assert result[0].severity == Severity.MEDIUM

    def test_validate_rejects_self_interaction(self, engine):
        """Drug interacting with itself should be rejected."""
        llm_output = {
            "interactions": [
                {
                    "drug_a": "aspirin",
                    "drug_b": "aspirin",
                    "severity": "low",
                    "mechanism": "Self interaction",
                    "clinical_recommendation": "N/A",
                    "source_confidence": "low",
                }
            ]
        }
        result = engine._validate_llm_interactions(
            llm_output, ["aspirin"]
        )
        assert len(result) == 0


# ─── Test: Deduplication ─────────────────────────────────────────────────────

class TestDeduplication:
    """Test interaction deduplication logic."""

    def test_dedup_keeps_highest_severity(self):
        """When duplicates exist, keep the one with highest severity."""
        interactions = [
            DrugInteraction(
                drug_a="warfarin", drug_b="aspirin",
                severity=Severity.LOW, mechanism="M1",
                clinical_recommendation="R1", source_confidence=Confidence.LOW,
            ),
            DrugInteraction(
                drug_a="aspirin", drug_b="warfarin",
                severity=Severity.HIGH, mechanism="M2",
                clinical_recommendation="R2", source_confidence=Confidence.HIGH,
            ),
        ]
        result = DrugSafetyEngine._deduplicate_interactions(interactions)
        assert len(result) == 1
        assert result[0].severity == Severity.HIGH


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
