"""
engine.py — Core Clinical Drug Safety Analysis Engine.

Pipeline:
  1. Normalize input
  2. Check cache
  3. Run LLM analysis
  4. Validate LLM output
  5. Apply allergy detection (class-level cross-reactivity)
  6. Apply condition contraindications
  7. Compute patient risk score
  8. Apply fallback if LLM failed
  9. Save to cache
  10. Return structured response

CRITICAL: This is healthcare-critical code. Every output is validated.
         No raw LLM text is ever exposed to the client.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

from models import (
    AllergySeverity,
    AllergyAlert,
    AnalysisSource,
    AnalyzeRequest,
    AnalyzeResponse,
    Confidence,
    ContraindicationAlert,
    DrugInteraction,
    RiskLevel,
    Severity,
)
from cache import InMemoryCache, create_cache

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"
FALLBACK_DATA_PATH = BASE_DIR / "data" / "fallback_interactions.json"


# ─── Fallback Data Loader ────────────────────────────────────────────────────

class FallbackDatabase:
    """Loads and indexes the fallback drug interaction dataset."""

    def __init__(self, path: Path = FALLBACK_DATA_PATH) -> None:
        self._data: dict = {}
        self._interaction_index: dict[tuple[str, str], dict] = {}
        self._allergy_map: dict[str, list[str]] = {}
        self._contraindication_rules: list[dict] = []
        self._load(path)

    def _load(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

            # Index interactions by sorted drug pair for O(1) lookup
            for interaction in self._data.get("interactions", []):
                pair = tuple(sorted([
                    interaction["drug_a"].lower(),
                    interaction["drug_b"].lower(),
                ]))
                self._interaction_index[pair] = interaction

            # Load allergy cross-reactivity map
            self._allergy_map = {
                k.lower(): [v.lower() for v in vs]
                for k, vs in self._data.get("allergy_cross_reactivity", {}).items()
            }

            # Load contraindication rules
            self._contraindication_rules = self._data.get("condition_contraindications", [])

            logger.info(
                "Fallback DB loaded: %d interactions, %d allergy classes, %d contraindication rules",
                len(self._interaction_index),
                len(self._allergy_map),
                len(self._contraindication_rules),
            )
        except Exception as e:
            logger.error("Failed to load fallback database: %s", e)

    def lookup_interaction(self, drug_a: str, drug_b: str) -> Optional[dict]:
        """Look up a specific drug pair interaction."""
        pair = tuple(sorted([drug_a.lower(), drug_b.lower()]))
        return self._interaction_index.get(pair)

    def lookup_interactions_for_drugs(self, drugs: list[str]) -> list[dict]:
        """Find all known interactions among a list of drugs."""
        results = []
        drug_set = {d.lower() for d in drugs}
        for pair, interaction in self._interaction_index.items():
            if pair[0] in drug_set and pair[1] in drug_set:
                results.append(interaction)
        return results

    def get_allergy_class(self, allergy: str) -> Optional[str]:
        """Find which drug class an allergy belongs to."""
        allergy_lower = allergy.lower()
        for drug_class, members in self._allergy_map.items():
            if allergy_lower == drug_class or allergy_lower in members:
                return drug_class
        return None

    def get_class_members(self, drug_class: str) -> list[str]:
        """Get all drugs in a given allergy class."""
        return self._allergy_map.get(drug_class.lower(), [])

    def check_allergy_cross_reactivity(
        self, medicine: str, allergies: list[str]
    ) -> Optional[tuple[str, str]]:
        """
        Check if a medicine triggers cross-reactivity with any allergy.
        Returns (allergy, drug_class) or None.
        """
        medicine_lower = medicine.lower()
        for allergy in allergies:
            allergy_lower = allergy.lower()
            # Direct match
            if medicine_lower == allergy_lower:
                return (allergy, allergy)
            # Class-level match
            allergy_class = self.get_allergy_class(allergy_lower)
            if allergy_class:
                members = self.get_class_members(allergy_class)
                if medicine_lower in members or medicine_lower == allergy_class:
                    return (allergy, allergy_class)
            # Check if the medicine belongs to a class the allergy is in
            for drug_class, members in self._allergy_map.items():
                if medicine_lower in members:
                    if allergy_lower == drug_class or allergy_lower in members:
                        return (allergy, drug_class)
        return None

    def check_contraindications(
        self, medicine: str, conditions: list[str]
    ) -> list[dict]:
        """Check if a medicine is contraindicated for any patient condition."""
        medicine_lower = medicine.lower()
        results = []
        for rule in self._contraindication_rules:
            condition = rule["condition"].lower()
            drugs = [d.lower() for d in rule["drugs"]]
            if medicine_lower in drugs:
                for patient_condition in conditions:
                    if condition in patient_condition.lower() or patient_condition.lower() in condition:
                        results.append(rule)
                        break
        return results


# ─── LLM Interface ───────────────────────────────────────────────────────────

class MedicalLLM:
    """
    Interface to a medical-specific LLM (BioMistral, Med42, Meditron, OpenBioLLM).
    Uses llama-cpp-python for GGUF models optimized for low VRAM (<2GB).
    Falls back gracefully if model is unavailable.
    """

    def __init__(self) -> None:
        self._model = None
        self._system_prompt = self._load_system_prompt()
        self._available = False
        self._model_name = os.getenv("MEDICAL_LLM_MODEL", "BioMistral-7B-GGUF")
        self._model_path = os.getenv("MEDICAL_LLM_PATH", "")
        self._n_ctx = int(os.getenv("LLM_CONTEXT_LENGTH", "2048"))
        self._n_gpu_layers = int(os.getenv("LLM_GPU_LAYERS", "0"))
        self._max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def _load_system_prompt(self) -> str:
        try:
            return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.error("Failed to load system prompt: %s", e)
            return "You are a clinical pharmacology expert. Output ONLY valid JSON."

    def initialize(self) -> bool:
        """
        Load the GGUF model via llama-cpp-python.
        Returns True if model loaded successfully.
        """
        if not self._model_path:
            logger.warning(
                "No MEDICAL_LLM_PATH set. LLM unavailable — fallback mode active."
            )
            return False

        try:
            from llama_cpp import Llama

            logger.info("Loading medical LLM from: %s", self._model_path)
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
                n_threads=int(os.getenv("LLM_THREADS", "4")),
                seed=42,
            )
            self._available = True
            logger.info("Medical LLM loaded: %s", self._model_name)
            return True
        except ImportError:
            logger.warning("llama-cpp-python not installed. LLM unavailable.")
            return False
        except Exception as e:
            logger.error("Failed to load medical LLM: %s", e)
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def analyze_interactions(self, medicines: list[str]) -> Optional[dict]:
        """
        Query the medical LLM for drug-drug interactions.
        Returns parsed JSON dict or None on failure.
        """
        if not self._available or self._model is None:
            return None

        prompt = self._build_prompt(medicines)

        try:
            response = self._model(
                prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stop=["```", "\n\n\n"],
                echo=False,
            )

            raw_text = response["choices"][0]["text"].strip()
            return self._parse_llm_output(raw_text)

        except Exception as e:
            logger.error("LLM inference failed: %s", e)
            return None

    def _build_prompt(self, medicines: list[str]) -> str:
        """Build the full prompt with system context and medicine list."""
        medicine_list = ", ".join(medicines)
        return (
            f"{self._system_prompt}\n"
            f"Medicines to analyze: [{medicine_list}]\n\n"
            f"Analyze all pairwise interactions. Respond with ONLY valid JSON:\n"
        )

    def _parse_llm_output(self, raw_text: str) -> Optional[dict]:
        """
        Attempt to extract valid JSON from LLM output.
        Handles common issues: markdown wrappers, trailing text, etc.
        """
        if not raw_text:
            return None

        # Strip markdown code block wrappers
        text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Attempt direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse LLM output as JSON")
        return None


# ─── Main Engine ──────────────────────────────────────────────────────────────

class DrugSafetyEngine:
    """
    Production-grade drug safety analysis engine.

    Orchestrates the full pipeline from input normalization through
    LLM analysis, validation, allergy detection, contraindication
    checking, risk scoring, and fallback handling.
    """

    def __init__(
        self,
        cache_backend: str = "memory",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: str | None = None,
        cache_ttl: int = 3600,
    ) -> None:
        self._cache = create_cache(
            backend=cache_backend,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password,
            ttl=cache_ttl,
        )
        self._fallback = FallbackDatabase()
        self._llm = MedicalLLM()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize engine components (call on startup)."""
        llm_ok = self._llm.initialize()
        if not llm_ok:
            logger.warning(
                "LLM not available. Engine will operate in fallback-only mode."
            )
        self._initialized = True
        logger.info("Drug Safety Engine initialized (LLM available: %s)", llm_ok)

    @property
    def is_llm_available(self) -> bool:
        return self._llm.is_available

    # ── Main Analysis Pipeline ────────────────────────────────────────────

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """
        Execute the full analysis pipeline.

        Steps:
          1. Normalize input (handled by Pydantic validators)
          2. Check cache
          3. Run LLM analysis
          4. Validate LLM output
          5. Apply allergy detection
          6. Apply contraindications
          7. Compute risk score
          8. Apply fallback if needed
          9. Save cache
          10. Return response
        """
        start_time = time.monotonic()

        all_drugs = list(set(
            request.medicines + request.patient_history.current_medications
        ))

        # ── Step 2: Check cache ───────────────────────────────────────────
        cache_key = self._cache.build_key(
            request.medicines,
            request.patient_history.current_medications,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            try:
                response = AnalyzeResponse(**cached)
                response.cache_hit = True
                response.processing_time_ms = round(
                    (time.monotonic() - start_time) * 1000, 2
                )
                logger.info("Cache hit for key: %s", cache_key[:16])
                return response
            except Exception:
                logger.warning("Invalid cache entry, recomputing")

        # ── Step 3–4: Run LLM analysis & validate ────────────────────────
        interactions: list[DrugInteraction] = []
        source = AnalysisSource.FALLBACK
        requires_review = False

        llm_result = self._llm.analyze_interactions(all_drugs)
        if llm_result is not None:
            validated = self._validate_llm_interactions(llm_result, all_drugs)
            if validated:
                interactions = validated
                source = AnalysisSource.LLM
                requires_review = llm_result.get("requires_doctor_review", False)

        # ── Step 8: Fallback if LLM failed or returned nothing ────────────
        if not interactions:
            interactions = self._fallback_interactions(all_drugs)
            source = AnalysisSource.FALLBACK
            if not interactions:
                # Even if no known interactions, that's a valid result
                logger.info("No interactions found (LLM + fallback)")

        # Deduplicate interactions
        interactions = self._deduplicate_interactions(interactions)

        # ── Step 5: Allergy detection ─────────────────────────────────────
        allergy_alerts = self._detect_allergies(
            request.medicines, request.patient_history.allergies
        )

        # ── Step 6: Contraindication checking ─────────────────────────────
        contraindication_alerts = self._check_contraindications(
            request.medicines, request.patient_history.conditions
        )

        # ── Step 7: Risk scoring ──────────────────────────────────────────
        risk_score = self._compute_risk_score(
            interactions, allergy_alerts, contraindication_alerts
        )

        # ── Determine safety decisions ────────────────────────────────────
        safe = self._determine_safe_to_prescribe(interactions, allergy_alerts)
        risk_level = self._determine_risk_level(risk_score)

        if not safe or any(a.severity == AllergySeverity.CRITICAL for a in allergy_alerts):
            requires_review = True

        # ── Build response ────────────────────────────────────────────────
        response = AnalyzeResponse(
            interactions=interactions,
            allergy_alerts=allergy_alerts,
            contraindication_alerts=contraindication_alerts,
            safe_to_prescribe=safe,
            overall_risk_level=risk_level,
            patient_risk_score=risk_score,
            requires_doctor_review=requires_review,
            source=source,
            cache_hit=False,
            processing_time_ms=0.0,
        )

        # ── Step 9: Save to cache ─────────────────────────────────────────
        try:
            self._cache.set(cache_key, response.model_dump(mode="json"))
        except Exception as e:
            logger.error("Cache save failed: %s", e)

        # ── Step 10: Final timing ─────────────────────────────────────────
        response.processing_time_ms = round(
            (time.monotonic() - start_time) * 1000, 2
        )

        return response

    # ── LLM Output Validation ─────────────────────────────────────────────

    def _validate_llm_interactions(
        self, llm_output: dict, all_drugs: list[str]
    ) -> list[DrugInteraction]:
        """
        Validate and sanitize LLM output into proper DrugInteraction objects.
        Rejects entries with missing/invalid fields or hallucinated drugs.
        """
        valid_interactions: list[DrugInteraction] = []
        drug_set = {d.lower() for d in all_drugs}

        raw_interactions = llm_output.get("interactions", [])
        if not isinstance(raw_interactions, list):
            logger.warning("LLM returned non-list interactions")
            return []

        for raw in raw_interactions:
            try:
                if not isinstance(raw, dict):
                    continue

                drug_a = str(raw.get("drug_a", "")).strip().lower()
                drug_b = str(raw.get("drug_b", "")).strip().lower()

                # Reject if drugs aren't in our input set (anti-hallucination)
                if drug_a not in drug_set or drug_b not in drug_set:
                    logger.warning(
                        "LLM hallucinated drugs: %s, %s (not in %s)",
                        drug_a, drug_b, drug_set,
                    )
                    continue

                # Reject self-interactions
                if drug_a == drug_b:
                    continue

                # Validate severity
                severity_raw = str(raw.get("severity", "")).lower()
                if severity_raw not in ("high", "medium", "low"):
                    severity_raw = "medium"  # Conservative default

                # Validate confidence
                confidence_raw = str(raw.get("source_confidence", "")).lower()
                if confidence_raw not in ("high", "medium", "low"):
                    confidence_raw = "low"

                mechanism = str(raw.get("mechanism", "")).strip()
                recommendation = str(raw.get("clinical_recommendation", "")).strip()

                if not mechanism:
                    mechanism = "Mechanism requires verification by a healthcare professional."
                if not recommendation:
                    recommendation = "Consult prescribing physician for clinical guidance."

                interaction = DrugInteraction(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    severity=Severity(severity_raw),
                    mechanism=mechanism,
                    clinical_recommendation=recommendation,
                    source_confidence=Confidence(confidence_raw),
                )
                valid_interactions.append(interaction)

            except Exception as e:
                logger.warning("Failed to validate LLM interaction entry: %s", e)
                continue

        return valid_interactions

    # ── Fallback Interactions ─────────────────────────────────────────────

    def _fallback_interactions(self, drugs: list[str]) -> list[DrugInteraction]:
        """Look up interactions from the fallback database."""
        results: list[DrugInteraction] = []
        raw = self._fallback.lookup_interactions_for_drugs(drugs)

        for entry in raw:
            try:
                interaction = DrugInteraction(
                    drug_a=entry["drug_a"],
                    drug_b=entry["drug_b"],
                    severity=Severity(entry["severity"]),
                    mechanism=entry["mechanism"],
                    clinical_recommendation=entry["clinical_recommendation"],
                    source_confidence=Confidence(entry["source_confidence"]),
                )
                results.append(interaction)
            except Exception as e:
                logger.warning("Invalid fallback entry: %s", e)

        return results

    # ── Allergy Detection ─────────────────────────────────────────────────

    def _detect_allergies(
        self, medicines: list[str], allergies: list[str]
    ) -> list[AllergyAlert]:
        """
        Detect drug allergies including class-level cross-reactivity.

        Example: Penicillin allergy → Amoxicillin → CRITICAL alert
        """
        if not allergies:
            return []

        alerts: list[AllergyAlert] = []
        seen: set[str] = set()

        for medicine in medicines:
            result = self._fallback.check_allergy_cross_reactivity(
                medicine, allergies
            )
            if result and medicine.lower() not in seen:
                allergy, drug_class = result
                seen.add(medicine.lower())
                alerts.append(AllergyAlert(
                    medicine=medicine,
                    reason=(
                        f"Patient has documented '{allergy}' allergy. "
                        f"'{medicine}' belongs to the '{drug_class}' class "
                        f"and poses cross-reactivity risk."
                    ),
                    severity=AllergySeverity.CRITICAL,
                ))

        return alerts

    # ── Contraindication Checking ─────────────────────────────────────────

    def _check_contraindications(
        self, medicines: list[str], conditions: list[str]
    ) -> list[ContraindicationAlert]:
        """Check medicines against patient conditions for contraindications."""
        if not conditions:
            return []

        alerts: list[ContraindicationAlert] = []
        seen: set[tuple[str, str]] = set()

        for medicine in medicines:
            rules = self._fallback.check_contraindications(medicine, conditions)
            for rule in rules:
                key = (medicine.lower(), rule["condition"].lower())
                if key not in seen:
                    seen.add(key)
                    alerts.append(ContraindicationAlert(
                        medicine=medicine,
                        condition=rule["condition"],
                        risk_level=Severity(rule["risk_level"]),
                        recommendation=rule["recommendation"],
                    ))

        return alerts

    # ── Risk Scoring ──────────────────────────────────────────────────────

    def _compute_risk_score(
        self,
        interactions: list[DrugInteraction],
        allergy_alerts: list[AllergyAlert],
        contraindications: list[ContraindicationAlert],
    ) -> int:
        """
        Compute patient risk score (0–100).

        Scoring:
          - High interaction: +30
          - Medium interaction: +15
          - Low interaction: +5
          - Critical allergy: +40
          - Warning allergy: +15
          - Contraindication: +25
        """
        score = 0

        for interaction in interactions:
            if interaction.severity == Severity.HIGH:
                score += 30
            elif interaction.severity == Severity.MEDIUM:
                score += 15
            else:
                score += 5

        for alert in allergy_alerts:
            if alert.severity == AllergySeverity.CRITICAL:
                score += 40
            else:
                score += 15

        for _ in contraindications:
            score += 25

        # Normalize to max 100
        return min(score, 100)

    # ── Safety Decisions ──────────────────────────────────────────────────

    @staticmethod
    def _determine_safe_to_prescribe(
        interactions: list[DrugInteraction],
        allergy_alerts: list[AllergyAlert],
    ) -> bool:
        """
        NOT safe if:
          - Any HIGH severity interaction
          - Any CRITICAL allergy alert
        """
        if any(i.severity == Severity.HIGH for i in interactions):
            return False
        if any(a.severity == AllergySeverity.CRITICAL for a in allergy_alerts):
            return False
        return True

    @staticmethod
    def _determine_risk_level(risk_score: int) -> RiskLevel:
        """
        0–30 → low
        31–70 → medium
        71–100 → high
        """
        if risk_score <= 30:
            return RiskLevel.LOW
        elif risk_score <= 70:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    # ── Deduplication ─────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_interactions(
        interactions: list[DrugInteraction],
    ) -> list[DrugInteraction]:
        """Remove duplicate interactions (same pair, keep highest severity)."""
        best: dict[tuple[str, str], DrugInteraction] = {}
        severity_order = {Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}

        for interaction in interactions:
            pair = tuple(sorted([interaction.drug_a, interaction.drug_b]))
            existing = best.get(pair)
            if existing is None or severity_order.get(
                interaction.severity, 0
            ) > severity_order.get(existing.severity, 0):
                best[pair] = interaction

        return list(best.values())

    # ── Cache Stats ───────────────────────────────────────────────────────

    @property
    def cache_stats(self) -> dict:
        if hasattr(self._cache, "stats"):
            return self._cache.stats
        return {}
