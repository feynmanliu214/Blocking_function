"""Tests for scorer_requires_blocking_detection helper."""
import pytest

from forecast_analysis.scoring import (
    scorer_requires_blocking_detection,
    SCORER_REGISTRY,
)


class TestScorerRequiresBlockingDetection:
    def test_gridpoint_scorers_do_not_require(self):
        assert scorer_requires_blocking_detection("GridpointIntensityScorer") is False
        assert scorer_requires_blocking_detection("GridpointPersistenceScorer") is False

    def test_ano_scorer_requires(self):
        assert scorer_requires_blocking_detection("ANOScorer") is True

    def test_legacy_names_default_to_true(self):
        assert scorer_requires_blocking_detection("IntegratedScorer") is True
        assert scorer_requires_blocking_detection("DriftPenalizedScorer") is True

    def test_unknown_scorer_defaults_to_true(self):
        assert scorer_requires_blocking_detection("NonExistentScorer") is True

    def test_consistent_with_class_attribute(self):
        """All registry entries must match their class attribute."""
        for name, cls in SCORER_REGISTRY.items():
            assert scorer_requires_blocking_detection(name) == cls.requires_blocking_detection
