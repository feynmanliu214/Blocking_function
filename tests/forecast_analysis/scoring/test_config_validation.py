"""Tests for scorer validation helpers (name, region, variable)."""

import pytest

from forecast_analysis.scoring import (
    SCORER_REGISTRY,
    validate_scorer_name,
    scorer_requires_anomaly,
    scorer_required_variable,
    validate_scorer_region,
    validate_scorer_variable,
)


# ---------------------------------------------------------------------------
# validate_scorer_name
# ---------------------------------------------------------------------------

class TestValidateScorerName:
    def test_registry_names_accepted(self):
        for name in SCORER_REGISTRY:
            validate_scorer_name(name)  # should not raise

    def test_legacy_names_accepted(self):
        for name in ("IntegratedScorer", "DriftPenalizedScorer", "RMSEScorer"):
            validate_scorer_name(name)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer"):
            validate_scorer_name("FooBarScorer")

    def test_heatwave_mean_scorer_accepted(self):
        validate_scorer_name("HeatwaveMeanScorer")


# ---------------------------------------------------------------------------
# scorer_requires_anomaly
# ---------------------------------------------------------------------------

class TestScorerRequiresAnomaly:
    def test_blocking_scorers_require_anomaly(self):
        assert scorer_requires_anomaly("ANOScorer") is True
        assert scorer_requires_anomaly("GridpointPersistenceScorer") is True
        assert scorer_requires_anomaly("GridpointIntensityScorer") is True

    def test_heatwave_does_not_require_anomaly(self):
        assert scorer_requires_anomaly("HeatwaveMeanScorer") is False

    def test_legacy_names_default_true(self):
        assert scorer_requires_anomaly("IntegratedScorer") is True
        assert scorer_requires_anomaly("DriftPenalizedScorer") is True

    def test_unknown_defaults_true(self):
        assert scorer_requires_anomaly("NoSuchScorer") is True


# ---------------------------------------------------------------------------
# scorer_required_variable
# ---------------------------------------------------------------------------

class TestScorerRequiredVariable:
    def test_blocking_scorers_require_z500(self):
        assert scorer_required_variable("ANOScorer") == "z500"
        assert scorer_required_variable("GridpointPersistenceScorer") == "z500"
        assert scorer_required_variable("GridpointIntensityScorer") == "z500"

    def test_heatwave_requires_tas(self):
        assert scorer_required_variable("HeatwaveMeanScorer") == "tas"

    def test_legacy_names(self):
        assert scorer_required_variable("IntegratedScorer") == "z500"
        assert scorer_required_variable("DriftPenalizedScorer") == "z500"
        assert scorer_required_variable("RMSEScorer") == "z500"

    def test_unknown_defaults_z500(self):
        assert scorer_required_variable("UnknownScorer") == "z500"


# ---------------------------------------------------------------------------
# validate_scorer_region
# ---------------------------------------------------------------------------

class TestValidateScorerRegion:
    def test_allowed_region_passes(self):
        validate_scorer_region("ANOScorer", "NorthAtlantic")
        validate_scorer_region("HeatwaveMeanScorer", "Chicago")
        validate_scorer_region("HeatwaveMeanScorer", "France")

    def test_disallowed_region_raises(self):
        with pytest.raises(ValueError, match="not allowed"):
            validate_scorer_region("HeatwaveMeanScorer", "NorthAtlantic")

    def test_blocking_scorer_rejects_wrong_region(self):
        with pytest.raises(ValueError, match="not allowed"):
            validate_scorer_region("ANOScorer", "Chicago")

    def test_gridpoint_scorer_rejects_wrong_region(self):
        with pytest.raises(ValueError, match="not allowed"):
            validate_scorer_region("GridpointIntensityScorer", "France")

    def test_legacy_names_no_restriction(self):
        # Legacy names are not in registry, so no restriction enforced
        validate_scorer_region("IntegratedScorer", "Anywhere")

    def test_scorer_with_none_allowed_regions(self):
        """A scorer with allowed_regions = None should accept any region."""
        # BlockingScorer base has allowed_regions = None, but it's abstract.
        # We verify the logic: if a scorer had None, it would accept anything.
        # Currently all registered scorers have explicit allowed_regions,
        # so we test via a legacy name (not in registry).
        validate_scorer_region("DriftPenalizedScorer", "Antarctica")


# ---------------------------------------------------------------------------
# validate_scorer_variable
# ---------------------------------------------------------------------------

class TestValidateScorerVariable:
    def test_correct_variable_passes(self):
        validate_scorer_variable("ANOScorer", "z500")
        validate_scorer_variable("HeatwaveMeanScorer", "tas")

    def test_wrong_variable_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            validate_scorer_variable("HeatwaveMeanScorer", "z500")

    def test_blocking_scorer_wrong_variable(self):
        with pytest.raises(ValueError, match="does not match"):
            validate_scorer_variable("ANOScorer", "tas")

    def test_legacy_scorer_variable(self):
        validate_scorer_variable("IntegratedScorer", "z500")
        with pytest.raises(ValueError, match="does not match"):
            validate_scorer_variable("IntegratedScorer", "tas")

    def test_unknown_scorer_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer"):
            validate_scorer_variable("UnknownScorer", "z500")


# ---------------------------------------------------------------------------
# Consistency: class attributes match helper output
# ---------------------------------------------------------------------------

class TestConsistencyWithClassAttributes:
    def test_requires_anomaly_matches_class(self):
        for name, cls in SCORER_REGISTRY.items():
            assert scorer_requires_anomaly(name) == cls.requires_anomaly

    def test_required_variable_matches_class(self):
        for name, cls in SCORER_REGISTRY.items():
            assert scorer_required_variable(name) == cls.required_variable

    def test_allowed_regions_validated_for_all_registered(self):
        for name, cls in SCORER_REGISTRY.items():
            if cls.allowed_regions is not None:
                for region in cls.allowed_regions:
                    validate_scorer_region(name, region)  # should not raise
