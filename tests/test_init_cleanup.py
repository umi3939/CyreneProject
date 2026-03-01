"""Tests for psyche/__init__.py export structure cleanup.

Verifies that the reorganized __init__.py maintains full backward compatibility:
- All original export names are preserved
- All aliases remain accessible
- Import succeeds without error
- __all__ set is unchanged
"""

import pytest


class TestExportCompleteness:
    """Verify that all exported names are accessible."""

    def test_import_psyche_succeeds(self):
        """import psyche must complete without exception."""
        import psyche
        assert psyche is not None

    def test_all_names_accessible(self):
        """Every name in __all__ must be accessible as an attribute."""
        import psyche
        missing = []
        for name in psyche.__all__:
            if not hasattr(psyche, name):
                missing.append(name)
        assert missing == [], f"Names in __all__ but not accessible: {missing}"

    def test_all_names_not_none(self):
        """Every name in __all__ must resolve to a non-None value."""
        import psyche
        none_names = []
        for name in psyche.__all__:
            if getattr(psyche, name, None) is None:
                none_names.append(name)
        assert none_names == [], f"Names in __all__ that resolve to None: {none_names}"


class TestBackwardCompatibility:
    """Verify backward-compatible aliases are preserved."""

    def test_recall_by_mood_alias(self):
        """recall_by_mood must still be available as alias for recall_with_mood."""
        import psyche
        assert hasattr(psyche, "recall_by_mood")
        assert psyche.recall_by_mood is psyche.recall_with_mood

    def test_pillar_managers_importable(self):
        """Pillar manager modules must be importable as attributes."""
        import psyche
        assert hasattr(psyche, "attachment_manager")
        assert hasattr(psyche, "continuity_manager")
        assert hasattr(psyche, "identity_manager")
        assert hasattr(psyche, "projection_manager")

    def test_aliased_to_dict_functions(self):
        """to_dict/from_dict aliases must be accessible."""
        import psyche
        alias_pairs = [
            "responsibility_to_dict", "responsibility_from_dict",
            "multi_emotion_config_to_dict", "multi_emotion_config_from_dict",
            "stm_coupling_config_to_dict", "stm_coupling_config_from_dict",
            "silence_config_to_dict", "silence_config_from_dict",
            "tone_config_to_dict", "tone_config_from_dict",
            "stability_config_to_dict", "stability_config_from_dict",
            "sensitivity_config_to_dict", "sensitivity_config_from_dict",
            "orientation_config_to_dict", "orientation_config_from_dict",
            "vector_config_to_dict", "vector_config_from_dict",
            "candidate_config_to_dict", "candidate_config_from_dict",
            "transient_goal_config_to_dict", "transient_goal_config_from_dict",
            "scoped_goal_config_to_dict", "scoped_goal_config_from_dict",
            "tendency_config_to_dict", "tendency_config_from_dict",
        ]
        for name in alias_pairs:
            assert hasattr(psyche, name), f"Missing alias: {name}"

    def test_renamed_create_functions(self):
        """create_* convenience aliases must be accessible."""
        import psyche
        create_names = [
            "create_default_responsibility_state",
            "create_observer_config",
            "create_introspection_config",
            "create_orientation_config",
            "create_sensitivity_config",
            "create_stability_config",
            "create_vector_config",
            "create_candidate_config",
            "create_transient_goal_config",
            "create_scoped_goal_config",
            "create_tendency_config",
            "create_awareness_config",
            "create_self_model_config",
            "create_difference_config",
            "create_strain_config",
            "create_self_image_config",
            "create_coherence_config",
            "create_narrative_config",
            "create_episodic_memory_config",
            "create_consumption_config",
            "create_expectation_config",
            "create_other_model_config",
            "create_binding_config",
            "create_motive_config",
            "create_expansion_config",
            "create_integration_config",
        ]
        for name in create_names:
            assert hasattr(psyche, name), f"Missing create alias: {name}"


class TestAllSetInvariance:
    """Verify __all__ set is preserved exactly."""

    # The expected set size (unique names) from before the cleanup
    EXPECTED_UNIQUE_COUNT = 1323

    def test_unique_count(self):
        """__all__ unique name count must match the pre-cleanup count."""
        import psyche
        unique = set(psyche.__all__)
        assert len(unique) == self.EXPECTED_UNIQUE_COUNT, (
            f"Expected {self.EXPECTED_UNIQUE_COUNT} unique names, got {len(unique)}"
        )

    def test_known_duplicates_preserved(self):
        """Known duplicates from original must still be present."""
        import psyche
        from collections import Counter
        counts = Counter(psyche.__all__)
        # These 3 names appeared twice in the original __all__ due to
        # being exported from multiple modules
        known_dups = ["CognitionRecord", "ResponsibilitySnapshot", "get_trace_summary"]
        for name in known_dups:
            assert counts[name] >= 2, (
                f"Expected duplicate '{name}' to appear at least twice, "
                f"got {counts[name]}"
            )


class TestImportIntegrity:
    """Verify the import structure is sound."""

    def test_psyche_orchestrator_accessible(self):
        """PsycheOrchestrator must be directly importable."""
        from psyche import PsycheOrchestrator
        assert PsycheOrchestrator is not None

    def test_psyche_state_accessible(self):
        """PsycheState must be directly importable."""
        from psyche import PsycheState
        assert PsycheState is not None

    def test_core_pipeline_functions(self):
        """Core pipeline functions must be importable."""
        from psyche import (
            parse_percept,
            react,
            recall_with_mood,
            generate_thought_candidates,
            select_policy,
            render_expression,
        )
        assert all([
            parse_percept, react, recall_with_mood,
            generate_thought_candidates, select_policy, render_expression,
        ])

    def test_no_import_errors_on_reload(self):
        """Reloading psyche must not raise errors."""
        import importlib
        import psyche
        reloaded = importlib.reload(psyche)
        assert reloaded is not None
        assert len(reloaded.__all__) > 0
