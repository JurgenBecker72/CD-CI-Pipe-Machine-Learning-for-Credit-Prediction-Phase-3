"""Sanity tests for project configuration.

These tests lock in the critical config invariants that, if broken,
would silently reintroduce the leakage / ID bugs we already fixed.
"""

from src import config


def test_target_is_bad():
    assert config.TARGET == "bad"


def test_random_state_is_set():
    # Reproducibility matters — this should never be None.
    assert isinstance(config.RANDOM_STATE, int)


def test_leakage_columns_contain_perf_window():
    # All three performance-window columns must be in the drop list,
    # otherwise the scorecard will leak outcome info into training.
    expected = {"num_accounts_perf", "highest_arrears_perf", "age_oldest_perf"}
    assert expected.issubset(set(config.LEAKAGE_COLUMNS))


def test_id_columns_exact_match_only():
    # ID_COLUMNS must be exact names, not substrings — otherwise
    # legitimate features like `num_accounts_assess` get dropped.
    assert "dummy_id" in config.ID_COLUMNS
    # Guard against the old substring bug:
    assert "num_accounts_assess" not in config.ID_COLUMNS
    assert "num_accounts_perf" not in config.ID_COLUMNS


def test_assess_features_not_in_leakage_list():
    # Assessment-window features are allowed in the model.
    # They must NOT accidentally be in LEAKAGE_COLUMNS.
    assert "num_accounts_assess" not in config.LEAKAGE_COLUMNS
    assert "highest_arrears_assess" not in config.LEAKAGE_COLUMNS
