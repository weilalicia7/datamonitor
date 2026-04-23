"""
Tests for config.py — Wave 3.7.4 (T3 coverage).

config.py defines paths, weights, sites, and the get_logger() factory.
The tests assert invariants that downstream code depends on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import config


class TestPaths:
    def test_base_dir_exists(self):
        assert config.BASE_DIR.exists()
        assert config.BASE_DIR.is_dir()

    def test_required_subdirs_created(self):
        # Module init creates these on import.
        for d in (config.MODELS_DIR, config.OUTPUT_DIR,
                  config.SCHEDULES_DIR, config.REPORTS_DIR,
                  config.DATA_CACHE_DIR):
            assert d.exists() and d.is_dir()


class TestOptimizationWeights:
    def test_default_weights_sum_to_one(self):
        s = sum(config.OPTIMIZATION_WEIGHTS.values())
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_pareto_weight_sets_each_sum_to_one(self):
        for cfg in config.PARETO_WEIGHT_SETS:
            numeric = {k: v for k, v in cfg.items() if k != "name"}
            s = sum(numeric.values())
            assert s == pytest.approx(1.0, abs=1e-6), \
                f"weight set {cfg['name']!r} sums to {s}"

    def test_pareto_weight_names_unique(self):
        names = [cfg["name"] for cfg in config.PARETO_WEIGHT_SETS]
        assert len(names) == len(set(names))


class TestLogger:
    def test_get_logger_returns_logger_instance(self):
        log = config.get_logger("test_config_xyz")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_config_xyz"

    def test_logger_emits_without_raising(self, caplog):
        log = config.get_logger("test_config_emit")
        with caplog.at_level(logging.INFO):
            log.info("hello from test")
        assert any("hello from test" in r.message for r in caplog.records)
