# -*- coding: utf-8 -*-
"""End-to-end tests for the `contrarian` methodology module.

Covers:
- `ContrarianMetrics` score arithmetic and stage classifier.
- `find_assets` keyword matching against the bundled JSON dictionary.
- `is_enabled` env var parsing.
- `load_asset_mapping` happy-path and missing-file fallback.
- `build_insight_prompt_addendum` / `build_report_prompt_addendum` content.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrarian import (  # noqa: E402  (path tweak above)
    ContrarianMetrics,
    build_insight_prompt_addendum,
    build_report_prompt_addendum,
    find_assets,
    is_enabled,
    load_asset_mapping,
)


# ---- ContrarianMetrics ----------------------------------------------------


def test_metrics_score_calculation():
    m = ContrarianMetrics(
        breakout_velocity=0.8,
        velocity_derivative=-0.2,
        sentiment_saturation=2.5,
        consensus_dispersion=0.1,
    )
    # V' × |S| × (1 - D) = -0.2 × 2.5 × 0.9 = -0.45
    assert m.contrarian_score == pytest.approx(-0.45, abs=1e-9)


def test_metrics_score_clamps_dispersion_above_one():
    """If D >= 1, the (1 - D) factor clamps to 0 (no negative weighting)."""
    m = ContrarianMetrics(
        breakout_velocity=0.8,
        velocity_derivative=0.5,
        sentiment_saturation=2.0,
        consensus_dispersion=1.5,
    )
    assert m.contrarian_score == 0.0


def test_stage_seedling_when_velocity_low():
    m = ContrarianMetrics(0.1, 0.05, 0.5, 0.5)
    assert m.stage == "萌芽期"


def test_stage_diffusion_default():
    m = ContrarianMetrics(0.5, 0.2, 0.5, 0.5)
    assert m.stage == "扩散期"


def test_stage_breakout_when_velocity_derivative_negative_and_consensus_tight():
    m = ContrarianMetrics(0.9, -0.3, 2.5, 0.1)
    assert m.stage == "破圈期"


def test_metrics_is_frozen():
    m = ContrarianMetrics(0.5, 0.1, 1.0, 0.2)
    with pytest.raises(Exception):
        m.breakout_velocity = 0.9  # type: ignore[misc]


# ---- find_assets ----------------------------------------------------------


def test_find_assets_medical_aesthetic_hit():
    hits = find_assets("玻尿酸")
    tickers = {h["ticker"] for h in hits}
    assert "300896.SZ" in tickers  # 爱美客
    assert "688363.SH" in tickers  # 华熙生物


def test_find_assets_miss_returns_empty_list():
    assert find_assets("一段完全无关的字符串_xyz_123") == []


def test_find_assets_accepts_keyword_sequence():
    hits = find_assets(["医美", "出境游"])
    categories = {h["category"] for h in hits}
    assert "美妆_医美" in categories
    assert "出境游_酒店_航空" in categories


def test_find_assets_includes_crypto_bucket():
    hits = find_assets("比特币")
    buckets = {h["bucket"] for h in hits}
    assert "crypto" in buckets


def test_find_assets_returns_category_and_bucket_annotations():
    hits = find_assets("玻尿酸")
    assert hits, "expected at least one hit"
    for h in hits:
        assert h["category"] == "美妆_医美"
        assert h["bucket"] in {"a_share", "hk_share", "us_adr", "crypto"}


# ---- is_enabled -----------------------------------------------------------


def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv("CONTRARIAN_MODE", raising=False)
    assert is_enabled() is False


@pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "on"])
def test_is_enabled_truthy_variants(monkeypatch, value):
    monkeypatch.setenv("CONTRARIAN_MODE", value)
    assert is_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "garbage"])
def test_is_enabled_falsy_variants(monkeypatch, value):
    monkeypatch.setenv("CONTRARIAN_MODE", value)
    assert is_enabled() is False


# ---- load_asset_mapping ---------------------------------------------------


def test_load_default_mapping_has_categories():
    mapping = load_asset_mapping()
    assert "categories" in mapping
    assert len(mapping["categories"]) >= 10


def test_load_missing_mapping_returns_safe_empty(tmp_path):
    missing = tmp_path / "nonexistent.json"
    mapping = load_asset_mapping(missing)
    assert mapping["categories"] == {}
    assert any("not found" in n for n in mapping["notes"])


# ---- prompt builders ------------------------------------------------------


def test_insight_prompt_contains_metrics_definitions():
    p = build_insight_prompt_addendum("医美话题")
    for token in ["破圈速度", "Contrarian Score", "医美话题", "strict JSON"]:
        assert token in p


def test_report_prompt_includes_score_stage_and_template_reference():
    p = build_report_prompt_addendum({"contrarian_score": 0.75, "stage": "破圈期"})
    assert "0.75" in p
    assert "破圈期" in p
    assert "反向交易信号深度分析报告模板" in p
    assert "不构成投资建议" in p
