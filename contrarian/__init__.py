# -*- coding: utf-8 -*-
"""Contrarian sentiment methodology — pluggable layer on top of BettaFish.

Defines three indicators used to detect the moment a topic "breaks out"
from a vertical community into mainstream consumer awareness — at which
point the contrarian (sell / short / reduce) signal is strongest.

Indicators
----------
V(t) — breakout velocity:
    d(unique_authors)/dt × H(cross-platform diffusion entropy)
S    — sentiment saturation:
    (extreme_positive_share - extreme_negative_share), 7-day z-score
D    — consensus dispersion:
    variance of comment polarity (lower = stronger consensus)

Contrarian Score = V'(t) × |S| × max(0, 1 - D)

Three-stage classifier:
    "萌芽期" — single community, low V
    "扩散期" — V and H rising
    "破圈期" — V' turning negative + H saturating + D collapsing → signal fires

This module is OFF by default. Activate by setting CONTRARIAN_MODE=true
in the project .env and explicitly calling build_*_prompt_addendum()
from InsightEngine/ReportEngine. The module is deliberately import-safe
and side-effect-free at import time.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_MAPPING_PATH: Path = PROJECT_ROOT / "config" / "topic_asset_mapping.json"

# Best-effort .env hydration. BettaFish uses pydantic-settings, which reads
# .env into its Settings object but does NOT populate os.environ. Without
# this load, is_enabled() / any os.environ reader would miss values that the
# user only ever wrote in .env. override=False keeps shell-exported vars
# authoritative — i.e. tests using monkeypatch.setenv still win.
try:
    from dotenv import load_dotenv as _load_dotenv

    _env_file = PROJECT_ROOT / ".env"
    if _env_file.exists():
        _load_dotenv(_env_file, override=False)
except ImportError:
    pass


@dataclass(frozen=True)
class ContrarianMetrics:
    """Immutable snapshot of the three contrarian indicators."""

    breakout_velocity: float
    velocity_derivative: float
    sentiment_saturation: float
    consensus_dispersion: float

    @property
    def contrarian_score(self) -> float:
        """V'(t) × |S| × max(0, 1 - D)."""
        return (
            self.velocity_derivative
            * abs(self.sentiment_saturation)
            * max(0.0, 1.0 - self.consensus_dispersion)
        )

    @property
    def stage(self) -> str:
        """Classify into one of three diffusion stages.

        Thresholds are intentionally conservative; tune against historical
        backtests before relying on them in production decisioning.
        """
        if self.breakout_velocity < 0.3:
            return "萌芽期"
        if self.velocity_derivative < 0 and self.consensus_dispersion < 0.15:
            return "破圈期"
        return "扩散期"


def is_enabled() -> bool:
    """Read CONTRARIAN_MODE from environment; default off."""
    return os.environ.get("CONTRARIAN_MODE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_insight_prompt_addendum(topic: str) -> str:
    """Prompt fragment for InsightEngine.

    Append after the engine's existing system prompt so the LLM is asked
    to produce a structured JSON snapshot of the three indicators.
    """
    return f"""
# 反指信号专项分析（话题：{topic}）

当前为反指分析模式（CONTRARIAN_MODE=true）。请在常规舆情分析之上额外评估:

1. **破圈速度 V(t)** = d(unique_authors)/dt × H(跨平台扩散熵)
   - 估算近 7 日趋势; 标记 accelerating / decelerating / peaking。
2. **情绪饱和度 S** = (极端正占比 - 极端负占比) 的 7 日 z-score。
   - |S| > 2σ 视作情绪极端化, 反指强信号。
3. **共识度 D** = 评论极性方差; 越低共识越强。
   - D 降到历史 10 分位以下 = 共识形成。
4. **Contrarian Score = V'(t) × |S| × max(0, 1 - D)**。
5. **阶段判定**: 萌芽期 / 扩散期 / 破圈期; 给出下一段拐点的时间窗预估。

请输出 strict JSON, schema 如下 (不要包裹 Markdown):
{{
  "breakout_velocity_estimate": float,
  "velocity_derivative_trend": "accelerating|decelerating|peaking",
  "sentiment_saturation_z": float,
  "consensus_dispersion": float,
  "contrarian_score": float,
  "stage": "萌芽期|扩散期|破圈期",
  "confidence": "high|medium|low",
  "key_evidence": [str, ...]
}}
"""


def build_report_prompt_addendum(metrics_payload: Mapping[str, object]) -> str:
    """Prompt fragment for ReportEngine — instructs choosing the contrarian template."""
    score = metrics_payload.get("contrarian_score", 0.0)
    stage = metrics_payload.get("stage", "未知")
    return f"""
# 报告生成模式: 反向交易信号

检测到反指信号 (Contrarian Score = {score}, Stage = {stage})。

请优先选择模板 "反向交易信号深度分析报告模板"。
在 6.0 章节调用 config/topic_asset_mapping.json 进行话题→资产标的映射;
未命中关键词则标记为 "待人工映射"。

报告必须包含:
- V/S/D 三指标的具体数值与置信度。
- 推荐反向操作 (卖出 / 做空 / 减仓 / 观望) + 入场与退出条件。
- 显式标注 "本报告仅供分析参考, 不构成投资建议"。
"""


def load_asset_mapping(path: Path | None = None) -> dict:
    """Load the topic→asset mapping dictionary; returns empty shape on miss."""
    target = path or DEFAULT_ASSET_MAPPING_PATH
    if not target.exists():
        return {
            "schema_version": "0.1.0",
            "categories": {},
            "manual_overrides": {},
            "notes": [f"mapping not found: {target}"],
        }
    return json.loads(target.read_text(encoding="utf-8"))


def find_assets(
    topic_or_keywords: str | Sequence[str],
    mapping: dict | None = None,
) -> list[dict]:
    """Find candidate tradable assets by matching keywords against the dictionary.

    Returns a flat list of asset dicts with added 'category' and 'bucket' fields.
    The match is substring-based and case-sensitive; upstream callers should
    normalise input (e.g. via the LLM) before calling.
    """
    resolved_mapping = mapping if mapping is not None else load_asset_mapping()
    queries: tuple[str, ...] = (
        (topic_or_keywords,)
        if isinstance(topic_or_keywords, str)
        else tuple(topic_or_keywords)
    )

    hits: list[dict] = []
    for category_name, payload in resolved_mapping.get("categories", {}).items():
        keywords: Iterable[str] = payload.get("keywords", [])
        if not any(any(kw in q for kw in keywords) for q in queries):
            continue
        for bucket in ("a_share", "hk_share", "us_adr", "crypto"):
            for asset in payload.get(bucket, []):
                hits.append({**asset, "category": category_name, "bucket": bucket})
    return hits


__all__ = [
    "ContrarianMetrics",
    "build_insight_prompt_addendum",
    "build_report_prompt_addendum",
    "find_assets",
    "is_enabled",
    "load_asset_mapping",
    "DEFAULT_ASSET_MAPPING_PATH",
]
