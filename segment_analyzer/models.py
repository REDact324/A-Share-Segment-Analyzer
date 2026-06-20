from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


Direction = str  # "up" or "down"
CandidateType = str  # "top" or "bottom"


@dataclass
class KRef:
    index: int
    date: str
    price: float
    role: str


@dataclass
class PrevPivotInfo:
    lower: float
    upper: float
    confirm_index: int | None = None
    used_indexes: list[int] = field(default_factory=list)


@dataclass
class PivotInfo:
    confirm_index: int
    confirm_date: str
    pivot_line: float
    pivot_type: str
    lower: float
    upper: float
    start_patterns: list[str]
    formation_end_index: int
    black_k_indices: list[int] = field(default_factory=list)
    extension_or_recomposition: str = "none"
    has_multiple_pivots: bool = False
    reasoning: list[str] = field(default_factory=list)


@dataclass
class CandidateEvaluation:
    candidate_type: CandidateType
    middle_index: int
    middle_date: str
    indexes: list[int]
    price: float
    checks: dict[str, bool] = field(default_factory=dict)
    turn_k_index: int | None = None
    turn_k_date: str | None = None
    status: str = "candidate"
    rejected_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class SegmentReport:
    segment_no: int
    segment_direction: Direction
    start_index: int
    start_date: str | None = None
    end_index: int | None = None
    end_date: str | None = None
    K0_index: int | None = None
    K0_date: str | None = None
    pivot_confirm_k: int | None = None
    pivot_confirm_date: str | None = None
    pivot_line: float | None = None
    pivot_type: str | None = None
    pivot_lower: float | None = None
    pivot_upper: float | None = None
    start_pattern: list[str] = field(default_factory=list)
    has_black_k: bool = False
    independent_2k_side: str = "未满足"
    candidate_top: list[CandidateEvaluation] = field(default_factory=list)
    candidate_bottom: list[CandidateEvaluation] = field(default_factory=list)
    confirmed_top: KRef | None = None
    confirmed_bottom: KRef | None = None
    turn_k: KRef | None = None
    ma34_check: bool = False
    initial_cross: KRef | None = None
    extension_or_recomposition: str = "none"
    refresh_status: str = "none"
    decision_status: str = "needs_more_k"
    reasoning_summary: str = ""
    unconfirmed_candidates: list[CandidateEvaluation] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisRun:
    symbol: str | None
    name: str | None
    trade_date: str | None
    data_start: str | None
    data_end: str | None
    segment_count: int
    reports: list[SegmentReport]
    errors: list[str] = field(default_factory=list)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return value
