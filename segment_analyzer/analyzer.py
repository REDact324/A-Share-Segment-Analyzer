from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .indicators import add_moving_averages, format_date
from .models import (
    CandidateEvaluation,
    KRef,
    PivotInfo,
    PrevPivotInfo,
    SegmentReport,
)


UP = "up"
DOWN = "down"
TOP = "top"
BOTTOM = "bottom"


@dataclass
class AnalyzerConfig:
    max_segments: int = 200
    min_remaining_bars: int = 45
    start_pattern_scan_bars: int = 80
    pivot_zone_scan_bars: int = 80
    visual_ma5_overlap_ratio: float = 0.0015
    batch_error_limit: int = 20
    max_discovery_attempts: int = 5000


class SegmentAnalyzer:
    def __init__(self, config: AnalyzerConfig | None = None):
        self.config = config or AnalyzerConfig()

    def analyze_all(
        self,
        df: pd.DataFrame,
        initial_direction: str = "auto",
        previous_pivot: PrevPivotInfo | None = None,
    ) -> list[SegmentReport]:
        data = add_moving_averages(df)
        reports: list[SegmentReport] = []
        start_index = first_valid_index(data, "ma34")
        if start_index is None:
            return [
                SegmentReport(
                    segment_no=1,
                    segment_direction=UP,
                    start_index=0,
                    start_date=format_date(data.iloc[0].date) if not data.empty else None,
                    decision_status="invalid",
                    reasoning_summary="数据不足，无法计算 MA34。",
                )
            ]

        direction = "both" if initial_direction == "auto" else self._initial_direction(data, start_index, initial_direction)
        prev_endpoint: KRef | None = None
        prev_pivot = previous_pivot
        prev_turn_index: int | None = None
        last_candidate: SegmentReport | None = None
        discovery_attempts = 0
        segment_no = 1

        while segment_no <= self.config.max_segments:
            if len(data) - start_index < self.config.min_remaining_bars:
                break

            report = self._analyze_direction_set(
                data=data,
                direction=direction,
                segment_no=segment_no,
                start_index=start_index,
                previous_pivot=prev_pivot,
                previous_endpoint=prev_endpoint,
                previous_turn_index=prev_turn_index,
            )
            actual_direction = report.segment_direction
            endpoint = report.confirmed_top if actual_direction == UP else report.confirmed_bottom
            if report.decision_status != "confirmed" or endpoint is None:
                if prev_endpoint is not None:
                    reports.append(report)
                    break
                last_candidate = report
                discovery_attempts += 1
                if discovery_attempts >= self.config.max_discovery_attempts:
                    break
                start_index += 1
                continue
            if endpoint.index <= start_index:
                report.decision_status = "invalid"
                report.reasoning_summary += " 连续线段推进失败：确认端点没有晚于当前起点。"
                reports.append(report)
                break

            reports.append(report)
            start_index = endpoint.index
            prev_endpoint = endpoint
            prev_turn_index = report.turn_k.index if report.turn_k else prev_turn_index
            if report.pivot_lower is not None and report.pivot_upper is not None:
                prev_pivot = PrevPivotInfo(
                    lower=report.pivot_lower,
                    upper=report.pivot_upper,
                    confirm_index=report.pivot_confirm_k,
                )
            direction = opposite(actual_direction)
            segment_no += 1
        if not reports and last_candidate is not None:
            reports.append(last_candidate)
        self._post_adjust_shared_endpoints_by_in_range_candidates(data, reports)
        return reports

    def analyze_from_lowest(self, df: pd.DataFrame, max_segments: int | None = None) -> list[SegmentReport]:
        data = add_moving_averages(df)
        start_index = first_valid_index(data, "ma34")
        if start_index is None:
            return [
                SegmentReport(
                    segment_no=1,
                    segment_direction=UP,
                    start_index=0,
                    start_date=format_date(data.iloc[0].date) if not data.empty else None,
                    decision_status="invalid",
                    reasoning_summary="数据不足，无法计算 MA34。",
                )
            ]
        if data.empty:
            return []

        low_index = int(data["low"].idxmin())
        low_row = data.iloc[low_index]
        anchor = KRef(
            index=low_index,
            date=format_date(low_row.date),
            price=float(low_row.low),
            role="historical_low_bottom",
        )

        remaining = max_segments or self.config.max_segments
        future_reports = self._scan_from_endpoint(
            data=data,
            start_index=low_index,
            previous_endpoint=anchor,
            direction=UP,
            start_segment_no=1,
            max_segments=remaining,
            enable_endpoint_extension=True,
        )
        remaining -= len(future_reports)

        past_reports: list[SegmentReport] = []
        if remaining > 0 and low_index - start_index >= 15:
            past_reports = self._scan_left_of_lowest_by_extreme_anchors(
                data=data,
                first_valid=start_index,
                low_index=low_index,
                max_segments=remaining,
            )

        reports = past_reports + future_reports
        self._post_adjust_shared_endpoints_by_in_range_candidates(data, reports)
        for segment_no, report in enumerate(reports, start=1):
            report.segment_no = segment_no
            report.diagnostics["scan_mode"] = "historical_low_anchor"
            report.diagnostics["historical_low_index"] = low_index
            report.diagnostics["historical_low_date"] = anchor.date
            report.diagnostics["historical_low_price"] = anchor.price
        return reports

    def _scan_from_endpoint(
        self,
        data: pd.DataFrame,
        start_index: int,
        previous_endpoint: KRef,
        direction: str,
        start_segment_no: int,
        max_segments: int,
        enable_endpoint_extension: bool = True,
        end_limit_index: int | None = None,
    ) -> list[SegmentReport]:
        reports: list[SegmentReport] = []
        prev_endpoint = previous_endpoint
        prev_pivot: PrevPivotInfo | None = None
        prev_turn_index: int | None = None
        current_start = start_index
        current_direction = direction

        for offset in range(max_segments):
            if len(data) - current_start < 5:
                break
            report = self.analyze_one(
                data,
                current_direction,
                segment_no=start_segment_no + offset,
                start_index=current_start,
                previous_pivot=prev_pivot,
                previous_endpoint=prev_endpoint,
                previous_turn_index=prev_turn_index,
                end_selection="earliest",
                finalize_output=False,
                end_limit_index=end_limit_index,
            )
            endpoint = report.confirmed_top if current_direction == UP else report.confirmed_bottom
            if report.decision_status != "confirmed" or endpoint is None:
                break
            if endpoint.index <= current_start:
                break
            if end_limit_index is not None and endpoint.index > end_limit_index:
                break

            if enable_endpoint_extension:
                self._extend_until_opposite_endpoint(
                    data=data,
                    report=report,
                    direction=current_direction,
                    previous_pivot=prev_pivot,
                    boundary_index=end_limit_index,
                )
            selected_end = self._selected_endpoint_candidate(report, current_direction)
            self._trim_candidates_for_report(report, selected_end)
            self._mark_unconfirmed(report, selected_end)

            reports.append(report)
            endpoint = report.confirmed_top if current_direction == UP else report.confirmed_bottom
            current_start = endpoint.index
            prev_endpoint = endpoint
            prev_turn_index = report.turn_k.index if report.turn_k else prev_turn_index
            if report.pivot_lower is not None and report.pivot_upper is not None:
                prev_pivot = PrevPivotInfo(
                    lower=report.pivot_lower,
                    upper=report.pivot_upper,
                    confirm_index=report.pivot_confirm_k,
                )
            current_direction = opposite(current_direction)
        return reports

    def _scan_left_of_lowest_by_extreme_anchors(
        self,
        data: pd.DataFrame,
        first_valid: int,
        low_index: int,
        max_segments: int,
    ) -> list[SegmentReport]:
        chunks: list[list[SegmentReport]] = []
        boundary_index = low_index
        boundary_row = data.iloc[low_index]
        boundary_ref = KRef(
            index=low_index,
            date=format_date(boundary_row.date),
            price=float(boundary_row.low),
            role="historical_low_bottom",
        )
        anchor_type = TOP
        direction = DOWN
        remaining = max_segments

        while remaining > 0 and boundary_index - first_valid >= 15:
            anchor = self._left_extreme_anchor(data, first_valid, boundary_index, anchor_type)
            if anchor is None or anchor.index >= boundary_index:
                break

            chunk = self._scan_from_endpoint(
                data=data,
                start_index=anchor.index,
                previous_endpoint=anchor,
                direction=direction,
                start_segment_no=1,
                max_segments=remaining,
                enable_endpoint_extension=True,
                end_limit_index=boundary_index,
            )
            if chunk:
                for report in chunk:
                    report.diagnostics["left_anchor_boundary_index"] = boundary_index
                    report.diagnostics["left_anchor_boundary_date"] = boundary_ref.date
                    report.diagnostics["left_anchor_index"] = anchor.index
                    report.diagnostics["left_anchor_date"] = anchor.date
                    report.diagnostics["left_anchor_role"] = anchor.role
                closures = self._boundary_closure_reports(data, chunk[-1], boundary_ref)
                if closures:
                    for closure in closures:
                        closure.diagnostics["left_anchor_boundary_index"] = boundary_index
                        closure.diagnostics["left_anchor_boundary_date"] = boundary_ref.date
                        closure.diagnostics["left_anchor_index"] = anchor.index
                        closure.diagnostics["left_anchor_date"] = anchor.date
                        closure.diagnostics["left_anchor_role"] = anchor.role
                    chunk.extend(closures)
                chunks.append(chunk)
                remaining -= len(chunk)
            else:
                closure = self._anchor_to_boundary_report(data, anchor, boundary_ref, direction)
                if closure is not None:
                    closure.diagnostics["left_anchor_boundary_index"] = boundary_index
                    closure.diagnostics["left_anchor_boundary_date"] = boundary_ref.date
                    closure.diagnostics["left_anchor_index"] = anchor.index
                    closure.diagnostics["left_anchor_date"] = anchor.date
                    closure.diagnostics["left_anchor_role"] = anchor.role
                    chunks.append([closure])
                    remaining -= 1

            boundary_index = anchor.index
            boundary_ref = anchor
            anchor_type = BOTTOM if anchor_type == TOP else TOP
            direction = UP if direction == DOWN else DOWN

        reports = [report for chunk in reversed(chunks) for report in chunk]
        reports.sort(key=lambda report: (report.start_index, report.end_index or 10**12))
        return reports[:max_segments]

    def _left_extreme_anchor(
        self, data: pd.DataFrame, first_valid: int, boundary_index: int, anchor_type: str
    ) -> KRef | None:
        window = data.iloc[first_valid:boundary_index]
        if window.empty:
            return None
        if anchor_type == TOP:
            index = int(window["high"].idxmax())
            row = data.iloc[index]
            return KRef(index=index, date=format_date(row.date), price=float(row.high), role="left_anchor_top")
        index = int(window["low"].idxmin())
        row = data.iloc[index]
        return KRef(index=index, date=format_date(row.date), price=float(row.low), role="left_anchor_bottom")

    def _boundary_closure_reports(
        self, data: pd.DataFrame, last_report: SegmentReport, boundary_ref: KRef
    ) -> list[SegmentReport]:
        if last_report.end_index is None or last_report.end_index >= boundary_ref.index:
            return []
        start_ref = last_report.confirmed_bottom if last_report.segment_direction == DOWN else last_report.confirmed_top
        if start_ref is None:
            return []
        direction = opposite(last_report.segment_direction)
        direct = self._anchor_to_boundary_report(data, start_ref, boundary_ref, direction)
        if direct is not None:
            return [direct]
        bridge = self._opposite_extreme_between(data, start_ref, boundary_ref)
        if bridge is None:
            return []
        first_direction = UP if "bottom" in start_ref.role else DOWN
        first = self._anchor_to_boundary_report(data, start_ref, bridge, first_direction)
        second = self._anchor_to_boundary_report(data, bridge, boundary_ref, opposite(first_direction))
        return [report for report in [first, second] if report is not None]

    def _opposite_extreme_between(self, data: pd.DataFrame, start_ref: KRef, boundary_ref: KRef) -> KRef | None:
        window = data.iloc[start_ref.index + 1 : boundary_ref.index]
        if window.empty:
            return None
        if "bottom" in start_ref.role and "bottom" in boundary_ref.role:
            index = int(window["high"].idxmax())
            row = data.iloc[index]
            return KRef(index=index, date=format_date(row.date), price=float(row.high), role="anchored_bridge_top")
        if "top" in start_ref.role and "top" in boundary_ref.role:
            index = int(window["low"].idxmin())
            row = data.iloc[index]
            return KRef(index=index, date=format_date(row.date), price=float(row.low), role="anchored_bridge_bottom")
        return None

    def _anchor_to_boundary_report(
        self, data: pd.DataFrame, start_ref: KRef, boundary_ref: KRef, direction: str
    ) -> SegmentReport | None:
        if start_ref.index >= boundary_ref.index:
            return None
        if direction == UP and not ("bottom" in start_ref.role and "top" in boundary_ref.role):
            return None
        if direction == DOWN and not ("top" in start_ref.role and "bottom" in boundary_ref.role):
            return None
        report = SegmentReport(
            segment_no=1,
            segment_direction=direction,
            start_index=start_ref.index,
            start_date=start_ref.date,
            end_index=boundary_ref.index,
            end_date=boundary_ref.date,
            decision_status="confirmed",
            refresh_status="anchored_boundary_closure",
        )
        if direction == UP:
            report.confirmed_bottom = start_ref
            report.confirmed_top = boundary_ref
        else:
            report.confirmed_top = start_ref
            report.confirmed_bottom = boundary_ref
        report.reasoning_summary = (
            f"已按左侧极值锚点闭合线段：方向={direction}，"
            f"底={describe_ref(report.confirmed_bottom)}，顶={describe_ref(report.confirmed_top)}。"
        )
        report.diagnostics["anchored_boundary_closure"] = True
        report.diagnostics["anchored_boundary_closure_reason"] = (
            "左侧按正时序扫描未自然闭合到当前极值边界，按用户规则将边界极值作为确认端点。"
        )
        return report

    def _reverse_data_for_backward_scan(self, data: pd.DataFrame) -> pd.DataFrame:
        reversed_data = data.copy()
        reversed_data["original_index"] = reversed_data.index
        return reversed_data.iloc[::-1].reset_index(drop=True)

    def _map_reversed_report_to_original(
        self, report: SegmentReport, reversed_data: pd.DataFrame
    ) -> SegmentReport:
        report.segment_direction = opposite(report.segment_direction)
        report.start_index = map_optional_index(report.start_index, reversed_data)
        report.start_date = date_for_original_index(reversed_data, report.start_index)
        report.end_index = map_optional_index(report.end_index, reversed_data)
        report.end_date = date_for_original_index(reversed_data, report.end_index)
        report.K0_index = map_optional_index(report.K0_index, reversed_data)
        report.K0_date = date_for_original_index(reversed_data, report.K0_index)
        report.pivot_confirm_k = map_optional_index(report.pivot_confirm_k, reversed_data)
        report.pivot_confirm_date = date_for_original_index(reversed_data, report.pivot_confirm_k)
        report.confirmed_top = map_kref_index(report.confirmed_top, reversed_data)
        report.confirmed_bottom = map_kref_index(report.confirmed_bottom, reversed_data)
        report.turn_k = map_kref_index(report.turn_k, reversed_data)
        report.initial_cross = map_kref_index(report.initial_cross, reversed_data)
        report.candidate_top = [map_candidate_index(ev, reversed_data) for ev in report.candidate_top]
        report.candidate_bottom = [map_candidate_index(ev, reversed_data) for ev in report.candidate_bottom]
        report.unconfirmed_candidates = [
            map_candidate_index(ev, reversed_data) for ev in report.unconfirmed_candidates
        ]
        map_index_date_diagnostics(report, reversed_data)
        map_endpoint_extension_diagnostics(report, reversed_data)
        if report.segment_direction == UP:
            report.start_index = report.confirmed_bottom.index if report.confirmed_bottom else report.start_index
            report.start_date = report.confirmed_bottom.date if report.confirmed_bottom else report.start_date
            report.end_index = report.confirmed_top.index if report.confirmed_top else report.end_index
            report.end_date = report.confirmed_top.date if report.confirmed_top else report.end_date
        else:
            report.start_index = report.confirmed_top.index if report.confirmed_top else report.start_index
            report.start_date = report.confirmed_top.date if report.confirmed_top else report.start_date
            report.end_index = report.confirmed_bottom.index if report.confirmed_bottom else report.end_index
            report.end_date = report.confirmed_bottom.date if report.confirmed_bottom else report.end_date
        report.diagnostics["mapped_from_reverse_time"] = True
        refresh_report_summary(report)
        return report

    def _extend_until_opposite_endpoint(
        self,
        data: pd.DataFrame,
        report: SegmentReport,
        direction: str,
        previous_pivot: PrevPivotInfo | None,
        boundary_index: int | None = None,
    ) -> None:
        target_pool = report.candidate_top if direction == UP else report.candidate_bottom
        current_endpoint = report.confirmed_top if direction == UP else report.confirmed_bottom
        if current_endpoint is None or report.pivot_lower is None or report.pivot_upper is None:
            return

        while True:
            opposite_direction = opposite(direction)
            current_pivot = self._report_pivot_info(report)
            opposite_report = self.analyze_one(
                data,
                opposite_direction,
                segment_no=report.segment_no,
                start_index=current_endpoint.index,
                previous_pivot=current_pivot or previous_pivot,
                previous_endpoint=current_endpoint,
                previous_turn_index=report.turn_k.index if report.turn_k else None,
                end_selection="earliest",
                finalize_output=False,
                end_limit_index=boundary_index,
            )
            opposite_endpoint = (
                opposite_report.confirmed_top
                if opposite_direction == UP
                else opposite_report.confirmed_bottom
            )
            opposite_boundary_index = (
                opposite_endpoint.index
                if opposite_report.decision_status == "confirmed" and opposite_endpoint is not None
                else 10**12
            )
            if boundary_index is not None:
                opposite_boundary_index = min(opposite_boundary_index, boundary_index)
            self._record_extension_boundary(
                data=data,
                report=report,
                current_endpoint=current_endpoint,
                opposite_report=opposite_report,
                opposite_boundary_index=opposite_boundary_index,
            )

            new_endpoint = self._find_endpoint_refind_candidate(
                direction=direction,
                candidates=target_pool,
                search_start=current_endpoint.index + 1,
                boundary_index=opposite_boundary_index,
                current_endpoint=current_endpoint,
                current_turn_index=report.turn_k.index if report.turn_k else None,
                data=data,
                formation_end_index=report.diagnostics.get("pivot_formation_end_index", report.pivot_confirm_k),
                start_endpoint=report.confirmed_bottom if direction == UP else report.confirmed_top,
                pivot=self._report_full_pivot_info(report),
            )
            if new_endpoint is None:
                return

            self._apply_endpoint_extension(data, report, direction, new_endpoint, opposite_boundary_index)
            current_endpoint = report.confirmed_top if direction == UP else report.confirmed_bottom

    def _record_extension_boundary(
        self,
        data: pd.DataFrame,
        report: SegmentReport,
        current_endpoint: KRef,
        opposite_report: SegmentReport,
        opposite_boundary_index: int,
    ) -> None:
        boundary_date = (
            format_date(data.iloc[opposite_boundary_index].date)
            if 0 <= opposite_boundary_index < len(data)
            else None
        )
        report.diagnostics.setdefault("extension_boundaries", []).append(
            {
                "current_endpoint_index": current_endpoint.index,
                "current_endpoint_date": current_endpoint.date,
                "opposite_segment_status": opposite_report.decision_status,
                "opposite_segment_direction": opposite_report.segment_direction,
                "opposite_segment_boundary_index": None
                if opposite_boundary_index == 10**12
                else opposite_boundary_index,
                "opposite_segment_boundary_date": boundary_date,
                "rule": "先寻找反向新线段；如果可延伸顶/底早于反向线段确认端点，则延伸并从延伸端点继续。",
            }
        )

    def _ma5_endpoint_growth_index(
        self,
        data: pd.DataFrame,
        direction: str,
        current_endpoint: KRef,
        boundary_index: int,
    ) -> int | None:
        endpoint_ma5 = data.iloc[current_endpoint.index].ma5
        if pd.isna(endpoint_ma5):
            return None
        scan_end = min(boundary_index, len(data))
        for i in range(current_endpoint.index + 1, scan_end):
            ma5 = data.iloc[i].ma5
            if pd.isna(ma5):
                continue
            if direction == UP and rule_gt(ma5, endpoint_ma5):
                return i
            if direction == DOWN and rule_lt(ma5, endpoint_ma5):
                return i
        return None

    def _outside_previous_pivot_start(
        self,
        data: pd.DataFrame,
        direction: str,
        start_index: int,
        pivot_lower: float,
        pivot_upper: float,
        boundary_index: int,
    ) -> int | None:
        scan_end = min(boundary_index, len(data))
        for i in range(start_index, scan_end):
            row = data.iloc[i]
            if direction == UP and rule_gt(row.low, pivot_upper):
                return i
            if direction == DOWN and rule_lt(row.high, pivot_lower):
                return i
        return None

    def _find_endpoint_refind_candidate(
        self,
        direction: str,
        candidates: list[CandidateEvaluation],
        search_start: int,
        boundary_index: int,
        current_endpoint: KRef,
        current_turn_index: int | None,
        data: pd.DataFrame,
        formation_end_index: int | None,
        start_endpoint: KRef | None = None,
        pivot: PivotInfo | None = None,
    ) -> CandidateEvaluation | None:
        valid = []
        for ev in candidates:
            if not (search_start <= ev.middle_index < boundary_index):
                continue
            if formation_end_index is not None and ev.middle_index <= formation_end_index:
                continue
            if pivot is not None and start_endpoint is not None:
                ev.checks["pair_independent_2k"] = self._pair_independent_2k(data, start_endpoint, ev, pivot)
                ev.checks["structure_perfect"] = self._structure_perfect(data, direction, start_endpoint, ev, pivot)
            if not self._is_extension_endpoint(
                direction, ev, current_endpoint, current_turn_index, data, start_endpoint
            ):
                continue
            if not (
                ev.checks.get("ma34_side_and_strict_ma5")
                and ev.checks.get("not_black_k")
                and ev.checks.get("structure_perfect")
                and ev.checks.get("has_turn_k")
                and ev.checks.get("not_turn_k_itself")
            ):
                continue
            valid.append(ev)
        if not valid:
            return None
        valid = self._keep_nearest_candidate_per_turn_k(valid)
        valid.sort(key=lambda ev: (ev.turn_k_index or 10**12, ev.middle_index))
        return valid[0]

    def _is_extension_endpoint(
        self,
        direction: str,
        candidate: CandidateEvaluation,
        current_endpoint: KRef,
        current_turn_index: int | None,
        data: pd.DataFrame,
        start_endpoint: KRef | None = None,
    ) -> bool:
        current_turn_ma5 = (
            data.iloc[current_turn_index].ma5
            if current_turn_index is not None and 0 <= current_turn_index < len(data)
            else None
        )
        candidate_turn_ma5 = (
            data.iloc[candidate.turn_k_index].ma5
            if candidate.turn_k_index is not None and 0 <= candidate.turn_k_index < len(data)
            else None
        )
        segment_start = start_endpoint.index if start_endpoint is not None else current_endpoint.index
        segment_low = min(segment_start, candidate.middle_index)
        segment_high = max(segment_start, candidate.middle_index)
        current_row = data.iloc[current_endpoint.index]
        candidate_row = data.iloc[candidate.middle_index]
        if direction == UP:
            segment_max = data.iloc[segment_low : segment_high + 1]["high"].max()
            price_extends = (
                rule_gt(candidate.price, current_endpoint.price)
                and rule_ge(candidate.price, segment_max)
                and rule_gt(body_high(candidate_row), body_low(current_row))
            )
            ma5_extends = (
                pd.notna(current_turn_ma5)
                and pd.notna(candidate_turn_ma5)
                and rule_gt(candidate_turn_ma5, current_turn_ma5)
            )
            return bool(price_extends or ma5_extends)
        segment_min = data.iloc[segment_low : segment_high + 1]["low"].min()
        price_extends = (
            rule_lt(candidate.price, current_endpoint.price)
            and rule_le(candidate.price, segment_min)
            and rule_lt(body_low(candidate_row), body_high(current_row))
        )
        ma5_extends = (
            pd.notna(current_turn_ma5)
            and pd.notna(candidate_turn_ma5)
            and rule_lt(candidate_turn_ma5, current_turn_ma5)
        )
        return bool(price_extends or ma5_extends)

    def _apply_endpoint_extension(
        self,
        df: pd.DataFrame,
        report: SegmentReport,
        direction: str,
        extension: CandidateEvaluation,
        boundary_index: int,
    ) -> None:
        if direction == UP:
            report.confirmed_top = eval_to_ref(df, extension, TOP)
        else:
            report.confirmed_bottom = eval_to_ref(df, extension, BOTTOM)
        report.turn_k = KRef(
            index=extension.turn_k_index,
            date=extension.turn_k_date or format_date(df.iloc[extension.turn_k_index].date),
            price=float(df.iloc[extension.turn_k_index].close),
            role="turn_k",
        )
        report.end_index = extension.middle_index
        report.end_date = extension.middle_date
        report.refresh_status = "confirmed_endpoint_extension"
        report.diagnostics.setdefault("endpoint_extensions", []).append(
            {
                "extended_to_index": extension.middle_index,
                "extended_to_date": extension.middle_date,
                "extended_to_price": extension.price,
                "boundary_index": None if boundary_index == 10**12 else boundary_index,
            }
        )
        extension.status = "confirmed"
        refresh_report_summary(report)

    def _post_adjust_shared_endpoints_by_in_range_candidates(
        self, df: pd.DataFrame, reports: list[SegmentReport]
    ) -> None:
        for pos, report in enumerate(reports):
            if report.decision_status != "confirmed":
                continue
            previous = reports[pos - 1] if pos > 0 else None
            previous_start_turn = reports[pos - 2].turn_k if pos > 1 else None
            replacement = self._select_post_adjustment_candidate(df, report, previous, previous_start_turn)
            if replacement is None:
                continue
            old_ref = report.confirmed_top if report.segment_direction == DOWN else report.confirmed_bottom
            if old_ref is None:
                continue
            new_ref = eval_to_ref(df, replacement, TOP if report.segment_direction == DOWN else BOTTOM)
            self._apply_post_endpoint_replacement(df, report, old_ref, new_ref, replacement)
            if pos > 0:
                self._sync_previous_shared_endpoint(df, reports[pos - 1], old_ref, new_ref)

    def _select_post_adjustment_candidate(
        self,
        df: pd.DataFrame,
        report: SegmentReport,
        previous: SegmentReport | None,
        previous_start_turn: KRef | None,
    ) -> CandidateEvaluation | None:
        if report.pivot_lower is None or report.pivot_upper is None:
            return None
        pivot = self._report_full_pivot_info(report)
        if pivot is None:
            return None
        previous_ma5_extreme = self._previous_segment_ma5_threshold(
            df, previous, previous_start_turn, report.segment_direction
        )
        if previous_ma5_extreme is None:
            return None

        if report.segment_direction == DOWN:
            current = report.confirmed_top
            opposite = report.confirmed_bottom
            pool = report.candidate_top
            candidate_type = TOP
        else:
            current = report.confirmed_bottom
            opposite = report.confirmed_top
            pool = report.candidate_bottom
            candidate_type = BOTTOM
        if current is None or opposite is None:
            return None

        left, right = sorted((current.index, opposite.index))
        valid: list[CandidateEvaluation] = []
        for ev in pool:
            if ev.candidate_type != candidate_type:
                continue
            if ev.middle_index == current.index or not (left <= ev.middle_index <= right):
                continue
            row = df.iloc[ev.middle_index]
            if pd.isna(row.ma5):
                continue
            matched_ma5_index = self._candidate_neighbor_ma5_match(
                df, ev, report.segment_direction, previous_ma5_extreme
            )
            if matched_ma5_index is None:
                continue
            if not self._post_adjustment_candidate_satisfies_rules(df, report, ev, opposite, pivot):
                continue
            ev.notes.append(
                "全局后处理："
                f"候选MA5={rule_value(row.ma5):.2f}，"
                f"满足阈值K={format_date(df.iloc[matched_ma5_index].date)}#{matched_ma5_index} "
                f"MA5={rule_value(df.iloc[matched_ma5_index].ma5):.2f}，"
                f"上一线段MA5阈值={rule_value(previous_ma5_extreme):.2f}。"
            )
            valid.append(ev)

        if not valid:
            return None
        if report.segment_direction == DOWN:
            valid.sort(key=lambda ev: (rule_value(df.iloc[ev.middle_index].ma5), ev.middle_index), reverse=True)
        else:
            valid.sort(key=lambda ev: (rule_value(df.iloc[ev.middle_index].ma5), ev.middle_index))
        return valid[0]

    def _candidate_neighbor_ma5_match(
        self,
        df: pd.DataFrame,
        candidate: CandidateEvaluation,
        direction: str,
        threshold: float,
    ) -> int | None:
        matched: list[int] = []
        for index in range(candidate.middle_index - 2, candidate.middle_index + 3):
            if index < 0 or index >= len(df):
                continue
            ma5 = df.iloc[index].ma5
            if pd.isna(ma5):
                continue
            if direction == DOWN and rule_gt(ma5, threshold):
                matched.append(index)
            elif direction == UP and rule_lt(ma5, threshold):
                matched.append(index)
        if not matched:
            return None
        if direction == DOWN:
            return max(matched, key=lambda index: (rule_value(df.iloc[index].ma5), index))
        return min(matched, key=lambda index: (rule_value(df.iloc[index].ma5), index))

    def _previous_segment_ma5_threshold(
        self,
        df: pd.DataFrame,
        report: SegmentReport | None,
        start_turn: KRef | None,
        direction: str,
    ) -> float | None:
        if report is None or start_turn is None or report.turn_k is None:
            return None
        left, right = sorted((start_turn.index, report.turn_k.index))
        if left < 0 or right >= len(df):
            return None
        values = df.iloc[left : right + 1]["ma5"].dropna()
        if values.empty:
            return None
        if direction == DOWN:
            return float(values.max())
        return float(values.min())

    def _post_adjustment_candidate_satisfies_rules(
        self,
        df: pd.DataFrame,
        report: SegmentReport,
        candidate: CandidateEvaluation,
        opposite: KRef,
        pivot: PivotInfo,
    ) -> bool:
        candidate.checks["pair_independent_2k"] = self._pair_independent_2k(df, opposite, candidate, pivot)
        candidate.checks["ma34_side_and_strict_ma5"] = self._ma34_check(
            df, candidate.middle_index, candidate.candidate_type
        )
        candidate.checks["not_black_k"] = candidate.middle_index not in set(pivot.black_k_indices)
        candidate.checks.setdefault("not_turn_k_itself", True)
        candidate.checks["structure_perfect"] = self._post_adjustment_structure_perfect(
            df, candidate, pivot
        )
        candidate.checks["has_initial_cross"] = (
            self._find_initial_cross(
                df,
                min(candidate.middle_index, opposite.index),
                max(candidate.middle_index, opposite.index),
            )
            is not None
        )
        checks = [
            candidate.checks.get("pair_independent_2k"),
            candidate.checks.get("ma34_side_and_strict_ma5"),
            candidate.checks.get("not_black_k"),
            candidate.checks.get("not_turn_k_itself"),
            candidate.checks.get("structure_perfect"),
            candidate.checks.get("has_turn_k"),
            candidate.checks.get("has_initial_cross"),
        ]
        if "dynamic_ma5_survives" in candidate.checks:
            checks.append(candidate.checks.get("dynamic_ma5_survives"))
        return all(checks)

    def _post_adjustment_structure_perfect(
        self, df: pd.DataFrame, candidate: CandidateEvaluation, pivot: PivotInfo
    ) -> bool:
        candidate_low, candidate_high = endpoint_low_high(df, candidate)
        if candidate.candidate_type == TOP:
            if not bool(candidate.checks.get("independent_2k")) and not pivot.has_multiple_pivots:
                return rule_gt(candidate_high, pivot.lower)
            return rule_ge(candidate_low, pivot.upper)
        if not bool(candidate.checks.get("independent_2k")) and not pivot.has_multiple_pivots:
            return rule_lt(candidate_low, pivot.upper)
        return rule_le(candidate_high, pivot.lower)

    def _apply_post_endpoint_replacement(
        self,
        df: pd.DataFrame,
        report: SegmentReport,
        old_ref: KRef,
        new_ref: KRef,
        candidate: CandidateEvaluation,
    ) -> None:
        if report.segment_direction == DOWN:
            report.confirmed_top = new_ref
            report.start_index = new_ref.index
            report.start_date = new_ref.date
        else:
            report.confirmed_bottom = new_ref
            report.start_index = new_ref.index
            report.start_date = new_ref.date
        report.refresh_status = "post_in_range_candidate_ma5_replacement"
        report.diagnostics.setdefault("post_endpoint_replacements", []).append(
            {
                "old_index": old_ref.index,
                "old_date": old_ref.date,
                "old_ma5": None if pd.isna(df.iloc[old_ref.index].ma5) else float(df.iloc[old_ref.index].ma5),
                "new_index": new_ref.index,
                "new_date": new_ref.date,
                "new_ma5": None if pd.isna(df.iloc[new_ref.index].ma5) else float(df.iloc[new_ref.index].ma5),
                "candidate_type": candidate.candidate_type,
                "rule": "全局完成后，候选顶/底及前后各2根K中任一根的 MA5 满足上一线段两个端点对应转折K之间的 MA5 极值阈值，且候选满足四法则，则替换当前端点；多个候选满足时按候选顶/底自身 MA5 最大/最小选择。",
            }
        )
        candidate.status = "confirmed"
        candidate.rejected_reasons = []
        report.unconfirmed_candidates = [
            ev for ev in report.unconfirmed_candidates if ev.middle_index != candidate.middle_index
        ]
        refresh_report_summary(report)

    def _sync_previous_shared_endpoint(
        self, df: pd.DataFrame, previous: SegmentReport, old_ref: KRef, new_ref: KRef
    ) -> None:
        updated = False
        if previous.confirmed_top and previous.confirmed_top.index == old_ref.index and "top" in old_ref.role:
            previous.confirmed_top = KRef(new_ref.index, new_ref.date, new_ref.price, "confirmed_top")
            previous.end_index = new_ref.index
            previous.end_date = new_ref.date
            updated = True
        elif previous.confirmed_bottom and previous.confirmed_bottom.index == old_ref.index and "bottom" in old_ref.role:
            previous.confirmed_bottom = KRef(new_ref.index, new_ref.date, new_ref.price, "confirmed_bottom")
            previous.end_index = new_ref.index
            previous.end_date = new_ref.date
            updated = True
        if not updated:
            return
        previous.refresh_status = "post_shared_endpoint_sync"
        previous.diagnostics.setdefault("post_shared_endpoint_syncs", []).append(
            {
                "old_index": old_ref.index,
                "old_date": old_ref.date,
                "old_ma5": None if pd.isna(df.iloc[old_ref.index].ma5) else float(df.iloc[old_ref.index].ma5),
                "new_index": new_ref.index,
                "new_date": new_ref.date,
                "new_ma5": None if pd.isna(df.iloc[new_ref.index].ma5) else float(df.iloc[new_ref.index].ma5),
                "rule": "后一线段共享端点被线段范围内 MA5 更优候选替换，前一线段同步共享端点。",
            }
        )
        refresh_report_summary(previous)

    def _report_pivot_info(self, report: SegmentReport) -> PrevPivotInfo | None:
        if report.pivot_lower is None or report.pivot_upper is None:
            return None
        return PrevPivotInfo(
            lower=report.pivot_lower,
            upper=report.pivot_upper,
            confirm_index=report.pivot_confirm_k,
        )

    def _report_full_pivot_info(self, report: SegmentReport) -> PivotInfo | None:
        if (
            report.pivot_confirm_k is None
            or report.pivot_confirm_date is None
            or report.pivot_line is None
            or report.pivot_type is None
            or report.pivot_lower is None
            or report.pivot_upper is None
        ):
            return None
        return PivotInfo(
            confirm_index=report.pivot_confirm_k,
            confirm_date=report.pivot_confirm_date,
            pivot_line=report.pivot_line,
            pivot_type=report.pivot_type,
            lower=report.pivot_lower,
            upper=report.pivot_upper,
            start_patterns=report.start_pattern,
            formation_end_index=report.diagnostics.get("pivot_formation_end_index", report.pivot_confirm_k),
            black_k_indices=report.diagnostics.get("black_k_indices", []),
            extension_or_recomposition=report.extension_or_recomposition,
            has_multiple_pivots=bool(report.diagnostics.get("has_multiple_pivots")),
            reasoning=report.diagnostics.get("pivot_reasoning", []),
        )

    def _selected_endpoint_candidate(
        self, report: SegmentReport, direction: str
    ) -> CandidateEvaluation | None:
        endpoint = report.confirmed_top if direction == UP else report.confirmed_bottom
        if endpoint is None:
            return None
        pool = report.candidate_top if direction == UP else report.candidate_bottom
        for ev in pool:
            if ev.middle_index == endpoint.index:
                return ev
        return None

    def _analyze_direction_set(
        self,
        data: pd.DataFrame,
        direction: str,
        segment_no: int,
        start_index: int,
        previous_pivot: PrevPivotInfo | None,
        previous_endpoint: KRef | None,
        previous_turn_index: int | None = None,
    ) -> SegmentReport:
        if direction == "both":
            candidates = [
                self.analyze_one(data, UP, segment_no, start_index, previous_pivot, previous_endpoint, previous_turn_index),
                self.analyze_one(data, DOWN, segment_no, start_index, previous_pivot, previous_endpoint, previous_turn_index),
            ]
            return choose_better_report(candidates)
        return self.analyze_one(
            data,
            direction,
            segment_no,
            start_index,
            previous_pivot,
            previous_endpoint,
            previous_turn_index,
        )

    def analyze_one(
        self,
        df: pd.DataFrame,
        direction: str,
        segment_no: int = 1,
        start_index: int = 0,
        previous_pivot: PrevPivotInfo | None = None,
        previous_endpoint: KRef | None = None,
        previous_turn_index: int | None = None,
        prefer_recent_turn: bool = False,
        require_dynamic_survival: bool = False,
        apply_refresh: bool = False,
        end_selection: str = "earliest",
        finalize_output: bool = True,
        end_limit_index: int | None = None,
    ) -> SegmentReport:
        report = SegmentReport(
            segment_no=segment_no,
            segment_direction=direction,
            start_index=start_index,
            start_date=format_date(df.iloc[start_index].date) if 0 <= start_index < len(df) else None,
        )
        k0, pivot = self._find_k0_with_pivot(df, direction, start_index, previous_turn_index, previous_endpoint)
        report.K0_index = k0
        report.K0_date = format_date(df.iloc[k0].date) if k0 is not None else None
        if k0 is None:
            report.decision_status = "invalid"
            report.reasoning_summary = "无法确认中枢线：未找到独立 K0。"
            return report

        if pivot is None:
            report.decision_status = "needs_more_k"
            report.reasoning_summary = "无法确认线段中枢：每个 K0 后的唯一中枢线确认K候选均未满足起手三式或黑K约束。"
            return report

        self._fill_pivot_fields(report, pivot, df)
        tops = self._evaluate_candidates(df, TOP, direction, start_index, pivot, require_dynamic_survival)
        bottoms = self._evaluate_candidates(df, BOTTOM, direction, start_index, pivot, require_dynamic_survival)
        report.candidate_top = tops
        report.candidate_bottom = bottoms
        report.independent_2k_side = independent_side(tops, bottoms)

        start_eval = self._select_start_candidate(direction, previous_endpoint, pivot, tops, bottoms)
        end_eval = self._select_end_candidate(
            df,
            direction,
            start_index,
            pivot,
            start_eval,
            tops,
            bottoms,
            prefer_recent_turn=prefer_recent_turn,
            require_dynamic_survival=require_dynamic_survival,
            end_selection=end_selection,
            end_limit_index=end_limit_index,
        )

        if start_eval is None:
            missing = "未找到结构起点候选"
        else:
            missing = ""

        if end_eval is None:
            if missing:
                missing += "；"
            missing += "未找到可确认的结构终点候选"

        initial_cross = None
        if start_eval and end_eval:
            start_mid = start_eval.index if isinstance(start_eval, KRef) else start_eval.middle_index
            initial_cross = self._find_initial_cross(df, min(start_mid, end_eval.middle_index), max(start_mid, end_eval.middle_index))
            report.initial_cross = initial_cross

        if start_eval and end_eval and initial_cross:
            self._apply_confirmed_pair(df, report, direction, start_eval, end_eval)
            if apply_refresh:
                self._apply_dynamic_refresh(df, report, direction)
            report.ma34_check = True
            report.decision_status = "confirmed" if report.turn_k else "candidate"
        else:
            report.decision_status = "candidate"

        if report.decision_status != "confirmed":
            reasons = []
            if missing:
                reasons.append(missing)
            if start_eval and end_eval and not initial_cross:
                reasons.append("MA5/MA34 初始交叉点未确认")
            if not report.has_black_k:
                reasons.append("黑K约束未满足")
            report.reasoning_summary = "当前只能标记为候选顶底，不能确认线段顶底。原因：" + "；".join(reasons or ["关键条件不足"])
        else:
            top_txt = describe_ref(report.confirmed_top)
            bottom_txt = describe_ref(report.confirmed_bottom)
            turn_txt = describe_ref(report.turn_k)
            report.reasoning_summary = (
                f"已确认线段：方向={direction}，底={bottom_txt}，顶={top_txt}，"
                f"转折K={turn_txt}，中枢区间=[{report.pivot_lower:.3f}, {report.pivot_upper:.3f}]。"
            )

        if finalize_output:
            self._trim_candidates_for_report(report, end_eval)
            self._mark_unconfirmed(report, end_eval)
        return report

    def _find_k0(
        self, df: pd.DataFrame, direction: str, start_index: int, previous_pivot: PrevPivotInfo | None = None
    ) -> int | None:
        return self._find_next_k0(df, direction, start_index, previous_pivot)

    def _find_next_k0(
        self, df: pd.DataFrame, direction: str, start_index: int, previous_pivot: PrevPivotInfo | None
    ) -> int | None:
        for i in range(start_index, len(df)):
            if self._is_k0_candidate(df, direction, i, previous_pivot):
                return i
        return None

    def _is_k0_candidate(
        self,
        df: pd.DataFrame,
        direction: str,
        index: int,
        previous_pivot: PrevPivotInfo | None,
    ) -> bool:
        row = df.iloc[index]
        if pd.isna(row.ma5):
            return False
        positive_side = rule_gt(body_low(row), row.ma5) if direction == UP else rule_lt(body_high(row), row.ma5)
        if not positive_side:
            return False
        if previous_pivot and strict_overlap(row.low, row.high, previous_pivot.lower, previous_pivot.upper):
            return False
        if previous_pivot and index in previous_pivot.used_indexes:
            return False
        return True

    def _find_k0_with_pivot(
        self,
        df: pd.DataFrame,
        direction: str,
        start_index: int,
        previous_turn_index: int | None,
        previous_endpoint: KRef | None = None,
    ) -> tuple[int | None, PivotInfo | None]:
        same_segment_previous_pivot = None
        k0 = self._find_next_k0(df, direction, start_index, same_segment_previous_pivot)
        if k0 is None:
            return None, None
        k0_indexes = self._collect_k0_indexes(df, direction, k0, same_segment_previous_pivot)
        pivot_candidates = []
        active_zone: tuple[float, float] | None = None
        active_zone_k0: int | None = None
        candidate_scan_end = min(len(df), k0 + self.config.start_pattern_scan_bars + 1)
        for i in range(k0 + 1, candidate_scan_end):
            if not is_reverse_break_ma5(df, i, direction):
                continue
            nearest_k0 = self._nearest_k0_index(k0_indexes, i)
            if nearest_k0 is None:
                continue
            if active_zone is not None:
                row = df.iloc[i]
                if nearest_k0 == active_zone_k0 and strict_overlap(row.low, row.high, active_zone[0], active_zone[1]):
                    continue
            candidate = self._start_pattern_candidate(
                df,
                direction,
                start_index,
                nearest_k0,
                i,
                previous_turn_index,
                allow_empty=True,
            )
            if candidate is not None:
                pivot_candidates.append(candidate)
                if candidate["pattern_hits"]:
                    active_zone = (candidate["hypothetical_lower"], candidate["hypothetical_upper"])
                    active_zone_k0 = candidate["k0_index"]

        for pos, candidate in enumerate(pivot_candidates):
            next_candidate = next(
                (
                    later
                    for later in pivot_candidates[pos + 1 :]
                    if later["k0_index"] != candidate["k0_index"]
                ),
                None,
            )
            pivot = self._confirm_pivot(df, direction, candidate, next_candidate, start_index, previous_endpoint)
            if pivot is not None:
                return candidate["k0_index"], pivot
        return k0, None

    def _collect_k0_indexes(
        self,
        df: pd.DataFrame,
        direction: str,
        start_index: int,
        previous_pivot: PrevPivotInfo | None,
    ) -> list[int]:
        return [
            i
            for i in range(start_index, len(df))
            if self._is_k0_candidate(df, direction, i, previous_pivot)
        ]

    def _nearest_k0_index(self, k0_indexes: list[int], confirm_index: int) -> int | None:
        nearest = None
        for k0 in k0_indexes:
            if k0 >= confirm_index:
                break
            nearest = k0
        return nearest

    def _start_pattern_candidate(
        self,
        df: pd.DataFrame,
        direction: str,
        start_index: int,
        k0: int,
        confirm_index: int,
        previous_turn_index: int | None,
        allow_empty: bool = False,
    ) -> dict | None:
        pivot_line = calc_pivot_line(df, confirm_index, direction)
        pattern_hits = self._start_patterns(
            df,
            direction,
            start_index,
            k0,
            confirm_index,
            pivot_line,
            previous_turn_index,
        )
        if not pattern_hits:
            if not allow_empty:
                return None
            pattern_end = None
            lower = upper = pivot_type = zone_reason = None
        else:
            pattern_end = min(pattern_hits.values())
            lower, upper, pivot_type, zone_reason = self._pivot_zone(
                df,
                direction,
                confirm_index,
                pattern_end,
                pivot_line,
                previous_turn_index=previous_turn_index,
            )
        return {
            "k0_index": k0,
            "confirm_index": confirm_index,
            "pivot_line": pivot_line,
            "pattern_hits": pattern_hits,
            "pattern_end": pattern_end,
            "previous_turn_index": previous_turn_index,
            "hypothetical_lower": lower,
            "hypothetical_upper": upper,
            "hypothetical_pivot_type": pivot_type,
            "hypothetical_zone_reason": zone_reason,
        }

    def _confirm_pivot(
        self,
        df: pd.DataFrame,
        direction: str,
        candidate: dict,
        next_candidate: dict | None,
        start_index: int,
        previous_endpoint: KRef | None,
    ) -> PivotInfo | None:
        confirm_index = candidate["confirm_index"]
        pivot_line = candidate["pivot_line"]
        pattern_hits = candidate["pattern_hits"]
        if not pattern_hits:
            return None
        pattern_end = candidate["pattern_end"]
        formation_end = pattern_end
        if next_candidate is not None and next_candidate["confirm_index"] <= formation_end:
            return None

        black_scan_start = min(len(df) - 1, confirm_index + 1)
        black_scan_end = (
            next_candidate["confirm_index"] - 1
            if next_candidate is not None
            else min(len(df) - 1, confirm_index + self.config.start_pattern_scan_bars)
        )
        black_indices = self._black_k_indices(df, direction, black_scan_start, black_scan_end)
        if not black_indices:
            return None
        visual_scan_end = (
            next_candidate["confirm_index"] - 1
            if next_candidate is not None
            else min(len(df) - 1, confirm_index + self.config.start_pattern_scan_bars)
        )
        lower, upper, pivot_type, zone_reason = self._pivot_zone(
            df,
            direction,
            confirm_index,
            formation_end,
            pivot_line,
            visual_scan_end=visual_scan_end,
            previous_turn_index=candidate.get("previous_turn_index"),
        )
        extension_or_recomposition = "none"
        if next_candidate is not None and next_candidate["confirm_index"] > formation_end:
            provisional_pivot = self._build_pivot_info(
                df=df,
                candidate=candidate,
                pivot_line=pivot_line,
                pivot_type=pivot_type,
                lower=lower,
                upper=upper,
                pattern_hits=pattern_hits,
                pattern_end=pattern_end,
                black_indices=black_indices,
                extension_or_recomposition="none",
                has_multiple_pivots=False,
                reasoning=zone_reason,
            )
            endpoint_limit = max(formation_end, next_candidate["confirm_index"] - 1)
            if not self._pivot_can_confirm_endpoint(
                df,
                direction,
                start_index,
                previous_endpoint,
                provisional_pivot,
                endpoint_limit,
            ):
                if self._pivot_recomposition_required(
                    df,
                    direction,
                    candidate,
                    next_candidate,
                    lower,
                    upper,
                    pivot_type,
                ):
                    next_candidate["recomposed_from_previous"] = True
                    return None
                if self._pivot_extension_triggered(df, next_candidate, lower, upper):
                    extension_or_recomposition = "extension"
                    zone_reason = zone_reason + [
                        "当前中枢成立后出现新的中枢线确认K候选，但未满足重组条件；按中枢延伸处理，原中枢线和中枢区间不变。"
                    ]
        return self._build_pivot_info(
            df=df,
            candidate=candidate,
            pivot_line=pivot_line,
            pivot_type=pivot_type,
            lower=lower,
            upper=upper,
            pattern_hits=pattern_hits,
            pattern_end=pattern_end,
            black_indices=black_indices,
            extension_or_recomposition=extension_or_recomposition,
            has_multiple_pivots=bool(
                extension_or_recomposition != "none" or candidate.get("recomposed_from_previous")
            ),
            reasoning=zone_reason,
        )

    def _build_pivot_info(
        self,
        df: pd.DataFrame,
        candidate: dict,
        pivot_line: float,
        pivot_type: str,
        lower: float,
        upper: float,
        pattern_hits: dict[str, int],
        pattern_end: int,
        black_indices: list[int],
        extension_or_recomposition: str,
        has_multiple_pivots: bool,
        reasoning: list[str],
    ) -> PivotInfo:
        confirm_index = candidate["confirm_index"]
        return PivotInfo(
            confirm_index=confirm_index,
            confirm_date=format_date(df.iloc[confirm_index].date),
            pivot_line=float(pivot_line),
            pivot_type=pivot_type,
            lower=float(lower),
            upper=float(upper),
            start_patterns=[name for name, end in pattern_hits.items() if end == pattern_end],
            formation_end_index=pattern_end,
            black_k_indices=black_indices,
            extension_or_recomposition=extension_or_recomposition,
            has_multiple_pivots=has_multiple_pivots,
            reasoning=reasoning,
        )

    def _pivot_can_confirm_endpoint(
        self,
        df: pd.DataFrame,
        direction: str,
        start_index: int,
        previous_endpoint: KRef | None,
        pivot: PivotInfo,
        end_limit_index: int,
    ) -> bool:
        tops = self._evaluate_candidates(df, TOP, direction, start_index, pivot, require_dynamic_survival=False)
        bottoms = self._evaluate_candidates(df, BOTTOM, direction, start_index, pivot, require_dynamic_survival=False)
        start_eval = self._select_start_candidate(direction, previous_endpoint, pivot, tops, bottoms)
        end_eval = self._select_end_candidate(
            df,
            direction,
            start_index,
            pivot,
            start_eval,
            tops,
            bottoms,
            prefer_recent_turn=False,
            require_dynamic_survival=False,
            end_selection="earliest",
            end_limit_index=end_limit_index,
        )
        if start_eval is None or end_eval is None:
            return False
        start_mid = start_eval.index if isinstance(start_eval, KRef) else start_eval.middle_index
        return self._find_initial_cross(df, min(start_mid, end_eval.middle_index), max(start_mid, end_eval.middle_index)) is not None

    def _pivot_recomposition_required(
        self,
        df: pd.DataFrame,
        direction: str,
        old_candidate: dict,
        new_candidate: dict,
        old_lower: float,
        old_upper: float,
        old_pivot_type: str,
    ) -> bool:
        if not self._pivot_extension_triggered(df, new_candidate, old_lower, old_upper):
            return False

        if not self._pivot_line_advances(direction, old_candidate["pivot_line"], new_candidate["pivot_line"]):
            return False

        old_row = df.iloc[old_candidate["confirm_index"]]
        new_row = df.iloc[new_candidate["confirm_index"]]
        if direction == UP:
            body_breaks = rule_gt(body_low(new_row), body_low(old_row))
        else:
            body_breaks = rule_lt(body_high(new_row), body_high(old_row))
        if body_breaks:
            return True

        if old_pivot_type != "非肉眼可见中枢":
            return False
        return self._candidate_can_form_visual_pivot(df, direction, new_candidate)

    def _pivot_line_advances(self, direction: str, old_line: float, new_line: float) -> bool:
        if direction == UP:
            return rule_gt(new_line, old_line)
        return rule_lt(new_line, old_line)

    def _pivot_extension_triggered(
        self,
        df: pd.DataFrame,
        candidate: dict,
        pivot_lower: float,
        pivot_upper: float,
    ) -> bool:
        start = candidate["confirm_index"]
        end = candidate["pattern_end"]
        if end is None:
            end = min(len(df) - 1, start + self.config.start_pattern_scan_bars)
        for i in range(start, min(end, len(df) - 1) + 1):
            if kline_overlaps_zone(df.iloc[i], pivot_lower, pivot_upper):
                return True
        return False

    def _candidate_can_form_visual_pivot(self, df: pd.DataFrame, direction: str, candidate: dict) -> bool:
        if not candidate["pattern_hits"]:
            return False
        confirm_index = candidate["confirm_index"]
        pattern_end = candidate["pattern_end"]
        formation_end = pattern_end
        black_scan_end = min(len(df) - 1, confirm_index + self.config.start_pattern_scan_bars)
        black_indices = self._black_k_indices(df, direction, confirm_index + 1, black_scan_end)
        if not black_indices:
            return False
        pivot_line = candidate["pivot_line"]
        visual_scan_end = min(len(df) - 1, confirm_index + self.config.start_pattern_scan_bars)
        _, _, pivot_type, _ = self._pivot_zone(
            df,
            direction,
            confirm_index,
            formation_end,
            pivot_line,
            visual_scan_end=visual_scan_end,
            previous_turn_index=candidate.get("previous_turn_index"),
        )
        return pivot_type == "肉眼可见中枢"

    def _start_patterns(
        self,
        df: pd.DataFrame,
        direction: str,
        start_index: int,
        k0: int,
        confirm_index: int,
        pivot_line: float,
        previous_turn_index: int | None,
    ) -> dict[str, int]:
        scan_end = min(len(df) - 1, confirm_index + self.config.start_pattern_scan_bars)
        hits: dict[str, int] = {}
        anti_positive = self._anti_positive_cross(df, direction, confirm_index, scan_end, pivot_line)
        if anti_positive is not None:
            hits["反正两穿"] = anti_positive
        three = self._three_strokes(df, direction, start_index, confirm_index, scan_end)
        if three is not None:
            hits["三笔"] = three
        five = self._five_k_overlap(
            df,
            direction,
            start_index,
            k0,
            confirm_index,
            scan_end,
            pivot_line,
            previous_turn_index,
        )
        if five is not None:
            hits["5K重叠"] = five
        return hits

    def _anti_positive_cross(
        self, df: pd.DataFrame, direction: str, confirm_index: int, scan_end: int, pivot_line: float
    ) -> int | None:
        anti_seen = False
        anti_body_low = None
        anti_body_high = None
        for i in range(confirm_index, scan_end + 1):
            row = df.iloc[i]
            cur_body_low = body_low(row)
            cur_body_high = body_high(row)
            if direction == UP:
                if not anti_seen and rule_lt(cur_body_low, pivot_line):
                    anti_seen = True
                    anti_body_low = cur_body_low
                    anti_body_high = cur_body_high
                elif (
                    anti_seen
                    and rule_gt(cur_body_high, pivot_line)
                    and rule_gt(cur_body_high, anti_body_high)
                    and rule_gt(cur_body_low, anti_body_low)
                ):
                    return i
            else:
                if not anti_seen and rule_gt(cur_body_high, pivot_line):
                    anti_seen = True
                    anti_body_low = cur_body_low
                    anti_body_high = cur_body_high
                elif (
                    anti_seen
                    and rule_lt(cur_body_low, pivot_line)
                    and rule_lt(cur_body_high, anti_body_high)
                    and rule_lt(cur_body_low, anti_body_low)
                ):
                    return i
        return None

    def _three_strokes(
        self, df: pd.DataFrame, direction: str, start_index: int, confirm_index: int, scan_end: int
    ) -> int | None:
        for first in range(start_index, min(confirm_index, scan_end) + 1):
            best = df.iloc[first].low if direction == UP else df.iloc[first].high
            count = 1
            confirm_included = first == confirm_index
            for i in range(first + 1, scan_end + 1):
                value = df.iloc[i].low if direction == UP else df.iloc[i].high
                improves = rule_lt(value, best) if direction == UP else rule_gt(value, best)
                if not improves:
                    continue
                best = value
                count += 1
                if i == confirm_index:
                    confirm_included = True
                if i >= confirm_index and confirm_included and count >= 3:
                    return i
        return None

    def _five_k_overlap(
        self,
        df: pd.DataFrame,
        direction: str,
        start_index: int,
        k0: int,
        confirm_index: int,
        scan_end: int,
        pivot_line: float,
        previous_turn_index: int | None,
    ) -> int | None:
        cutoff = previous_turn_index if previous_turn_index is not None else start_index
        for first in self._first_k_overlap_candidates(df, direction, k0, cutoff, scan_end):
            end = self._five_k_overlap_end(df, first, confirm_index, scan_end)
            if end is not None:
                return end
        return None

    def _five_k_overlap_end(
        self, df: pd.DataFrame, first: int, confirm_index: int, scan_end: int
    ) -> int | None:
        breaker = first + 1
        if breaker >= len(df) or breaker > scan_end:
            return None
        low, high = overlap_zone_with_fallback(df.iloc[first], df.iloc[breaker])
        if not kline_overlaps_zone(df.iloc[confirm_index], low, high):
            return None
        count = 0
        gap_counted = 0
        for i in range(first, scan_end + 1):
            row = df.iloc[i]
            if kline_overlaps_zone(row, low, high):
                count += 1
            elif (
                count >= 4
                and gap_counted == 0
                and i > first
                and kline_gaps_over_zone(df.iloc[i - 1], row, low, high)
            ):
                count += 1
                gap_counted += 1
            if count >= 5:
                return i
        return None

    def _first_k_overlap_candidates(
        self, df: pd.DataFrame, direction: str, k0: int, cutoff: int, scan_end: int
    ) -> list[int]:
        candidates = []
        start = max(cutoff, 0)
        end = min(k0, scan_end - 1)
        for s in range(start, end + 1):
            if s + 1 >= len(df) or s + 1 > scan_end:
                continue
            if direction == UP and rule_gt(df.iloc[s].high, df.iloc[s + 1].high):
                candidates.append(s)
            elif direction == DOWN and rule_lt(df.iloc[s].low, df.iloc[s + 1].low):
                candidates.append(s)
        return candidates

    def _nearest_k0_before_confirm(
        self, df: pd.DataFrame, direction: str, k0: int, confirm_index: int
    ) -> int | None:
        for i in range(confirm_index - 1, k0 - 1, -1):
            row = df.iloc[i]
            if pd.isna(row.ma5):
                continue
            if direction == UP and rule_gt(body_low(row), row.ma5):
                return i
            if direction == DOWN and rule_lt(body_high(row), row.ma5):
                return i
        return None

    def _find_first_k_for_overlap(
        self, df: pd.DataFrame, direction: str, nearest_k0: int, cutoff: int, scan_end: int
    ) -> int | None:
        for s in range(min(nearest_k0, scan_end - 1), max(cutoff, 0) - 1, -1):
            if s + 1 >= len(df) or s + 1 > scan_end:
                continue
            if direction == UP and rule_gt(df.iloc[s].high, df.iloc[s + 1].high):
                return s
            if direction == DOWN and rule_lt(df.iloc[s].low, df.iloc[s + 1].low):
                return s
        return None

    def _black_k_indices(self, df: pd.DataFrame, direction: str, start: int, end: int) -> list[int]:
        for i in range(max(start, 1), min(end, len(df) - 1) + 1):
            if kline_strict_cross_ma(df.iloc[i], "ma5") and not is_positive_gap_ma5(df, i, direction):
                return [i]
        return []

    def _pivot_zone(
        self,
        df: pd.DataFrame,
        direction: str,
        confirm_index: int,
        formation_end: int,
        pivot_line: float,
        visual_scan_end: int | None = None,
        previous_turn_index: int | None = None,
    ) -> tuple[float, float, str, list[str]]:
        visual_end = formation_end if visual_scan_end is None else visual_scan_end
        visible = self._is_visual_pivot(df, direction, confirm_index, visual_end)
        reason = []
        if visible:
            rail = self._visual_rail(df, direction, confirm_index, visual_end)
            if rail is not None:
                reason.append("按肉眼可见中枢取轨：新的中枢线确认K出现前，确认K之后 MA5 出现反向越过确认K对应 MA5。")
                if direction == UP:
                    return normalize_zone(pivot_line, rail) + ("肉眼可见中枢", reason)
                return normalize_zone(rail, pivot_line) + ("肉眼可见中枢", reason)

        rail = self._non_visual_rail(df, direction, confirm_index, pivot_line, previous_turn_index)
        if self._has_three_higher_high_records(df, confirm_index, formation_end):
            reason.append("按非肉眼可见中枢取轨：未满足肉眼可见条件，按确认K向前至前一转折K的最近相邻两K重叠规则取轨。")
        else:
            reason.append("按非肉眼可见中枢取轨：未满足肉眼可见条件，按确认K向前至前一转折K的最近相邻两K重叠规则取轨。")
        if direction == UP:
            return normalize_zone(pivot_line, rail) + ("非肉眼可见中枢", reason)
        return normalize_zone(rail, pivot_line) + ("非肉眼可见中枢", reason)

    def _is_visual_pivot(self, df: pd.DataFrame, direction: str, confirm_index: int, formation_end: int) -> bool:
        confirm_ma5 = df.iloc[confirm_index].ma5
        if pd.isna(confirm_ma5):
            return False
        for i in range(confirm_index + 1, min(formation_end, len(df) - 1) + 1):
            ma5 = df.iloc[i].ma5
            if pd.isna(ma5):
                continue
            if direction == UP and rule_lt(ma5, confirm_ma5):
                return True
            if direction == DOWN and rule_gt(ma5, confirm_ma5):
                return True
        return False

    def _has_three_higher_high_records(self, df: pd.DataFrame, confirm_index: int, formation_end: int) -> bool:
        record_high = df.iloc[confirm_index].high
        record_count = 0
        for i in range(confirm_index + 1, min(formation_end, len(df) - 1) + 1):
            high = df.iloc[i].high
            if rule_gt(high, record_high):
                record_high = high
                record_count += 1
                if record_count >= 3:
                    return True
        return False

    def _visual_rail(self, df: pd.DataFrame, direction: str, confirm_index: int, formation_end: int) -> float | None:
        values = df.iloc[confirm_index : formation_end + 1]["ma5"]
        turns = []
        for i in range(max(confirm_index + 1, 1), min(formation_end, len(df) - 2) + 1):
            prev_v, cur_v, next_v = df.iloc[i - 1].ma5, df.iloc[i].ma5, df.iloc[i + 1].ma5
            if pd.isna(prev_v) or pd.isna(cur_v) or pd.isna(next_v):
                continue
            if direction == UP and rule_ge(cur_v, prev_v) and rule_ge(cur_v, next_v):
                turns.append(float(cur_v))
            elif direction == DOWN and rule_le(cur_v, prev_v) and rule_le(cur_v, next_v):
                turns.append(float(cur_v))
            if len(turns) >= 2:
                break
        if len(turns) >= 2:
            return min(turns) if direction == UP else max(turns)
        if turns:
            return turns[0]
        if not values.dropna().empty:
            return float(values.dropna().median())
        return None

    def _non_visual_rail(
        self,
        df: pd.DataFrame,
        direction: str,
        confirm_index: int,
        pivot_line: float,
        previous_turn_index: int | None = None,
    ) -> float:
        scan_start = (
            max(0, previous_turn_index)
            if previous_turn_index is not None and previous_turn_index < confirm_index
            else max(0, confirm_index - self.config.pivot_zone_scan_bars)
        )
        rail = None
        for i in range(confirm_index - 1, scan_start - 1, -1):
            a, b = df.iloc[i], df.iloc[i + 1]
            low = max(a.low, b.low)
            high = min(a.high, b.high)
            if rule_gt(low, high):
                continue
            if direction == UP and rule_gt(a.high, pivot_line) and rule_gt(b.high, pivot_line):
                rail = float(min(a.high, b.high))
                continue
            if direction == DOWN and rule_lt(a.low, pivot_line) and rule_lt(b.low, pivot_line):
                rail = float(max(a.low, b.low))
                continue
        return rail if rail is not None else float(pivot_line)

    def _evaluate_candidates(
        self,
        df: pd.DataFrame,
        candidate_type: str,
        direction: str,
        start_index: int,
        pivot: PivotInfo,
        require_dynamic_survival: bool,
    ) -> list[CandidateEvaluation]:
        candidates: list[CandidateEvaluation] = []
        for middle in range(max(start_index + 1, 1), len(df) - 1):
            if candidate_type == TOP:
                if not (
                    rule_ge(df.iloc[middle].high, df.iloc[middle - 1].high)
                    and rule_ge(df.iloc[middle].high, df.iloc[middle + 1].high)
                ):
                    continue
                price = float(df.iloc[middle].high)
            else:
                if not (
                    rule_le(df.iloc[middle].low, df.iloc[middle - 1].low)
                    and rule_le(df.iloc[middle].low, df.iloc[middle + 1].low)
                ):
                    continue
                price = float(df.iloc[middle].low)

            ev = CandidateEvaluation(
                candidate_type=candidate_type,
                middle_index=middle,
                middle_date=format_date(df.iloc[middle].date),
                indexes=[middle - 1, middle, middle + 1],
                price=price,
            )
            ev.checks["independent_2k"] = self._independent_2k(df, ev, pivot)
            ev.checks["ma34_side_and_strict_ma5"] = self._ma34_check(df, middle, candidate_type)
            pivot_black_k = pivot.black_k_indices[0] if pivot.black_k_indices else None
            ev.checks["not_black_k"] = pivot_black_k is None or pivot_black_k != middle
            ev.checks["after_pivot_for_endpoint"] = middle > pivot.confirm_index
            ev.checks["before_pivot_for_start"] = middle < pivot.confirm_index
            candidates.append(ev)
        self._assign_unique_turn_ks(df, candidates, candidate_type, direction, start_index, require_dynamic_survival)
        for ev in candidates:
            ev.rejected_reasons.extend(candidate_reasons(ev))
        return candidates

    def _independent_2k(self, df: pd.DataFrame, ev: CandidateEvaluation, pivot: PivotInfo) -> bool:
        if ev.candidate_type == TOP:
            middle_ok = rule_ge(df.iloc[ev.middle_index].low, pivot.upper)
            return middle_ok and sum(rule_ge(df.iloc[i].low, pivot.upper) for i in ev.indexes) >= 2
        middle_ok = rule_le(df.iloc[ev.middle_index].high, pivot.lower)
        return middle_ok and sum(rule_le(df.iloc[i].high, pivot.lower) for i in ev.indexes) >= 2

    def _ma34_check(self, df: pd.DataFrame, middle: int, candidate_type: str) -> bool:
        row = df.iloc[middle]
        if pd.isna(row.ma34) or pd.isna(row.ma5):
            return False
        if candidate_type == TOP:
            return rule_gt(body_high(row), row.ma34) and rule_gt(row.ma5, row.ma34)
        return rule_lt(body_low(row), row.ma34) and rule_lt(row.ma5, row.ma34)

    def _pair_independent_2k(
        self,
        df: pd.DataFrame,
        start_eval: CandidateEvaluation | KRef | None,
        end_eval: CandidateEvaluation,
        pivot: PivotInfo,
    ) -> bool:
        return self._endpoint_independent_2k(df, start_eval, pivot) or self._endpoint_independent_2k(
            df, end_eval, pivot
        )

    def _endpoint_independent_2k(
        self, df: pd.DataFrame, endpoint: CandidateEvaluation | KRef | None, pivot: PivotInfo
    ) -> bool:
        if endpoint is None:
            return False
        if isinstance(endpoint, CandidateEvaluation):
            candidate_type = endpoint.candidate_type
            indexes = endpoint.indexes
        else:
            if is_anchor_ref(endpoint):
                return True
            candidate_type = TOP if "top" in endpoint.role else BOTTOM if "bottom" in endpoint.role else None
            indexes = [endpoint.index - 1, endpoint.index, endpoint.index + 1]
        if candidate_type is None or any(i < 0 or i >= len(df) for i in indexes):
            return False
        middle_index = endpoint.middle_index if isinstance(endpoint, CandidateEvaluation) else endpoint.index
        if candidate_type == TOP:
            middle_ok = rule_ge(df.iloc[middle_index].low, pivot.upper)
            return middle_ok and sum(rule_ge(df.iloc[i].low, pivot.upper) for i in indexes) >= 2
        middle_ok = rule_le(df.iloc[middle_index].high, pivot.lower)
        return middle_ok and sum(rule_le(df.iloc[i].high, pivot.lower) for i in indexes) >= 2

    def _assign_unique_turn_ks(
        self,
        df: pd.DataFrame,
        candidates: list[CandidateEvaluation],
        candidate_type: str,
        direction: str,
        start_index: int,
        require_dynamic_survival: bool,
    ) -> None:
        turns = self._independent_turn_k_indices(df, candidate_type, start_index)
        turn_set = set(turns)
        for ev in candidates:
            ev.turn_k_index = None
            ev.turn_k_date = None
            ev.checks["not_turn_k_itself"] = ev.middle_index not in turn_set
            ev.checks["has_turn_k"] = False
            ev.checks["dynamic_ma5_survives"] = not require_dynamic_survival
            if not ev.checks["not_turn_k_itself"]:
                ev.notes.append("该K已被识别为转折K，不能作为顶/底候选。")

        available = sorted(
            [ev for ev in candidates if ev.checks.get("not_turn_k_itself")],
            key=lambda ev: ev.middle_index,
        )
        for turn in turns:
            target = self._nearest_unmatched_candidate_before_turn(available, turn)
            if target is None:
                continue
            for ev in available:
                if ev.middle_index >= target.middle_index:
                    break
                if ev.turn_k_index is None:
                    ev.notes.append("更晚候选已匹配转折K；当前更早候选不再参与后续转折K匹配。")
            target.turn_k_index = turn
            target.turn_k_date = format_date(df.iloc[turn].date)
            target.checks["has_turn_k"] = True
            if require_dynamic_survival:
                target.checks["dynamic_ma5_survives"] = not self._has_later_ma5_growth(
                    df, direction, start_index, turn
                )
            else:
                target.checks["dynamic_ma5_survives"] = True
            available = [ev for ev in available if ev.middle_index >= target.middle_index]

    def _independent_turn_k_indices(
        self, df: pd.DataFrame, candidate_type: str, start_index: int
    ) -> list[int]:
        turns: list[int] = []
        seen: set[int] = set()
        for i in range(max(start_index + 1, 1), len(df)):
            row = df.iloc[i]
            if pd.isna(row.ma5):
                continue
            if candidate_type == TOP:
                reverse = rule_lt(body_low(row), row.ma5)
                not_growing = rule_le(row.ma5, df.iloc[start_index:i]["ma5"].max())
            else:
                reverse = rule_gt(body_high(row), row.ma5)
                not_growing = rule_ge(row.ma5, df.iloc[start_index:i]["ma5"].min())
            if reverse and bool(not_growing):
                if self._turn_is_extreme(df, i, candidate_type, start_index):
                    turn = i + 1 if i + 1 < len(df) else None
                else:
                    turn = i
                if turn is not None and turn not in seen:
                    turns.append(turn)
                    seen.add(turn)
        return turns

    def _nearest_unmatched_candidate_before_turn(
        self, candidates: list[CandidateEvaluation], turn_index: int
    ) -> CandidateEvaluation | None:
        nearest = None
        for ev in candidates:
            if ev.turn_k_index is not None:
                continue
            if ev.middle_index >= turn_index:
                break
            nearest = ev
        return nearest

    def _turn_is_extreme(self, df: pd.DataFrame, turn_index: int, candidate_type: str, start_index: int) -> bool:
        window = df.iloc[start_index : turn_index + 1]
        if candidate_type == TOP:
            return rule_ge(df.iloc[turn_index].high, window["high"].max())
        return rule_le(df.iloc[turn_index].low, window["low"].min())

    def _has_later_ma5_growth(self, df: pd.DataFrame, direction: str, start_index: int, turn_index: int) -> bool:
        for i in range(turn_index + 1, len(df)):
            if direction == UP:
                previous = df.iloc[start_index:i]["ma5"].max()
                if pd.notna(df.iloc[i].ma5) and rule_gt(df.iloc[i].ma5, previous):
                    return True
            else:
                previous = df.iloc[start_index:i]["ma5"].min()
                if pd.notna(df.iloc[i].ma5) and rule_lt(df.iloc[i].ma5, previous):
                    return True
        return False

    def _select_start_candidate(
        self,
        direction: str,
        previous_endpoint: KRef | None,
        pivot: PivotInfo,
        tops: list[CandidateEvaluation],
        bottoms: list[CandidateEvaluation],
    ) -> CandidateEvaluation | KRef | None:
        if previous_endpoint:
            return previous_endpoint
        pool = bottoms if direction == UP else tops
        valid = [
            ev
            for ev in pool
            if ev.middle_index < pivot.confirm_index
            and ev.checks.get("ma34_side_and_strict_ma5")
            and ev.checks.get("not_black_k")
            and ev.checks.get("not_turn_k_itself")
        ]
        if not valid:
            return None
        if direction == UP:
            return min(valid, key=lambda ev: (ev.price, -ev.middle_index))
        return max(valid, key=lambda ev: (ev.price, ev.middle_index))

    def _select_end_candidate(
        self,
        df: pd.DataFrame,
        direction: str,
        start_index: int,
        pivot: PivotInfo,
        start_eval: CandidateEvaluation | KRef | None,
        tops: list[CandidateEvaluation],
        bottoms: list[CandidateEvaluation],
        prefer_recent_turn: bool,
        require_dynamic_survival: bool,
        end_selection: str,
        end_limit_index: int | None,
    ) -> CandidateEvaluation | None:
        pool = tops if direction == UP else bottoms
        valid = []
        for ev in pool:
            if ev.middle_index <= pivot.confirm_index:
                continue
            if ev.middle_index <= pivot.formation_end_index:
                ev.rejected_reasons.append("候选在中枢成立前，不能作为确认端点。")
                continue
            if end_limit_index is not None and ev.middle_index > end_limit_index:
                continue
            ev.checks["pair_independent_2k"] = self._pair_independent_2k(df, start_eval, ev, pivot)
            ev.checks["structure_perfect"] = self._structure_perfect(df, direction, start_eval, ev, pivot)
            checks = [
                ev.checks.get("pair_independent_2k"),
                ev.checks.get("ma34_side_and_strict_ma5"),
                ev.checks.get("not_black_k"),
                ev.checks.get("not_turn_k_itself"),
                ev.checks.get("structure_perfect"),
                ev.checks.get("has_turn_k"),
            ]
            if require_dynamic_survival:
                checks.append(ev.checks.get("dynamic_ma5_survives"))
            if all(checks):
                ev.checks["has_initial_cross"] = self._candidate_has_initial_cross(df, start_eval, ev)
            if all(checks) and ev.checks.get("has_initial_cross"):
                valid.append(ev)
            elif ev.checks.get("pair_independent_2k") is False:
                ev.rejected_reasons.append("顶底组合两侧均未满足独立2K")
            elif ev.checks.get("structure_perfect") is False:
                ev.rejected_reasons.append("结构完美不成立：顶底未严格位于线段中枢两侧或结构顺序错误")
            elif all(checks) and ev.checks.get("has_initial_cross") is False:
                ev.rejected_reasons.append("顶底组合之间缺少MA5/MA34初始交叉点")
        if not valid:
            return None
        valid = self._keep_nearest_candidate_per_turn_k(valid)
        if end_selection == "bounded_extreme":
            selected = self._select_bounded_extreme(direction, valid, bottoms if direction == UP else tops)
            if selected is None:
                return None
            for ev in valid:
                if ev.middle_index == selected.middle_index:
                    continue
                boundary_note = (
                    "满足候选条件，但反向确认端点已先出现，当前端点不再延伸。"
                    if ev.middle_index > selected.middle_index
                    else "满足候选条件，但被反向确认端点出现前的更极值确认端点替代。"
                )
                ev.notes.append(boundary_note)
            return selected
        if end_selection == "extreme":
            if direction == UP:
                valid.sort(key=lambda ev: (ev.price, ev.turn_k_index or -1, ev.middle_index), reverse=True)
            else:
                valid.sort(key=lambda ev: (ev.price, -(ev.turn_k_index or -1), -ev.middle_index))
        else:
            valid.sort(key=lambda ev: (ev.turn_k_index or -1, ev.middle_index), reverse=prefer_recent_turn)
        selected = valid[0]
        for ev in valid[1:]:
            if end_selection == "extreme":
                if direction == UP and rule_lt(ev.price, selected.price):
                    ev.notes.append("满足候选条件，但当前顶已延伸到更高的确认顶。")
                elif direction == DOWN and rule_gt(ev.price, selected.price):
                    ev.notes.append("满足候选条件，但当前底已延伸到更低的确认底。")
                else:
                    ev.notes.append("满足候选条件，但极值相同或更晚的端点被当前确认端点替代。")
            elif prefer_recent_turn:
                ev.notes.append("满足候选条件，但被转折K更近的候选替代。")
            else:
                ev.notes.append("满足候选条件，但历史连续扫描选择更早闭合的线段。")
        return selected

    def _keep_nearest_candidate_per_turn_k(
        self, candidates: list[CandidateEvaluation]
    ) -> list[CandidateEvaluation]:
        best_by_turn: dict[int | None, CandidateEvaluation] = {}
        removed: list[CandidateEvaluation] = []
        for ev in candidates:
            key = ev.turn_k_index
            if key is None:
                unique_key = -ev.middle_index - 1
                best_by_turn[unique_key] = ev
                continue
            current = best_by_turn.get(key)
            if current is None:
                best_by_turn[key] = ev
                continue
            if self._is_nearer_to_turn_k(ev, current):
                removed.append(current)
                best_by_turn[key] = ev
            else:
                removed.append(ev)
        for ev in removed:
            ev.notes.append("满足候选条件，但与其他候选共用同一转折K，距离转折K更远。")
        return list(best_by_turn.values())

    def _is_nearer_to_turn_k(self, challenger: CandidateEvaluation, current: CandidateEvaluation) -> bool:
        if challenger.turn_k_index is None:
            return False
        if current.turn_k_index is None:
            return True
        challenger_distance = abs(challenger.turn_k_index - challenger.middle_index)
        current_distance = abs(current.turn_k_index - current.middle_index)
        if challenger_distance != current_distance:
            return challenger_distance < current_distance
        return challenger.middle_index > current.middle_index

    def _candidate_has_initial_cross(
        self,
        df: pd.DataFrame,
        start_eval: CandidateEvaluation | KRef | None,
        end_eval: CandidateEvaluation,
    ) -> bool:
        if start_eval is None:
            return False
        start_mid = start_eval.index if isinstance(start_eval, KRef) else start_eval.middle_index
        return self._find_initial_cross(df, min(start_mid, end_eval.middle_index), max(start_mid, end_eval.middle_index)) is not None

    def _structure_perfect(
        self,
        df: pd.DataFrame,
        direction: str,
        start_eval: CandidateEvaluation | KRef | None,
        end_eval: CandidateEvaluation,
        pivot: PivotInfo,
    ) -> bool:
        if start_eval is None:
            return False
        start_type = endpoint_type(start_eval)
        end_type = endpoint_type(end_eval)
        start_index = start_eval.index if isinstance(start_eval, KRef) else start_eval.middle_index

        if direction == UP:
            if start_type != BOTTOM or end_type != TOP:
                return False
            if not (start_index < pivot.confirm_index and end_eval.middle_index > pivot.formation_end_index):
                return False
        else:
            if start_type != TOP or end_type != BOTTOM:
                return False
            if not (start_index < pivot.confirm_index and end_eval.middle_index > pivot.formation_end_index):
                return False

        candidate_low, candidate_high = endpoint_low_high(df, end_eval)

        if not bool(end_eval.checks.get("independent_2k")) and not pivot.has_multiple_pivots:
            if direction == UP:
                return rule_gt(candidate_high, pivot.lower)
            return rule_lt(candidate_low, pivot.upper)

        if direction == UP:
            return rule_ge(candidate_low, pivot.upper)
        return rule_le(candidate_high, pivot.lower)

    def _select_bounded_extreme(
        self,
        direction: str,
        targets: list[CandidateEvaluation],
        opposite_pool: list[CandidateEvaluation],
    ) -> CandidateEvaluation | None:
        valid_targets = sorted(targets, key=lambda ev: (ev.middle_index, ev.turn_k_index or 10**9))
        if not valid_targets:
            return None
        valid_opposites = sorted(
            [
                ev
                for ev in opposite_pool
                if ev.checks.get("ma34_side_and_strict_ma5")
                and ev.checks.get("not_black_k")
                and ev.checks.get("not_turn_k_itself")
                and ev.checks.get("has_turn_k")
            ],
            key=lambda ev: (ev.middle_index, ev.turn_k_index or 10**9),
        )

        current = valid_targets[0]
        while True:
            boundary = next((ev for ev in valid_opposites if ev.middle_index > current.middle_index), None)
            boundary_index = boundary.middle_index if boundary else 10**12
            same_side_before_boundary = [
                ev
                for ev in valid_targets
                if current.middle_index <= ev.middle_index < boundary_index
            ]
            if direction == UP:
                extended = max(same_side_before_boundary, key=lambda ev: (ev.price, ev.middle_index))
                should_continue = rule_gt(extended.price, current.price) and extended.middle_index > current.middle_index
            else:
                extended = min(same_side_before_boundary, key=lambda ev: (ev.price, -ev.middle_index))
                should_continue = rule_lt(extended.price, current.price) and extended.middle_index > current.middle_index
            if not should_continue:
                return current
            current = extended

    def _find_initial_cross(self, df: pd.DataFrame, start: int, end: int) -> KRef | None:
        start = max(start, first_valid_index(df, "ma34") or 1, 1)
        end = min(end, len(df) - 1)
        for i in range(start, end + 1):
            prev = rule_value(df.iloc[i - 1].ma5) - rule_value(df.iloc[i - 1].ma34)
            cur = rule_value(df.iloc[i].ma5) - rule_value(df.iloc[i].ma34)
            if pd.isna(prev) or pd.isna(cur):
                continue
            if prev == 0 or cur == 0 or prev * cur < 0:
                return KRef(index=i, date=format_date(df.iloc[i].date), price=float(df.iloc[i].close), role="initial_cross")
        return None

    def _apply_confirmed_pair(
        self,
        df: pd.DataFrame,
        report: SegmentReport,
        direction: str,
        start_eval: CandidateEvaluation | KRef,
        end_eval: CandidateEvaluation,
    ) -> None:
        start_ref = eval_to_ref(df, start_eval, BOTTOM if direction == UP else TOP)
        end_ref = eval_to_ref(df, end_eval, TOP if direction == UP else BOTTOM)
        if direction == UP:
            report.confirmed_bottom = start_ref
            report.confirmed_top = end_ref
        else:
            report.confirmed_top = start_ref
            report.confirmed_bottom = end_ref
        report.turn_k = KRef(
            index=end_eval.turn_k_index,
            date=end_eval.turn_k_date or format_date(df.iloc[end_eval.turn_k_index].date),
            price=float(df.iloc[end_eval.turn_k_index].close),
            role="turn_k",
        )
        report.end_index = end_ref.index
        report.end_date = end_ref.date
        end_eval.status = "confirmed"

    def _apply_dynamic_refresh(self, df: pd.DataFrame, report: SegmentReport, direction: str) -> None:
        if not report.turn_k:
            return
        if direction == UP and report.confirmed_top:
            current = report.confirmed_top
            previous_entity_floor = body_low(df.iloc[current.index])
            for i in range(report.turn_k.index + 1, len(df)):
                row = df.iloc[i]
                if rule_gt(row.high, current.price) and rule_gt(body_high(row), previous_entity_floor):
                    current = KRef(index=i, date=format_date(row.date), price=float(row.high), role="confirmed_top")
                    report.confirmed_top = current
                    report.end_index = i
                    report.end_date = current.date
                    report.refresh_status = "price_new_high_refresh"
                    if i + 1 < len(df):
                        nxt = df.iloc[i + 1]
                        report.turn_k = KRef(index=i + 1, date=format_date(nxt.date), price=float(nxt.close), role="turn_k_newborn")
                    else:
                        report.turn_k = None
                        report.decision_status = "needs_more_k"
                    previous_entity_floor = body_low(row)
        elif direction == DOWN and report.confirmed_bottom:
            current = report.confirmed_bottom
            previous_entity_floor = body_low(df.iloc[current.index])
            for i in range(report.turn_k.index + 1, len(df)):
                row = df.iloc[i]
                if rule_lt(row.low, current.price) and rule_gt(body_high(row), previous_entity_floor):
                    current = KRef(index=i, date=format_date(row.date), price=float(row.low), role="confirmed_bottom")
                    report.confirmed_bottom = current
                    report.end_index = i
                    report.end_date = current.date
                    report.refresh_status = "price_new_low_refresh"
                    if i + 1 < len(df):
                        nxt = df.iloc[i + 1]
                        report.turn_k = KRef(index=i + 1, date=format_date(nxt.date), price=float(nxt.close), role="turn_k_newborn")
                    else:
                        report.turn_k = None
                        report.decision_status = "needs_more_k"
                    previous_entity_floor = body_low(row)

    def _fill_pivot_fields(self, report: SegmentReport, pivot: PivotInfo, df: pd.DataFrame) -> None:
        report.pivot_confirm_k = pivot.confirm_index
        report.pivot_confirm_date = pivot.confirm_date
        report.pivot_line = pivot.pivot_line
        report.pivot_type = pivot.pivot_type
        report.pivot_lower = pivot.lower
        report.pivot_upper = pivot.upper
        report.start_pattern = pivot.start_patterns
        report.has_black_k = bool(pivot.black_k_indices)
        report.extension_or_recomposition = pivot.extension_or_recomposition
        report.diagnostics["has_multiple_pivots"] = pivot.has_multiple_pivots
        report.diagnostics["pivot_reasoning"] = pivot.reasoning
        report.diagnostics["black_k_indices"] = pivot.black_k_indices
        report.diagnostics["black_k_refs"] = krefs_for_indices(df, pivot.black_k_indices, "black_k")
        report.diagnostics["pivot_formation_end_index"] = pivot.formation_end_index
        report.diagnostics["pivot_formation_end_date"] = (
            format_date(df.iloc[pivot.formation_end_index].date)
            if 0 <= pivot.formation_end_index < len(df)
            else None
        )

    def _mark_unconfirmed(self, report: SegmentReport, selected_end: CandidateEvaluation | None) -> None:
        selected_index = selected_end.middle_index if selected_end else None
        all_candidates = report.candidate_top + report.candidate_bottom
        for ev in all_candidates:
            if ev.middle_index == selected_index and report.decision_status == "confirmed":
                continue
            if not ev.rejected_reasons and ev.status != "confirmed":
                ev.rejected_reasons.append("形态成立，但未进入唯一确认组合。")
            report.unconfirmed_candidates.append(ev)

    def _trim_candidates_for_report(
        self, report: SegmentReport, selected_end: CandidateEvaluation | None
    ) -> None:
        cutoff = None
        if report.turn_k:
            cutoff = report.turn_k.index
        elif selected_end and selected_end.turn_k_index is not None:
            cutoff = selected_end.turn_k_index
        elif report.pivot_confirm_k is not None:
            cutoff = report.pivot_confirm_k + self.config.min_remaining_bars
        if cutoff is None:
            return
        cutoff = max(cutoff, report.start_index)
        report.candidate_top = [
            ev for ev in report.candidate_top if ev.middle_index >= report.start_index and ev.indexes[0] <= cutoff
        ]
        report.candidate_bottom = [
            ev for ev in report.candidate_bottom if ev.middle_index >= report.start_index and ev.indexes[0] <= cutoff
        ]

    def _initial_direction(self, df: pd.DataFrame, start_index: int, value: str) -> str:
        if value in {UP, DOWN, "both"}:
            return value
        recent = df.iloc[max(start_index, len(df) - 20) :]
        if recent.empty:
            return UP
        return UP if rule_ge(recent.iloc[-1].close, recent.iloc[-1].ma34) else DOWN


RULE_DECIMALS = 2


def rule_value(value: object) -> float:
    return round(float(value), RULE_DECIMALS)


def rule_gt(left: object, right: object) -> bool:
    return rule_value(left) > rule_value(right)


def rule_ge(left: object, right: object) -> bool:
    return rule_value(left) >= rule_value(right)


def rule_lt(left: object, right: object) -> bool:
    return rule_value(left) < rule_value(right)


def rule_le(left: object, right: object) -> bool:
    return rule_value(left) <= rule_value(right)


def body_low(row: pd.Series) -> float:
    return rule_value(min(row.open, row.close))


def body_high(row: pd.Series) -> float:
    return rule_value(max(row.open, row.close))


def entity_strict_cross_ma(row: pd.Series, ma_col: str) -> bool:
    ma = row[ma_col]
    return pd.notna(ma) and rule_lt(body_low(row), ma) and rule_lt(ma, body_high(row))


def kline_strict_cross_ma(row: pd.Series, ma_col: str) -> bool:
    ma = row[ma_col]
    return pd.notna(ma) and rule_lt(row.low, ma) and rule_lt(ma, row.high)


def is_reverse_break_ma5(df: pd.DataFrame, index: int, direction: str) -> bool:
    row = df.iloc[index]
    if pd.isna(row.ma5):
        return False
    if direction == UP and rule_lt(body_low(row), row.ma5):
        return True
    if direction == DOWN and rule_gt(body_high(row), row.ma5):
        return True
    if index <= 0:
        return False
    prev = df.iloc[index - 1]
    if pd.isna(prev.ma5):
        return False
    if direction == UP:
        return rule_gt(body_low(prev), prev.ma5) and rule_lt(body_high(row), row.ma5)
    return rule_lt(body_high(prev), prev.ma5) and rule_gt(body_low(row), row.ma5)


def is_positive_gap_ma5(df: pd.DataFrame, index: int, direction: str) -> bool:
    if index <= 0:
        return False
    row, prev = df.iloc[index], df.iloc[index - 1]
    if pd.isna(row.ma5) or pd.isna(prev.ma5):
        return False
    if direction == UP:
        return rule_gt(row.low, row.ma5) and rule_lt(prev.high, prev.ma5)
    return rule_lt(row.high, row.ma5) and rule_gt(prev.low, prev.ma5)


def is_pivot_gap(row: pd.Series) -> bool:
    return not entity_strict_cross_ma(row, "ma5")


def is_double_gap_pivot(df: pd.DataFrame, index: int, direction: str) -> bool:
    if index <= 0:
        return False
    row, prev = df.iloc[index], df.iloc[index - 1]
    if not is_pivot_gap(row):
        return False
    if direction == UP:
        return rule_gt(prev.low, row.high)
    return rule_lt(prev.high, row.low)


def calc_pivot_line(df: pd.DataFrame, index: int, direction: str) -> float:
    row = df.iloc[index]
    if direction == UP:
        if rule_lt(body_low(row), row.ma5) and rule_lt(row.ma5, body_high(row)):
            return body_low(row)
        if is_double_gap_pivot(df, index, direction):
            return nearest_entity_price(row, row.ma5)
        return body_low(row)
    if rule_lt(body_low(row), row.ma5) and rule_lt(row.ma5, body_high(row)):
        return body_high(row)
    if is_double_gap_pivot(df, index, direction):
        return nearest_entity_price(row, row.ma5)
    return body_low(row)


def nearest_entity_price(row: pd.Series, value: float) -> float:
    low, high = body_low(row), body_high(row)
    target = rule_value(value)
    return low if abs(low - target) <= abs(high - target) else high


def strict_overlap(low1: float, high1: float, low2: float, high2: float) -> bool:
    return rule_gt(high1, low2) and rule_lt(low1, high2)


def normalize_zone(a: float, b: float) -> tuple[float, float]:
    left, right = rule_value(a), rule_value(b)
    return (float(min(left, right)), float(max(left, right)))


def overlap_zone_with_fallback(first: pd.Series, breaker: pd.Series) -> tuple[float, float]:
    low = max(rule_value(first.low), rule_value(breaker.low))
    high = min(rule_value(first.high), rule_value(breaker.high))
    if rule_le(low, high):
        return float(low), float(high)
    return normalize_zone(float(first.high), float(breaker.low))


def kline_overlaps_zone(row: pd.Series, low: float, high: float) -> bool:
    return bool(rule_ge(row.high, low) and rule_le(row.low, high))


def kline_gaps_over_zone(prev: pd.Series, row: pd.Series, low: float, high: float) -> bool:
    return bool(
        (rule_lt(prev.high, low) and rule_gt(row.low, high))
        or (rule_gt(prev.low, high) and rule_lt(row.high, low))
    )


def first_valid_index(df: pd.DataFrame, column: str) -> int | None:
    valid = df.index[df[column].notna()]
    if len(valid) == 0:
        return None
    return int(valid[0])


def candidate_reasons(ev: CandidateEvaluation) -> list[str]:
    labels = {
        "pair_independent_2k": "顶底组合两侧均未满足独立2K",
        "ma34_side_and_strict_ma5": "MA34两侧或实体边界MA5条件不成立",
        "not_black_k": "候选顶/底K本身是黑K",
        "not_turn_k_itself": "该K已被识别为转折K，不能作为顶/底候选",
        "structure_perfect": "结构完美不成立：顶底未严格位于线段中枢两侧或结构顺序错误",
        "has_turn_k": "缺少有效转折K",
        "dynamic_ma5_survives": "转折K后 MA5 继续正向生长，原候选失效",
    }
    if ev.checks.get("not_turn_k_itself") is False:
        return [labels["not_turn_k_itself"]]
    return [reason for key, reason in labels.items() if ev.checks.get(key) is False]


def independent_side(tops: Iterable[CandidateEvaluation], bottoms: Iterable[CandidateEvaluation]) -> str:
    top_ok = any(ev.checks.get("independent_2k") for ev in tops)
    bottom_ok = any(ev.checks.get("independent_2k") for ev in bottoms)
    if top_ok and bottom_ok:
        return "两侧满足"
    if top_ok:
        return "顶侧满足"
    if bottom_ok:
        return "底侧满足"
    return "未满足"


def eval_to_ref(df: pd.DataFrame, value: CandidateEvaluation | KRef, role: str) -> KRef:
    if isinstance(value, KRef):
        return value
    row = df.iloc[value.middle_index]
    price = float(row.high if role == TOP else row.low)
    return KRef(index=value.middle_index, date=format_date(row.date), price=price, role=f"confirmed_{role}")


def endpoint_type(value: CandidateEvaluation | KRef) -> str | None:
    if isinstance(value, CandidateEvaluation):
        return value.candidate_type
    if "top" in value.role:
        return TOP
    if "bottom" in value.role:
        return BOTTOM
    return None


def is_anchor_ref(value: KRef) -> bool:
    return any(
        marker in value.role
        for marker in ("anchor", "historical_low", "anchored_bridge")
    )


def endpoint_outside_pivot(value: CandidateEvaluation | KRef, pivot: PivotInfo) -> bool:
    kind = endpoint_type(value)
    if kind == TOP:
        return rule_gt(value.price, pivot.upper)
    if kind == BOTTOM:
        return rule_lt(value.price, pivot.lower)
    return False


def endpoint_low_high(df: pd.DataFrame, value: CandidateEvaluation | KRef) -> tuple[float, float]:
    index = value.index if isinstance(value, KRef) else value.middle_index
    row = df.iloc[index]
    return float(row.low), float(row.high)


def opposite(direction: str) -> str:
    return DOWN if direction == UP else UP


def choose_better_report(reports: list[SegmentReport]) -> SegmentReport:
    confirmed = [r for r in reports if r.decision_status == "confirmed" and r.end_index is not None]
    if confirmed:
        return min(confirmed, key=lambda r: r.end_index or 10**9)
    candidates = [r for r in reports if r.pivot_confirm_k is not None]
    if candidates:
        return min(candidates, key=lambda r: r.pivot_confirm_k or 10**9)
    return reports[0]


def describe_ref(ref: KRef | None) -> str:
    if not ref:
        return "None"
    return f"{ref.date}#{ref.index}@{ref.price:.3f}"


def map_optional_index(index: int | None, reversed_data: pd.DataFrame) -> int | None:
    if index is None:
        return None
    return int(reversed_data.iloc[int(index)].original_index)


def date_for_original_index(reversed_data: pd.DataFrame, original_index: int | None) -> str | None:
    if original_index is None:
        return None
    row = reversed_data[reversed_data["original_index"] == int(original_index)]
    if row.empty:
        return None
    return format_date(row.iloc[0].date)


def map_kref_index(ref: KRef | None, reversed_data: pd.DataFrame) -> KRef | None:
    if ref is None:
        return None
    reversed_index = int(ref.index)
    ref.index = map_optional_index(reversed_index, reversed_data)
    ref.date = format_date(reversed_data.iloc[reversed_index].date)
    return ref


def map_candidate_index(ev: CandidateEvaluation, reversed_data: pd.DataFrame) -> CandidateEvaluation:
    reversed_middle = int(ev.middle_index)
    ev.middle_index = map_optional_index(reversed_middle, reversed_data)
    ev.middle_date = format_date(reversed_data.iloc[reversed_middle].date)
    ev.indexes = sorted(map_optional_index(i, reversed_data) for i in ev.indexes)
    if ev.turn_k_index is not None:
        reversed_turn = int(ev.turn_k_index)
        ev.turn_k_index = map_optional_index(reversed_turn, reversed_data)
        ev.turn_k_date = format_date(reversed_data.iloc[reversed_turn].date)
    return ev


def map_index_date_diagnostics(report: SegmentReport, reversed_data: pd.DataFrame) -> None:
    black_indices = report.diagnostics.get("black_k_indices")
    if black_indices:
        pairs = [(i, map_optional_index(i, reversed_data)) for i in black_indices]
        mapped = [mapped_i for _, mapped_i in pairs]
        report.diagnostics["black_k_indices"] = mapped
        report.diagnostics["black_k_refs"] = [
            KRef(
                index=mapped_i,
                date=format_date(reversed_data.iloc[int(reversed_i)].date),
                price=float(reversed_data.iloc[int(reversed_i)].close),
                role="black_k",
            )
            for reversed_i, mapped_i in pairs
        ]
    formation_end = report.diagnostics.get("pivot_formation_end_index")
    if formation_end is not None:
        mapped_end = map_optional_index(formation_end, reversed_data)
        report.diagnostics["pivot_formation_end_index"] = mapped_end
        report.diagnostics["pivot_formation_end_date"] = date_for_original_index(reversed_data, mapped_end)


def map_endpoint_extension_diagnostics(report: SegmentReport, reversed_data: pd.DataFrame) -> None:
    extensions = report.diagnostics.get("endpoint_extensions") or []
    for extension in extensions:
        reversed_index = extension.get("extended_to_index")
        if reversed_index is not None:
            extension["extended_to_index"] = map_optional_index(reversed_index, reversed_data)
            extension["extended_to_date"] = format_date(reversed_data.iloc[int(reversed_index)].date)
        boundary_index = extension.get("boundary_index")
        if boundary_index is not None:
            extension["boundary_index"] = map_optional_index(boundary_index, reversed_data)


def krefs_for_indices(df: pd.DataFrame, indices: list[int], role: str) -> list[KRef]:
    refs = []
    for index in indices:
        if 0 <= index < len(df):
            row = df.iloc[index]
            refs.append(KRef(index=index, date=format_date(row.date), price=float(row.close), role=role))
    return refs


def refresh_report_summary(report: SegmentReport) -> None:
    if report.decision_status == "confirmed":
        top_txt = describe_ref(report.confirmed_top)
        bottom_txt = describe_ref(report.confirmed_bottom)
        turn_txt = describe_ref(report.turn_k)
        if report.pivot_lower is not None and report.pivot_upper is not None:
            zone = f"[{report.pivot_lower:.3f}, {report.pivot_upper:.3f}]"
        else:
            zone = "None"
        report.reasoning_summary = (
            f"已确认线段：方向={report.segment_direction}，底={bottom_txt}，顶={top_txt}，"
            f"转折K={turn_txt}，中枢区间={zone}。"
        )
