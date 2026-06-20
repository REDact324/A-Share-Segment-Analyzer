from __future__ import annotations

import json
import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .agent import SegmentAnalysisAgent, segment_source_label
from .analyzer import AnalyzerConfig
from .data_fetcher import compact_date, normalize_symbol
from .models import AnalysisRun, KRef, jsonable


ADJUST_EMPTY_LABEL = "<空>"
SYMBOL_HISTORY_LIMIT = 20


class SegmentAnalyzerGUI:
    def __init__(
        self,
        root: tk.Tk,
        *,
        cache_dir: str | Path | None = "data/cache",
        use_cache: bool = True,
        max_segments: int = 200,
        initial_scan_mode: str = "lowest",
    ):
        self.root = root
        self.root.title("日线线段分析")
        self.root.geometry("980x720")
        self.root.minsize(720, 520)

        config = AnalyzerConfig(max_segments=max_segments)
        self.agent = SegmentAnalysisAgent(config=config)
        self.agent.data_client.cache_dir = Path(cache_dir) if cache_dir else None
        self.agent.data_client.use_cache = use_cache
        if self.agent.data_client.cache_dir:
            self.agent.data_client.cache_dir.mkdir(parents=True, exist_ok=True)
        self.symbol_history_path = (
            self.agent.data_client.cache_dir / "gui_symbol_history.json"
            if self.agent.data_client.cache_dir
            else None
        )

        self.current_run: AnalysisRun | None = None
        self.current_text = ""
        self.current_json: dict[str, Any] | None = None
        self.symbol_history = self._load_symbol_history()

        self.symbol_var = tk.StringVar()
        self.start_var = tk.StringVar(value="20000101")
        self.end_var = tk.StringVar(value=datetime.now().strftime("%Y%m%d"))
        self.adjust_var = tk.StringVar(value="qfq")
        self.scan_mode_var = tk.StringVar(value=initial_scan_mode)
        self.direction_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value="输入股票代码后开始分析。")
        self.input_widgets: list[tk.Widget] = []
        self.readonly_widgets: list[ttk.Combobox] = []
        self.symbol_suggestions_popup: tk.Toplevel | None = None
        self.symbol_suggestions_listbox: tk.Listbox | None = None
        self.symbol_suggestions: list[str] = []
        self.spinner_angle = 0
        self.spinner_job: str | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.LabelFrame(self.root, text="分析参数", padding=(12, 10, 12, 10))
        top.grid(row=0, column=0, sticky="ew")
        for col in (1, 3, 5):
            top.columnconfigure(col, weight=1, uniform="control")

        ttk.Label(top, text="股票代码").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 8))
        self.symbol_entry = ttk.Entry(top, textvariable=self.symbol_var, width=16)
        self.symbol_entry.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=(0, 8))
        self.symbol_entry.bind("<Return>", lambda _event: self.start_analysis())
        self.symbol_entry.bind("<KeyRelease>", self._on_symbol_key_release)
        self.symbol_entry.bind("<Down>", self._focus_symbol_suggestions)
        self.symbol_entry.bind("<Escape>", lambda _event: self._hide_symbol_suggestions())
        self.symbol_entry.bind("<FocusOut>", lambda _event: self.root.after(120, self._hide_symbol_suggestions_if_unfocused))
        self.symbol_entry.bind("<Configure>", lambda _event: self._position_symbol_suggestions())
        self.input_widgets.append(self.symbol_entry)

        ttk.Label(top, text="开始").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=(0, 8))
        self.start_entry = ttk.Entry(top, textvariable=self.start_var, width=12)
        self.start_entry.grid(row=0, column=3, sticky="ew", padx=(0, 12), pady=(0, 8))
        self.input_widgets.append(self.start_entry)

        ttk.Label(top, text="结束").grid(row=0, column=4, sticky="w", padx=(0, 6), pady=(0, 8))
        self.end_entry = ttk.Entry(top, textvariable=self.end_var, width=12)
        self.end_entry.grid(row=0, column=5, sticky="ew", pady=(0, 8))
        self.input_widgets.append(self.end_entry)

        ttk.Label(top, text="复权").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(0, 8))
        self.adjust_combo = ttk.Combobox(
            top,
            textvariable=self.adjust_var,
            values=("qfq", "hfq", ADJUST_EMPTY_LABEL),
            width=7,
            state="readonly",
        )
        self.adjust_combo.grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=(0, 8))
        self.readonly_widgets.append(self.adjust_combo)
        enable_combobox_option_hover(self.adjust_combo)

        ttk.Label(top, text="扫描").grid(row=1, column=2, sticky="w", padx=(0, 6), pady=(0, 8))
        self.scan_mode_combo = ttk.Combobox(
            top,
            textvariable=self.scan_mode_var,
            values=("lowest", "history"),
            width=9,
            state="readonly",
        )
        self.scan_mode_combo.grid(row=1, column=3, sticky="ew", padx=(0, 12), pady=(0, 8))
        self.readonly_widgets.append(self.scan_mode_combo)
        enable_combobox_option_hover(self.scan_mode_combo)

        ttk.Label(top, text="方向").grid(row=1, column=4, sticky="w", padx=(0, 6), pady=(0, 8))
        self.direction_combo = ttk.Combobox(
            top,
            textvariable=self.direction_var,
            values=("auto", "up", "down", "both"),
            width=8,
            state="readonly",
        )
        self.direction_combo.grid(row=1, column=5, sticky="ew", pady=(0, 8))
        self.readonly_widgets.append(self.direction_combo)
        enable_combobox_option_hover(self.direction_combo)

        actions = ttk.Frame(top)
        actions.grid(row=2, column=0, columnspan=6, sticky="ew")
        actions.columnconfigure(0, weight=1)

        self.analyze_button = ttk.Button(actions, text="分析", command=self.start_analysis)
        self.analyze_button.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self.txt_button = ttk.Button(actions, text="下载 TXT", command=self.save_txt, state="disabled")
        self.txt_button.grid(row=0, column=2, sticky="ew", padx=(0, 8))

        self.json_button = ttk.Button(actions, text="下载 JSON", command=self.save_json, state="disabled")
        self.json_button.grid(row=0, column=3, sticky="ew")

        output_frame = ttk.Frame(self.root, padding=(12, 4, 12, 6))
        output_frame.grid(row=1, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output = tk.Text(output_frame, wrap="word", undo=False)
        self.output.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(output_frame, command=self.output.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scroll.set)

        status = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(1, weight=1)
        self.spinner = tk.Canvas(status, width=24, height=24, highlightthickness=0)
        self.spinner.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.spinner.grid_remove()
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=1, sticky="ew")

        self.symbol_entry.focus_set()

    def start_analysis(self) -> None:
        try:
            symbol = normalize_symbol(self.symbol_var.get())
            start_date = self._clean_date(self.start_var.get(), "开始日期")
            end_date = self._clean_date(self.end_var.get(), "结束日期")
        except ValueError as exc:
            messagebox.showerror("输入错误", str(exc))
            return

        self.current_run = None
        self.current_text = ""
        self.current_json = None
        self._remember_symbol(symbol)
        self._set_export_state("disabled")
        self._set_busy(True)
        self._set_step(f"准备分析 {symbol}。")
        self._replace_output(f"正在分析 {symbol}，请稍候...\n")

        kwargs = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "direction": self.direction_var.get(),
            "adjust": normalize_adjust_option(self.adjust_var.get()),
            "scan_mode": self.scan_mode_var.get(),
        }
        worker = threading.Thread(target=self._analyze_worker, kwargs=kwargs, daemon=True)
        worker.start()

    def _analyze_worker(self, **kwargs: Any) -> None:
        try:
            symbol = kwargs["symbol"]
            self._set_step(f"1/4 正在抓取 {symbol} 的日线数据...")
            df = self.agent.data_client.fetch_daily(
                symbol,
                start_date=kwargs["start_date"],
                end_date=kwargs["end_date"],
                adjust=kwargs["adjust"],
            )
            self._set_step(f"2/4 已取得 {len(df)} 根日线，正在识别线段...")
            run = self.agent.analyze_dataframe(
                df,
                direction=kwargs["direction"],
                symbol=symbol,
                scan_mode=kwargs["scan_mode"],
            )
            self._set_step("3/4 正在整理线段摘要...")
            text = build_segments_text(run)
            self._set_step("4/4 正在生成 JSON 导出内容...")
            payload = build_json_payload(run)
        except Exception as exc:
            self.root.after(0, lambda error=exc: self._analysis_failed(error))
            return
        self.root.after(0, lambda: self._analysis_finished(run, text, payload))

    def _analysis_finished(self, run: AnalysisRun, text: str, payload: dict[str, Any]) -> None:
        self.current_run = run
        self.current_text = text
        self.current_json = payload
        self._replace_output(text)
        self._set_busy(False)
        self._set_export_state("normal")
        self.status_var.set(f"完成：{run.symbol}，线段数 {run.segment_count}。")

    def _analysis_failed(self, exc: Exception) -> None:
        self._set_busy(False)
        self._set_export_state("disabled")
        self._replace_output(f"分析失败：{exc}\n")
        self.status_var.set("分析失败。")
        messagebox.showerror("分析失败", str(exc))

    def save_txt(self) -> None:
        if not self.current_text:
            return
        path = filedialog.asksaveasfilename(
            title="保存线段 TXT",
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
            initialfile=self._default_export_name("segments", "txt"),
        )
        if not path:
            return
        Path(path).write_text(self.current_text, encoding="utf-8")
        self.status_var.set(f"TXT 已保存：{path}")

    def save_json(self) -> None:
        if self.current_json is None:
            return
        path = filedialog.asksaveasfilename(
            title="保存详细 JSON",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            initialfile=self._default_export_name("details", "json"),
        )
        if not path:
            return
        Path(path).write_text(
            json.dumps(self.current_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.status_var.set(f"JSON 已保存：{path}")

    def _default_export_name(self, suffix: str, ext: str) -> str:
        symbol = self.current_run.symbol if self.current_run else normalize_symbol(self.symbol_var.get())
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{symbol}_{suffix}_{date}.{ext}"

    def _clean_date(self, value: str, label: str) -> str:
        compact = compact_date(value)
        if len(compact) != 8:
            raise ValueError(f"{label} 必须是 YYYYMMDD 格式。")
        return compact

    def _load_symbol_history(self) -> list[str]:
        if self.symbol_history_path is None or not self.symbol_history_path.exists():
            return []
        try:
            payload = json.loads(self.symbol_history_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        symbols = []
        for item in payload:
            try:
                symbol = normalize_symbol(item)
            except ValueError:
                continue
            if symbol not in symbols:
                symbols.append(symbol)
        return symbols[:SYMBOL_HISTORY_LIMIT]

    def _remember_symbol(self, symbol: str) -> None:
        symbols = [symbol] + [item for item in self.symbol_history if item != symbol]
        self.symbol_history = symbols[:SYMBOL_HISTORY_LIMIT]
        if self.symbol_history_path is None:
            return
        try:
            self.symbol_history_path.parent.mkdir(parents=True, exist_ok=True)
            self.symbol_history_path.write_text(
                json.dumps(self.symbol_history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            self.status_var.set("历史记录保存失败，但本次分析会继续。")

    def _on_symbol_key_release(self, event: tk.Event) -> None:
        if not self.symbol_history:
            return
        if event.keysym in {"Return", "Escape", "Tab", "Up", "Down", "Left", "Right"}:
            return
        matches = self._matching_symbol_history()
        if matches:
            self._show_symbol_suggestions(matches)
        else:
            self._hide_symbol_suggestions()

    def _matching_symbol_history(self) -> list[str]:
        query = "".join(ch for ch in self.symbol_var.get().strip() if ch.isdigit())
        if not query:
            return []
        return [symbol for symbol in self.symbol_history if symbol.startswith(query)]

    def _show_symbol_suggestions(self, matches: list[str]) -> None:
        self.symbol_suggestions = matches
        if self.symbol_suggestions_popup is None:
            self.symbol_suggestions_popup = tk.Toplevel(self.root)
            self.symbol_suggestions_popup.wm_overrideredirect(True)
            self.symbol_suggestions_popup.transient(self.root)
            self.symbol_suggestions_listbox = tk.Listbox(
                self.symbol_suggestions_popup,
                activestyle="none",
                exportselection=False,
                selectbackground="#2563eb",
                selectforeground="white",
            )
            self.symbol_suggestions_listbox.pack(fill="both", expand=True)
            self.symbol_suggestions_listbox.bind("<Motion>", self._highlight_symbol_suggestion)
            self.symbol_suggestions_listbox.bind("<ButtonRelease-1>", self._accept_symbol_suggestion)
            self.symbol_suggestions_listbox.bind("<Return>", self._accept_symbol_suggestion)
            self.symbol_suggestions_listbox.bind("<Escape>", lambda _event: self._hide_symbol_suggestions())
            self.symbol_suggestions_listbox.bind("<FocusOut>", lambda _event: self.root.after(120, self._hide_symbol_suggestions_if_unfocused))
        if self.symbol_suggestions_listbox is None:
            return
        self.symbol_suggestions_listbox.delete(0, tk.END)
        for symbol in matches:
            self.symbol_suggestions_listbox.insert(tk.END, symbol)
        self.symbol_suggestions_listbox.configure(height=min(6, len(matches)))
        self.symbol_suggestions_listbox.selection_clear(0, tk.END)
        self.symbol_suggestions_listbox.selection_set(0)
        self.symbol_suggestions_listbox.activate(0)
        self._position_symbol_suggestions()
        self.symbol_suggestions_popup.deiconify()

    def _position_symbol_suggestions(self) -> None:
        if self.symbol_suggestions_popup is None or self.symbol_suggestions_listbox is None:
            return
        if not self.symbol_suggestions:
            return
        x = self.symbol_entry.winfo_rootx()
        y = self.symbol_entry.winfo_rooty() + self.symbol_entry.winfo_height()
        width = max(self.symbol_entry.winfo_width(), 120)
        self.symbol_suggestions_popup.update_idletasks()
        height = self.symbol_suggestions_listbox.winfo_reqheight()
        self.symbol_suggestions_popup.geometry(f"{width}x{height}+{x}+{y}")

    def _highlight_symbol_suggestion(self, event: tk.Event) -> None:
        listbox = event.widget
        index = listbox.nearest(event.y)
        if index < 0:
            return
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(index)
        listbox.activate(index)

    def _accept_symbol_suggestion(self, event: tk.Event) -> str:
        listbox = event.widget
        selected = listbox.curselection()
        index = selected[0] if selected else listbox.nearest(getattr(event, "y", 0))
        if index >= 0:
            self.symbol_var.set(listbox.get(index))
            self.symbol_entry.icursor(tk.END)
        self._hide_symbol_suggestions()
        self.symbol_entry.focus_set()
        return "break"

    def _focus_symbol_suggestions(self, _event: tk.Event) -> str | None:
        if not self.symbol_history:
            return None
        matches = self._matching_symbol_history()
        if matches:
            self._show_symbol_suggestions(matches)
            if self.symbol_suggestions_listbox is not None:
                self.symbol_suggestions_listbox.focus_set()
            return "break"
        return None

    def _hide_symbol_suggestions_if_unfocused(self) -> None:
        focus = self.root.focus_get()
        if focus not in {self.symbol_entry, self.symbol_suggestions_listbox}:
            self._hide_symbol_suggestions()

    def _hide_symbol_suggestions(self) -> None:
        self.symbol_suggestions = []
        if self.symbol_suggestions_popup is not None:
            self.symbol_suggestions_popup.withdraw()

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.analyze_button.configure(state=state)
        for widget in self.input_widgets:
            widget.configure(state=state)
        for widget in self.readonly_widgets:
            widget.configure(state="disabled" if busy else "readonly")
        if busy:
            self._hide_symbol_suggestions()
            self._start_spinner()
        else:
            self._stop_spinner()

    def _set_export_state(self, state: str) -> None:
        self.txt_button.configure(state=state)
        self.json_button.configure(state=state)

    def _set_step(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))

    def _start_spinner(self) -> None:
        if self.spinner_job is not None:
            return
        self.spinner.grid()
        self.spinner_angle = 0
        self._draw_spinner()

    def _stop_spinner(self) -> None:
        if self.spinner_job is not None:
            self.root.after_cancel(self.spinner_job)
            self.spinner_job = None
        self.spinner.delete("all")
        self.spinner.grid_remove()

    def _draw_spinner(self) -> None:
        self.spinner.delete("all")
        self.spinner.create_oval(4, 4, 20, 20, outline="#d1d5db", width=3)
        self.spinner.create_arc(
            4,
            4,
            20,
            20,
            start=self.spinner_angle,
            extent=115,
            style="arc",
            outline="#2563eb",
            width=3,
        )
        self.spinner_angle = (self.spinner_angle + 18) % 360
        self.spinner_job = self.root.after(55, self._draw_spinner)

    def _replace_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")


def enable_combobox_option_hover(combo: ttk.Combobox) -> None:
    combo.configure(postcommand=lambda widget=combo: widget.after_idle(lambda: _bind_popdown_hover(widget)))


def _bind_popdown_hover(combo: ttk.Combobox) -> None:
    try:
        popdown = combo.tk.call("ttk::combobox::PopdownWindow", combo)
        listbox = f"{popdown}.f.l"
        combo.tk.call(
            listbox,
            "configure",
            "-activestyle",
            "none",
            "-selectbackground",
            "#2563eb",
            "-selectforeground",
            "white",
        )
        script = (
            f"set idx [{listbox} nearest %y];"
            f"if {{$idx >= 0}} {{"
            f"{listbox} selection clear 0 end;"
            f"{listbox} selection set $idx;"
            f"{listbox} activate $idx;"
            f"}}"
        )
        combo.tk.call("bind", listbox, "<Motion>", script)
    except tk.TclError:
        return


def run_gui(
    *,
    cache_dir: str | Path | None = "data/cache",
    use_cache: bool = True,
    max_segments: int = 200,
    initial_scan_mode: str = "lowest",
) -> None:
    root = tk.Tk()
    SegmentAnalyzerGUI(
        root,
        cache_dir=cache_dir,
        use_cache=use_cache,
        max_segments=max_segments,
        initial_scan_mode=initial_scan_mode,
    )
    root.mainloop()


def default_packaged_cache_dir(app_name: str = "SegmentAnalyzer") -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name / "cache"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / app_name / "cache"
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / app_name / "cache"


def build_segments_text(run: AnalysisRun) -> str:
    title = run.symbol or "未知股票"
    if run.name:
        title += f" {run.name}"
    lines = [
        f"{title}: {run.data_start} 至 {run.data_end}",
        f"线段数: {run.segment_count}",
        "",
    ]
    if not run.reports:
        lines.append("没有识别到线段。")
        return "\n".join(lines)

    for report in run.reports:
        source = segment_source_label(report)
        lines.extend(
            [
                f"#{report.segment_no} {direction_label(report.segment_direction)} [{source}]",
                f"状态: {report.decision_status}",
                f"起点: {format_index_date(report.start_index, report.start_date)}",
                f"终点: {format_index_date(report.end_index, report.end_date)}",
                f"确认底: {format_ref(report.confirmed_bottom)}",
                f"确认顶: {format_ref(report.confirmed_top)}",
                f"中枢: {format_pivot(report)}",
                f"转折K: {format_ref(report.turn_k)}",
                f"初始交叉: {format_ref(report.initial_cross)}",
                f"独立2K: {report.independent_2k_side}",
                f"刷新状态: {report.refresh_status}",
            ]
        )
        if report.start_pattern:
            lines.append(f"起手三式: {', '.join(report.start_pattern)}")
        if report.reasoning_summary:
            lines.append(f"说明: {report.reasoning_summary}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_json_payload(run: AnalysisRun) -> dict[str, Any]:
    payload = jsonable(run)
    if not isinstance(payload, dict):
        return {"result": payload}
    reports = payload.get("reports")
    if isinstance(reports, list):
        for index, report in enumerate(run.reports):
            if index < len(reports) and isinstance(reports[index], dict):
                reports[index]["segment_source"] = segment_source_label(report)
    return payload


def format_index_date(index: int | None, date: str | None) -> str:
    index_text = "-" if index is None else str(index)
    return f"{date or '-'} (index={index_text})"


def format_ref(ref: KRef | None) -> str:
    if ref is None:
        return "-"
    return f"{ref.date} index={ref.index} price={ref.price:.3f} role={ref.role}"


def format_pivot(report: Any) -> str:
    if report.pivot_confirm_k is None:
        return "-"
    zone = "-"
    if report.pivot_lower is not None and report.pivot_upper is not None:
        zone = f"[{report.pivot_lower:.3f}, {report.pivot_upper:.3f}]"
    line = "-" if report.pivot_line is None else f"{report.pivot_line:.3f}"
    return (
        f"{report.pivot_confirm_date or '-'} index={report.pivot_confirm_k}; "
        f"line={line}; type={report.pivot_type or '-'}; zone={zone}"
    )


def direction_label(direction: str) -> str:
    if direction == "up":
        return "向上"
    if direction == "down":
        return "向下"
    return direction


def normalize_adjust_option(value: str) -> str:
    return "" if value == ADJUST_EMPTY_LABEL else value


if __name__ == "__main__":
    run_gui()
