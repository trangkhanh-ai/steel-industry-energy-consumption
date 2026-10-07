"""Sequential local replay controller. No future labels in browser payloads."""
from datetime import datetime, timezone
from pathlib import Path
import ctypes
import os
import sys
from time import perf_counter, process_time
from uuid import uuid4

import pandas as pd

from .monitoring import ForecastMonitor, valid_time
from .inference import DEFAULT_MODEL_SUBDIR
from .preprocessing import load_official_raw


def resident_memory_bytes():
    """Current process working set, not host RAM or peak Python allocations."""
    if sys.platform == "win32":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize
    elif sys.platform.startswith("linux"):
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return None


class ReplayDemo:
    START = pd.Timestamp("2018-09-01 00:00")
    LAST_ORIGIN = pd.Timestamp("2018-09-30 22:45")
    END = pd.Timestamp("2018-09-30 23:45")

    def __init__(self, root, model_dir=None, sessions_dir=None):
        self.root = Path(root)
        self.model_dir = Path(model_dir) if model_dir else self.root / DEFAULT_MODEL_SUBDIR
        self.sessions_dir = Path(sessions_dir) if sessions_dir else self.root / "reports/steel/modeling/demo_sessions"
        raw, self.audit = load_official_raw(self.root / "data/steel/raw/Steel_industry_data.csv")
        self.history = raw.loc[raw.observation_time.between(self.START-pd.Timedelta(days=7), self.END), ["observation_time", "Usage_kWh"]].reset_index(drop=True)
        self.monitor = None
        self.current = None
        self.last_response = None
        self.session = None
        self.timings = []

    def close(self):
        if self.monitor:
            self.monitor.close()
            self.monitor = None

    def _tick(self):
        observed = self.history.loc[self.history.observation_time.le(self.current)].tail(673)
        try:
            self.monitor.observe(observed, self.current)
        except (ValueError, TypeError):
            # observe already logs the rejected actuals. Forecast validation below
            # exposes the input failure instead of terminating replay with an HTTP 500.
            pass
        self.last_response = self.monitor.forecast(observed, self.current) if self.current <= self.LAST_ORIGIN else None

    def dispatch(self, action, start=None, decision=None):
        begin, cpu = perf_counter(), process_time()
        if action == "reset":
            t = valid_time(start)
            if not self.START <= t <= self.LAST_ORIGIN:
                raise ValueError("Chọn mốc 15 phút từ 01/09 00:00 đến 30/09/2018 22:45.")
            self.close()
            self.session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:10]
            self.monitor = ForecastMonitor(self.model_dir, self.sessions_dir / self.session / "monitor.sqlite3")
            self.current, self.timings = t, []
            self._tick()
        elif action == "step":
            if self.monitor is None:
                raise ValueError("Chọn thời điểm và mở phiên trước.")
            if self.current < self.END:
                self.current += pd.Timedelta(minutes=15)
                self._tick()
        elif action == "decision":
            if not isinstance(decision, dict) or self.monitor is None or self.last_response is None:
                raise ValueError("Cần một quan sát hiện tại để ghi nhật ký.")
            if (decision.get("session") != self.session or decision.get("clock") != self.current.isoformat()
                    or decision.get("request_id") != self.last_response["request_id"]):
                raise ValueError("Phiên hoặc thời điểm đã đổi. Tải lại trạng thái trước khi ghi nhật ký.")
            self.monitor.record_decision(self.last_response["request_id"], self.current,
                                         decision.get("operator"), decision.get("action"),
                                         decision.get("note"), decision.get("token"))
        elif action != "state":
            raise ValueError("Thao tác không hợp lệ.")
        payload = self.state()
        duration = 1000*(perf_counter()-begin)
        cpu_ms = 1000*(process_time()-cpu)
        if action in ("step", "reset"):
            self.timings.append(duration)
        timings = sorted(self.timings)
        # Linear empirical quantile, same definition as pandas quantile.
        p95 = float(pd.Series(timings).quantile(.95)) if timings else None
        model_path = self.model_dir / "selected_model.joblib"
        if self.monitor and self.monitor.model and self.monitor.model.duy_forecaster:
            relative = self.monitor.model.duy_forecaster.manifest["model_path"]
            model_path = self.model_dir / relative if relative else model_path
        payload["performance"] = {"backend_ms": duration, "process_cpu_ms": cpu_ms,
                                  "process_rss_bytes": resident_memory_bytes(), "session_backend_p95_ms": p95,
                                  "timed_actions": len(timings), "model_bytes": model_path.stat().st_size if model_path.exists() else None}
        return payload

    def state(self):
        base = {"session": self.session, "clock": self.current.isoformat() if self.current is not None else None,
                "first_start": self.START.isoformat(), "last_start": self.LAST_ORIGIN.isoformat(),
                "end": self.END.isoformat(), "finished": self.current == self.END,
                "alert_policy": "not_configured", "source": "UCI Steel · dữ liệu đo năm 2018",
                "history": [], "forecasts": [], "errors": [], "metrics": None, "current_forecast": None,
                "decisions": [], "model_info": None}
        if self.monitor is None:
            return base
        if self.monitor.model is not None:
            base["model_info"] = self.monitor.model.display_metadata()
        observed = self.history.loc[self.history.observation_time.le(self.current)].tail(96)
        base["history"] = [{"time": t.isoformat(), "usage_kWh": float(v)} for t,v in observed.itertuples(index=False, name=None)]
        base["current_forecast"] = self.last_response
        base["metrics"] = self.monitor.snapshot(self.current)
        base["decisions"] = self.monitor.decisions(self.current)
        rows = self.monitor.db.execute("SELECT * FROM forecasts WHERE issue_time<=? ORDER BY issue_time DESC LIMIT 20", (self.current.isoformat(),)).fetchall()
        for r in rows:
            ready = r["evaluated_as_of"] is not None and r["evaluated_as_of"] <= self.current.isoformat()
            base["forecasts"].append({"issue_time": r["issue_time"], "forecast_end": r["forecast_end"],
                                      "prediction": r["predicted_kWh"], "actual": r["actual_kWh"] if ready else None,
                                      "error": r["predicted_kWh"]-r["actual_kWh"] if ready else None,
                                      "label_status": "available" if ready else
                                      ("waiting_for_observations" if r["forecast_end"] <= self.current.isoformat() else "waiting")})
        base["errors"] = [dict(r) for r in self.monitor.db.execute("SELECT issue_time,status,detail FROM requests WHERE status!='forecast_ready' ORDER BY id DESC LIMIT 5")]
        return base
