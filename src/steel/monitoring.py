"""Local persistent forecast/error log and delayed scoring from real observations.

SQLite is a local sequential-demo store, not a production ingestion service.
No alert policy, synthetic observations, imputation or automatic retraining.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import json
from time import perf_counter

import numpy as np
import pandas as pd

from .inference import SteelForecaster, features_at_issue
from .preprocessing import FREQUENCY
from .policy import DEFAULT_POLICY, inspect_policy


def valid_time(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is not None or t != t.floor("15min"):
        raise ValueError("Expected an aligned, timezone-naive dataset timestamp")
    return t


class ForecastMonitor:
    def __init__(self, model_dir: Path, database: Path, policy_path=DEFAULT_POLICY):
        database = Path(database)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(database)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY, issue_time TEXT NOT NULL,
                forecast_start TEXT NOT NULL, forecast_end TEXT NOT NULL,
                predicted_kWh REAL NOT NULL, model_sha256 TEXT NOT NULL,
                model_configuration TEXT NOT NULL, actual_kWh REAL,
                evaluated_as_of TEXT,
                UNIQUE(issue_time, model_sha256)
            );
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY, logged_utc TEXT NOT NULL,
                issue_time TEXT, status TEXT NOT NULL, detail TEXT,
                elapsed_ms REAL NOT NULL, forecast_id INTEGER REFERENCES forecasts(id)
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE,
                request_id INTEGER NOT NULL REFERENCES requests(id),
                recorded_utc TEXT NOT NULL, replay_time TEXT NOT NULL,
                operator TEXT NOT NULL, action TEXT NOT NULL,
                note TEXT NOT NULL, context_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
        """)
        self.model = None
        self.model_error = None
        try:
            self.model = SteelForecaster(model_dir)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError) as error:
            self.model_error = f"{type(error).__name__}: {error}"
            self._log(None, "model_unavailable", self.model_error, 0, None)
        saved = self.db.execute("SELECT value FROM session_metadata WHERE key='policy_inspection'").fetchone()
        if saved is None:
            self.policy_info = inspect_policy(policy_path, self.model)
            self.db.execute("INSERT INTO session_metadata(key,value) VALUES('policy_inspection',?)",
                            (json.dumps(self.policy_info, ensure_ascii=False, allow_nan=False),))
            self.db.commit()
        else:
            # Preserve the handoff observed by this session, including after reopen.
            self.policy_info = json.loads(saved["value"])

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _log(self, issue, status, detail, elapsed, forecast_id):
        cursor = self.db.execute("INSERT INTO requests(logged_utc,issue_time,status,detail,elapsed_ms,forecast_id) VALUES(?,?,?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(), issue, status, detail, elapsed, forecast_id))
        self.db.commit()
        return cursor.lastrowid

    def record_decision(self, request_id, as_of, operator, action, note, token):
        """Append a human observation, never an equipment command or model label.

        Context is taken from the saved request, not supplied by the browser.
        A retry token prevents duplicate writes; corrections are new entries.
        Operator is a self-declared name in this local, unauthenticated demo.
        """
        now = valid_time(as_of).isoformat()
        if type(request_id) is not int:
            raise ValueError("Yêu cầu quan sát không hợp lệ.")
        for value, limit in ((operator, 80), (note, 500), (token, 100)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError("Cần tên người ghi (tối đa 80 ký tự), ghi chú (500) và mã thao tác hợp lệ.")
        if action not in ("acknowledge", "inspect", "dismiss"):
            raise ValueError("Chỉ ghi nhận đã xem, đề nghị kiểm tra hoặc bỏ qua.")
        operator, note = operator.strip(), note.strip()
        row = self.db.execute("""SELECT r.*, f.predicted_kWh, f.model_sha256,
            f.model_configuration FROM requests r LEFT JOIN forecasts f ON f.id=r.forecast_id
            WHERE r.id=?""", (request_id,)).fetchone()
        if row is None or row["issue_time"] != now:
            raise ValueError("Chỉ ghi quyết định cho quan sát tại thời điểm đang phát lại.")
        if action == "acknowledge" and row["status"] != "forecast_ready":
            raise ValueError("Không xác nhận dự báo khi dữ liệu hoặc mô hình đang lỗi; hãy đề nghị kiểm tra.")
        context = {key: row[key] for key in ("issue_time", "status", "detail", "forecast_id",
                   "predicted_kWh", "model_sha256", "model_configuration")}
        # This monitor has no calibrated alert policy. Never label these as alert decisions.
        context["alert_policy"] = "not_configured"
        context["policy_inspection"] = self.policy_info
        context_json = json.dumps(context, ensure_ascii=False, allow_nan=False, sort_keys=True)
        values = (request_id, now, operator, action, note, context_json)
        previous = self.db.execute("SELECT * FROM decisions WHERE token=?", (token,)).fetchone()
        if previous:
            fields = ("request_id", "replay_time", "operator", "action", "note", "context_json")
            if tuple(previous[key] for key in fields) != values:
                raise ValueError("Mã thao tác đã dùng cho nội dung khác; nhật ký cũ được giữ nguyên.")
            return previous["id"]
        with self.db:
            cursor = self.db.execute("""INSERT INTO decisions
                (token,recorded_utc,request_id,replay_time,operator,action,note,context_json)
                VALUES(?,?,?,?,?,?,?,?)""", (token, datetime.now(timezone.utc).isoformat(), *values))
        return cursor.lastrowid

    def decisions(self, as_of, limit=20):
        rows = self.db.execute("SELECT * FROM decisions WHERE replay_time<=? ORDER BY id DESC LIMIT ?",
                               (valid_time(as_of).isoformat(), limit)).fetchall()
        return [{**{key: row[key] for key in ("id", "recorded_utc", "replay_time", "operator", "action", "note")},
                 "context": json.loads(row["context_json"])} for row in rows]

    def forecast(self, history: pd.DataFrame, issue_time) -> dict:
        start = perf_counter()
        issue = str(issue_time)
        status, detail, forecast_id, prediction = "invalid_history", None, None, None
        if self.model is None:
            status, detail = "model_unavailable", self.model_error
            # Use the same timestamp format as replay decisions and metric filters,
            # even when no model can be loaded. str(Timestamp) uses a space.
            try:
                issue = valid_time(issue_time).isoformat()
            except (ValueError, TypeError):
                pass
        else:
            try:
                t = valid_time(issue_time)
                issue = t.isoformat()
                # Separate invalid observations from a model execution failure.
                features_at_issue(history, t)
            except (ValueError, TypeError, AttributeError, KeyError) as error:
                detail = str(error)
            else:
                try:
                    prediction = self.model.predict(history, t)
                except (ValueError, TypeError, RuntimeError, OSError) as error:
                    status, detail = "prediction_failed", str(error)
                else:
                    self.db.execute("""INSERT OR IGNORE INTO forecasts
                        (issue_time,forecast_start,forecast_end,predicted_kWh,model_sha256,model_configuration)
                        VALUES(?,?,?,?,?,?)""", (issue, prediction["forecast_start"], prediction["forecast_end"],
                        prediction["predicted_next_60m_kWh"], prediction["model_sha256"], prediction["model_configuration"]))
                    saved = self.db.execute("SELECT id,predicted_kWh FROM forecasts WHERE issue_time=? AND model_sha256=?",
                                            (issue, prediction["model_sha256"])).fetchone()
                    forecast_id = saved["id"]
                    if saved["predicted_kWh"] != prediction["predicted_next_60m_kWh"]:
                        # Never replace a forecast after its outcome could be known.
                        status, detail, prediction = "forecast_conflict", "Different prediction already stored for this timestamp/model", None
                    else:
                        status = "forecast_ready"
        elapsed = 1000*(perf_counter()-start)
        request_id = self._log(issue, status, detail, elapsed, forecast_id)
        return {"status": status, "detail": detail, "forecast_id": forecast_id, "request_id": request_id,
                "prediction": prediction, "request_ms": elapsed}

    def observe(self, history: pd.DataFrame, as_of) -> dict:
        try:
            return self._observe(history, as_of)
        except (ValueError, TypeError) as error:
            self._log(str(as_of), "invalid_actual_history", str(error), 0, None)
            raise

    def _observe(self, history: pd.DataFrame, as_of) -> dict:
        """Score only matured forecasts with all four measured future intervals.

        The caller supplies observations already available at as_of, never a
        whole future CSV. Gaps are allowed here but prevent labels being scored.
        Existing labels are immutable; corrections require a separate reviewed run.
        """
        now = valid_time(as_of)
        try:
            times = pd.to_datetime(history.observation_time, errors="raise")
            usage = pd.to_numeric(history.Usage_kWh, errors="raise").to_numpy(dtype=float)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError("Observation history needs valid observation_time and Usage_kWh") from error
        if times.dt.tz is not None or times.isna().any() or not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError("Observation timestamps must be unique, sorted and timezone-naive")
        if not times.eq(times.dt.floor("15min")).all() or times.gt(now).any():
            raise ValueError("Observations must be aligned and available at or before as_of")
        if not np.isfinite(usage).all() or (usage < 0).any():
            raise ValueError("Observations must be finite and nonnegative")
        observed = pd.Series(usage, index=pd.DatetimeIndex(times))
        pending = self.db.execute("SELECT * FROM forecasts WHERE actual_kWh IS NULL AND forecast_end<=? ORDER BY forecast_end,id", (now.isoformat(),)).fetchall()
        evaluated, missing = 0, 0
        for row in pending:
            required = pd.date_range(row["forecast_start"], row["forecast_end"], freq=FREQUENCY)
            measured = observed.reindex(required)
            if len(measured) != 4 or measured.isna().any():
                missing += 1
                continue
            self.db.execute("UPDATE forecasts SET actual_kWh=?,evaluated_as_of=? WHERE id=? AND actual_kWh IS NULL",
                            (float(measured.sum()), now.isoformat(), row["id"]))
            evaluated += 1
        self.db.commit()
        return {"as_of": now.isoformat(), "newly_evaluated": evaluated, "matured_waiting_for_observations": missing}

    def snapshot(self, as_of) -> dict:
        """Metrics use only labels recorded by as_of; 24h is a display window."""
        now = valid_time(as_of)
        frame = pd.read_sql_query("SELECT * FROM forecasts WHERE issue_time<=?", self.db, params=(now.isoformat(),))
        result = {"as_of": now.isoformat(), "forecasts": len(frame), "models": [],
                  "request_counts_all_logged": dict(self.db.execute("SELECT status,COUNT(*) FROM requests GROUP BY status").fetchall()),
                  "policy": "monitoring_only_no_threshold_or_automatic_action"}
        counts = dict(self.db.execute("""SELECT status,COUNT(*) FROM requests
            WHERE issue_time<=? AND status IN
            ('forecast_ready','invalid_history','model_unavailable','prediction_failed','forecast_conflict')
            GROUP BY status""", (now.isoformat(),)).fetchall())
        attempts = sum(counts.values())
        invalid = counts.get("invalid_history", 0)
        result["input_quality"] = {"forecast_attempts": attempts, "invalid_history_attempts": invalid,
                                   "invalid_history_rate": invalid / attempts if attempts else None}
        if frame.empty:
            return result
        scored = frame.loc[frame.actual_kWh.notna() & frame.evaluated_as_of.le(now.isoformat())].copy()
        for sha, all_rows in frame.groupby("model_sha256"):
            part = scored.loc[scored.model_sha256.eq(sha)]
            recent = part.loc[part.forecast_end.gt((now-pd.Timedelta(hours=24)).isoformat())]
            def errors(rows):
                if rows.empty:
                    return {"rows": 0, "mae_kWh": None, "bias_kWh": None}
                error = rows.predicted_kWh-rows.actual_kWh
                return {"rows": len(rows), "mae_kWh": float(error.abs().mean()), "bias_kWh": float(error.mean())}
            result["models"].append({"model_sha256": sha, "model_configuration": all_rows.model_configuration.iloc[0],
                                     "pending": len(all_rows)-len(part), "all_matured": errors(part), "last_24h_by_forecast_end": errors(recent)})
        return result
