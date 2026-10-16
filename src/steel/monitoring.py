"""Local persistent forecast/error log and delayed scoring from real observations.

SQLite is a local sequential-demo store, not a production ingestion service.
Optional research policy; no synthetic observations or automatic equipment actions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import json
from time import perf_counter

import numpy as np
import pandas as pd

from .inference import SteelForecaster
from .preprocessing import FREQUENCY
from .policy import AlertPolicy, performance_watch, DEFAULT_POLICY, inspect_policy
from .evaluation import evaluate_alerts


def valid_time(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is not None or t != t.floor("15min"):
        raise ValueError("Expected an aligned, timezone-naive dataset timestamp")
    return t


class ForecastMonitor:
    def __init__(self, model_dir: Path, database: Path, policy_path=None, probabilistic_dir=None):
        database = Path(database)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(database)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        # WAL giảm thao tác journal mỗi lần ghi; FULL vẫn đồng bộ transaction.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
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
        """)
        self.db.execute("CREATE TABLE IF NOT EXISTS session_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute('CREATE INDEX IF NOT EXISTS forecast_pending ON forecasts(forecast_end) WHERE actual_kWh IS NULL')
        self._forecast_contexts = {}
        self._context_data_version = self.db.execute('PRAGMA data_version').fetchone()[0]
        self.model = None
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(forecasts)")}
        if "decision_json" not in columns:
            self.db.execute("ALTER TABLE forecasts ADD COLUMN decision_json TEXT")
        if "probabilistic_json" not in columns:
            self.db.execute("ALTER TABLE forecasts ADD COLUMN probabilistic_json TEXT")
        self.policy, self.policy_error = None, None
        self.model_error = None
        try:
            self.model = SteelForecaster(model_dir)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError) as error:
            self.model_error = f"{type(error).__name__}: {error}"
            self._log(None, "model_unavailable", self.model_error, 0, None)
        saved = self.db.execute("SELECT value FROM session_metadata WHERE key='policy_inspection'").fetchone()
        if saved is None:
            # Historical LightGBM handoff is inspected separately; it never enables alerts.
            handoff_path = DEFAULT_POLICY
            if policy_path is not None:
                try:
                    candidate = json.loads(Path(policy_path).read_text(encoding="utf-8-sig"))
                    if not isinstance(candidate, dict) or "policy_metadata" in candidate:
                        handoff_path = policy_path
                except (OSError, ValueError):
                    handoff_path = policy_path
            self.policy_info = inspect_policy(handoff_path, self.model)
            self.db.execute("INSERT INTO session_metadata(key,value) VALUES('policy_inspection',?)",
                            (json.dumps(self.policy_info, ensure_ascii=False, allow_nan=False),))
            self.db.commit()
        else:
            self.policy_info = json.loads(saved["value"])
        if policy_path is not None and self.model is not None:
            try:
                if self.model.duy_forecaster is None:
                    raise ValueError("Policy yêu cầu artifact Duy có model_handoff.json")
                self.policy = AlertPolicy(policy_path, self.model.duy_forecaster.manifest)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.policy_error = str(error)
                self._log(None, "policy_unavailable", self.policy_error, 0, None)
        elif policy_path is not None:
            self.policy_error = "Không thể dùng policy khi model chưa được nạp hợp lệ"
        self.probabilistic, self.probabilistic_error = None, None
        if probabilistic_dir is not None:
            try:
                from .probabilistic import ProbabilisticForecaster
                if self.model is None or self.model.duy_forecaster is None:
                    raise ValueError('Phân vị cần model nguồn Duy có manifest hợp lệ')
                source = self.model.duy_forecaster.manifest['source_sha256']
                self.probabilistic = ProbabilisticForecaster(probabilistic_dir, source)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError) as error:
                self.probabilistic_error = str(error)
                self._log(None, 'probabilistic_unavailable', str(error), 0, None)

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
            f.model_configuration, f.decision_json, f.probabilistic_json FROM requests r LEFT JOIN forecasts f ON f.id=r.forecast_id
            WHERE r.id=?""", (request_id,)).fetchone()
        if row is None or row["issue_time"] != now:
            raise ValueError("Chỉ ghi quyết định cho quan sát tại thời điểm đang phát lại.")
        if action == "acknowledge" and row["status"] != "forecast_ready":
            raise ValueError("Không xác nhận dự báo khi dữ liệu hoặc mô hình đang lỗi; hãy đề nghị kiểm tra.")
        context = {key: row[key] for key in ("issue_time", "status", "detail", "forecast_id",
                   "predicted_kWh", "model_sha256", "model_configuration")}
        context["decision"] = json.loads(row["decision_json"]) if row["decision_json"] else None
        context["alert_policy"] = "configured" if context["decision"] and context["decision"].get("policy_sha256") else "not_configured"
        context["policy_inspection"] = self.policy_info
        context['probabilistic'] = json.loads(row['probabilistic_json']) if row['probabilistic_json'] else None
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
        try:
            issue = valid_time(issue_time).isoformat()
        except (ValueError, TypeError):
            issue = None
        status, detail, forecast_id, prediction = "invalid_history", None, None, None
        probabilistic, probabilistic_error = None, self.probabilistic_error
        decision = {"status": "not_configured", "alert": None,
                    "recommendation": "Chỉ dự báo; chưa có policy cảnh báo."}
        usage_watch = None
        if self.model is None:
            status, detail = "model_unavailable", self.model_error
        else:
            try:
                t = valid_time(issue_time)
                issue = t.isoformat()
                # Separate invalid observations from a model execution failure.
                features = self.model.prepare_features(history, t)
            except (ValueError, TypeError, AttributeError, KeyError) as error:
                detail = str(error)
            else:
                try:
                    prediction = self.model.predict_features(features, t, history)
                except (ValueError, TypeError, RuntimeError, OSError) as error:
                    status, detail = "prediction_failed", str(error)
                else:
                    if self.probabilistic is not None:
                        try:
                            probabilistic = self.probabilistic.predict_features(features, t)
                        except (ValueError, TypeError, RuntimeError) as error:
                            probabilistic_error = str(error)
                            self._log(issue, 'probabilistic_failed', str(error), 0, None)
                    if self.policy_error:
                        decision = {"status": "policy_unavailable", "alert": None, "detail": self.policy_error,
                                    "recommendation": "Kiểm phiên bản policy; không suy ra an toàn từ dự báo thấp."}
                    elif self.policy is not None:
                        try:
                            decision = self.policy.apply(prediction["predicted_next_60m_kWh"], t)
                            usage_watch = self.policy.usage_watch(history.Usage_kWh)
                        except (ValueError, KeyError, TypeError) as error:
                            decision = {"status": "policy_unavailable", "alert": None, "detail": str(error),
                                        "recommendation": "Kiểm policy và thời điểm calibration."}
                    decision_json = json.dumps(decision, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    probabilistic_json = json.dumps(probabilistic, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    self.db.execute("""INSERT OR IGNORE INTO forecasts
                        (issue_time,forecast_start,forecast_end,predicted_kWh,model_sha256,model_configuration,decision_json,probabilistic_json)
                        VALUES(?,?,?,?,?,?,?,?)""", (issue, prediction["forecast_start"], prediction["forecast_end"],
                        prediction["predicted_next_60m_kWh"], prediction["model_sha256"], prediction["model_configuration"], decision_json, probabilistic_json))
                    saved = self.db.execute("SELECT id,predicted_kWh,decision_json,probabilistic_json FROM forecasts WHERE issue_time=? AND model_sha256=?",
                                            (issue, prediction["model_sha256"])).fetchone()
                    forecast_id = saved["id"]
                    if (saved["predicted_kWh"] != prediction["predicted_next_60m_kWh"]
                            or saved["decision_json"] != decision_json
                            or (saved['probabilistic_json'] or 'null') != probabilistic_json):
                        # Never replace a forecast after its outcome could be known.
                        status, detail, prediction = "forecast_conflict", "Different prediction already stored for this timestamp/model", None
                    else:
                        status = "forecast_ready"
        elapsed = 1000*(perf_counter()-start)
        request_id = self._log(issue, status, detail, elapsed, forecast_id)
        if status != "forecast_ready":
            decision = {"status": "unavailable", "alert": None, "recommendation": "Kiểm dữ liệu/model trước khi sử dụng."}
        return {"status": status, "detail": detail, "forecast_id": forecast_id, "request_id": request_id,
                "prediction": prediction, "decision": decision, "usage_watch": usage_watch, "request_ms": elapsed,
                'probabilistic': probabilistic if status=='forecast_ready' else None,
                'probabilistic_error': probabilistic_error}

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

    def snapshot(self, as_of, include_alert_metrics=True) -> dict:
        """Metrics use only labels recorded by as_of; 24h is a display window."""
        now = valid_time(as_of)
        # JSON dự báo bất biến: chỉ đọc/giải mã một lần mỗi id. Actual vẫn đọc
        # lại và lọc evaluated_as_of, nên cache không làm nhãn xuất hiện sớm.
        version = self.db.execute('PRAGMA data_version').fetchone()[0]
        if version != self._context_data_version:
            self._forecast_contexts.clear()  # Một connection khác đã ghi DB.
            self._context_data_version = version
        last = max(self._forecast_contexts, default=0)
        for row in self.db.execute('SELECT id,decision_json,probabilistic_json FROM forecasts WHERE id>?', (last,)):
            self._forecast_contexts[row['id']] = (
                json.loads(row['decision_json']) if row['decision_json'] else {},
                json.loads(row['probabilistic_json']) if row['probabilistic_json'] else None)
        frame = pd.read_sql_query('''SELECT id,issue_time,forecast_start,forecast_end,
            predicted_kWh,model_sha256,model_configuration,actual_kWh,evaluated_as_of
            FROM forecasts WHERE issue_time<=?''', self.db, params=(now.isoformat(),))
        result = {"as_of": now.isoformat(), "forecasts": len(frame), "models": [],
                  "request_counts_all_logged": dict(self.db.execute("SELECT status,COUNT(*) FROM requests GROUP BY status").fetchall()),
                  "policy": "monitoring_only_no_threshold_or_automatic_action"}
        if self.policy is not None:
            result["policy"] = self.policy.information()
        if self.policy_error:
            result["policy_error"] = self.policy_error
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
            if self.policy is not None and sha == self.policy.data["model_sha256"]:
                same_policy = part.id.map(lambda value: self._forecast_contexts[value][0].get('policy_sha256') == self.policy.sha256)
                eligible = part.loc[same_policy]
                result["performance_watch"] = performance_watch(eligible, self.policy.reference_mae)
                result["alert_metrics"] = evaluate_alerts(eligible.actual_kWh, eligible.predicted_kWh,
                    pd.to_datetime(eligible.issue_time), self.policy.threshold, self.policy.buffer) if include_alert_metrics and len(eligible) else None
                result["alert_metrics_status"] = "computed" if include_alert_metrics else "available_on_demand"
        # Tính metric model mới theo yêu cầu; chỉ dùng tổng actual đã đủ 4 bản đo.
        if self.probabilistic is not None:
            from .probabilistic import interval_metrics, research_alert_metrics
            records = []
            for row in scored.itertuples():
                saved = self._forecast_contexts[row.id][1]
                if saved and saved['model_sha256'] == self.probabilistic.metadata['model_sha256']:
                    records.append((row.issue_time, row.actual_kWh, saved))
            result['probabilistic_metrics'] = {'rows': len(records), 'hour_total': None,
                'point_hour': {'rows': 0, 'mae_kWh': None, 'bias_kWh': None},
                'model_sha256': self.probabilistic.metadata['model_sha256'],
                'research_alert': None, 'alert_metrics_status': 'computed' if include_alert_metrics else 'available_on_demand'}
            reference = self.probabilistic.monitoring_reference
            if reference:
                from .reliability import reliability_watch
                result['probabilistic_metrics']['performance_watch'] = reliability_watch([], [], [], [], [], reference)
            if records:
                times = [r[0] for r in records]
                actual = np.array([r[1] for r in records])
                points = np.array([r[2]['forecast_sum_of_interval_points_kWh'] for r in records])
                result['probabilistic_metrics']['point_hour'] = {'rows': len(records),
                    'mae_kWh': float(np.mean(np.abs(points-actual))), 'bias_kWh': float(np.mean(points-actual))}
                q = np.array([[r[2]['hour_total'][f'p{p}_kWh'] for p in (10, 50, 90)] for r in records])
                margin = self.probabilistic.metadata['calibration_margins_kWh']['hour_total']
                threshold = self.probabilistic.metadata['train_research_thresholds_kWh']['hour_total']
                result['probabilistic_metrics']['hour_total'] = interval_metrics(actual, q, margin)
                if reference:
                    result['probabilistic_metrics']['performance_watch'] = reliability_watch(
                        times, actual, points, np.maximum(q[:, 0]-margin, 0), q[:, 2]+margin, reference)
                if include_alert_metrics:
                    result['probabilistic_metrics']['research_alert'] = research_alert_metrics(actual, q, times, threshold, margin)
        return result
