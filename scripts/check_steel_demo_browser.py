"""Optional headless Edge check and real-interaction timing for a running demo."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright, expect


def run(url, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new check output directory")
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="networkidle")
        expect(page.locator("#reset")).to_be_enabled()
        period = page.request.get(url + "/api/state").json()
        first_start = datetime.fromisoformat(period["first_start"])
        last_start = datetime.fromisoformat(period["last_start"])
        end = datetime.fromisoformat(period["end"])
        initial_start = min(first_start + timedelta(hours=8), last_start).isoformat(timespec="minutes")

        def start(value):
            page.locator("#start").fill(value)
            page.locator("#reset").click()
            expect(page.locator("#step")).to_be_enabled()
            return page.request.get(url+"/api/state").json()

        def step(final=False):
            page.locator("#step").click()
            if final:
                expect(page.locator("#playState")).to_contain_text("Đã đi hết dữ liệu")
                expect(page.locator("#reset")).to_be_enabled()
            else:
                expect(page.locator("#step")).to_be_enabled()
            return page.request.get(url+"/api/state").json()

        def no_future(state):
            assert all(row["time"] <= state["clock"] for row in state["history"])
            assert all(row["actual"] is None for row in state["forecasts"] if row["forecast_end"] > state["clock"])
            for row in state['forecasts']:
                q = row.get('probabilistic')
                if q:
                    assert all(step['actual_kWh'] is None for step in q['intervals'] if step['time'] > state['clock'])
                    if q['forecast_end'] > state['clock']:
                        assert q['hour_total']['actual_kWh'] is None and q['max_interval']['actual_kWh'] is None

        initial = start(initial_start)
        no_future(initial)
        assert initial["forecasts"][0]["actual"] is None
        quantiles_enabled = initial.get('probabilistic_info', {}).get('status') == 'configured' if initial.get('probabilistic_info') else False
        if quantiles_enabled:
            expect(page.locator('#probabilisticPanel')).to_be_visible()
            assert page.locator('#intervalRows tr').count() == 4
            assert page.locator('#intervalChart .actual').count() == 0
            expect(page.locator('#predictionHeading')).to_contain_text('MODEL MỚI')
            expect(page.locator('#alertHeading')).to_contain_text('phân vị')
            assert initial['metrics']['probabilistic_metrics']['point_hour']['rows'] == 0
        page.locator("#inspect").click()
        expect(page.locator("#decisionFeedback")).to_contain_text("Nhập tên")
        assert not page.request.get(url+"/api/state").json()["decisions"]
        page.locator("#operator").fill("QA trình duyệt · không phải thao tác nhà máy")
        note = "Kiểm tra nhật ký demo. <b>Chỉ là văn bản</b>"
        page.locator("#decisionNote").fill(note)
        page.locator("#inspect").click()
        expect(page.locator("#decisionFeedback")).to_contain_text("Đã lưu")
        recorded = page.request.get(url+"/api/state").json()
        assert len(recorded["decisions"]) == 1
        assert recorded["decisions"][0]["note"] == note
        context = recorded["decisions"][0]["context"]
        assert context["model_sha256"] == initial["current_forecast"]["prediction"]["model_sha256"]
        assert "actual" not in context and "actual_kWh" not in context
        current_decision = initial["current_forecast"].get("decision")
        if current_decision:
            assert context["decision"] == current_decision
            if current_decision.get("policy_sha256"):
                assert context["alert_policy"] == "configured"
                assert current_decision["policy_sha256"] == initial["policy_info"]["policy_sha256"]
                expected_upper = initial["current_forecast"]["prediction"]["predicted_next_60m_kWh"] + current_decision["upper_buffer_kWh"]
                assert abs(current_decision["upper_proxy_kWh"] - expected_upper) < 1e-9
                assert current_decision["alert"] == (expected_upper > current_decision["threshold_kWh"])
        if initial.get("alert_policy") == "not_configured":
            assert context["alert_policy"] == "not_configured"
        assert page.locator("#decisions b").count() == 0
        page.locator("#decisionNote").fill("Bỏ qua quan sát trong bài kiểm tra giao diện; không kết luận tải an toàn.")
        page.locator("#dismiss").click()
        expect(page.locator("#decisionFeedback")).to_contain_text("Đã lưu")
        page.reload(wait_until="networkidle")
        expect(page.locator("#step")).to_be_enabled()
        assert len(page.request.get(url+"/api/state").json()["decisions"]) == 2
        measurements = []
        for i in range(24):
            state = step()
            no_future(state)
            measurements.append(page.evaluate("window.lastRenderMeasurement"))
            if i < 3:
                assert state["metrics"]["models"][0]["all_matured"]["rows"] == 0
            if i == 3:
                assert state["metrics"]["models"][0]["all_matured"]["rows"] == 1
                page.screenshot(path=str(output / "desktop.png"), full_page=True)
            if quantiles_enabled and i < 4:
                q = next(row['probabilistic'] for row in state['forecasts'] if row['issue_time']==initial['clock'])
                assert sum(step['actual_kWh'] is not None for step in q['intervals']) == i+1
                assert page.locator('#intervalChart .actual').count() == i+1
                measured = {row['time']:row['usage_kWh'] for row in state['history']}
                for interval in q['intervals']:
                    if interval['actual_kWh'] is not None:
                        assert interval['actual_kWh'] == measured[interval['time']]
                assert state['decisions'][0]['context']['probabilistic'] == context['probabilistic']
                assert all(step['actual_kWh'] is None for step in context['probabilistic']['intervals'])
                if i==3:
                    assert state['metrics']['probabilistic_metrics']['point_hour']['rows'] == 1
                    expect(page.locator('#quantileCoverage')).to_contain_text('1 nhãn')
        # Play advances the clock; pause stops subsequent scheduled steps.
        before = page.locator("#clock").inner_text()
        page.locator("#play").click()
        expect(page.locator("#clock")).not_to_have_text(before)
        page.locator("#play").click()
        expect(page.locator("#step")).to_be_enabled()
        paused = page.locator("#clock").inner_text()
        page.wait_for_timeout(1800)
        assert page.locator("#clock").inner_text() == paused
        reset = start(initial_start)
        assert reset["session"] != initial["session"]
        assert reset["metrics"]["models"][0]["all_matured"]["rows"] == 0
        assert all(row["actual"] is None for row in reset["forecasts"])
        assert not reset["decisions"]
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1080})
        start(last_start.isoformat(timespec="minutes"))
        for i in range(4):
            final = step(final=i==3)
            no_future(final)
        assert final["finished"] and final["metrics"]["models"][0]["pending"] == 0
        expect(page.locator("#step")).to_be_disabled()
        expect(page.locator("#play")).to_be_disabled()
        outside = (end + timedelta(minutes=15)).isoformat(timespec="minutes")
        assert page.request.post(url+"/api/reset", data={"start": outside}).status == 400
        assert page.request.post(url+"/api/step", data={}, headers={"Origin":"https://example.invalid"}).status == 403
        assert page.request.post(url+"/api/decision", data={}, headers={"Origin":"https://example.invalid"}).status == 403
        assert not errors, errors
        summary = {"checks": "real-data navigation, delayed actuals, play/pause, isolated reset, configured-period end, mobile overflow, invalid-date and foreign-origin rejection; decision required fields, inspection/dismissal, immutable policy context, escaped text and persistence on reload",
                   "browser": browser.version, "javascript_errors": errors, "measured_step_requests": len(measurements),
                   "request_to_render_p50_ms": float(np.median([r["request_to_render_ms"] for r in measurements])),
                   "request_to_render_p95_ms": float(np.quantile([r["request_to_render_ms"] for r in measurements], .95)),
                   "backend_p95_ms": float(np.quantile([r["backend_ms"] for r in measurements], .95)),
                   "process_cpu_p95_ms": float(np.quantile([r["process_cpu_ms"] for r in measurements], .95)),
                   "server_rss_max_MB": max(r["process_rss_bytes"] for r in measurements)/1048576,
                   "model_bytes": final["performance"]["model_bytes"],
                   'total_model_bytes': final['performance'].get('total_model_bytes', final['performance']['model_bytes']),
                   "period_start": period["first_start"], "period_end": period["end"],
                   "alert_policy": initial.get("alert_policy"),
                   'probabilistic_enabled': quantiles_enabled,
                   'probabilistic_model_sha256': initial['probabilistic_info']['model_sha256'] if quantiles_enabled else None,
                   "policy_info": initial.get("policy_info"),
                   "model_info": initial.get("model_info"),
                   "saved_decision_context": context,
                   "scope": "Headless Edge, local server, 24 measured real-data steps in the configured replay period. RAM covers server only, not browser. Not a production or sustained-load benchmark."}
        (output / "browser_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "browser_timings.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
        # Leave the user's demo in a fresh, paused session.
        start(initial_start)
        browser.close()
        # The saved file is UTF-8; escaped console text avoids legacy Windows errors.
        print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.url.rstrip("/"), args.output_dir.resolve())
