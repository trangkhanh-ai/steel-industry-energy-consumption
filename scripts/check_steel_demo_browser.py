"""Optional headless Edge check and real-interaction timing for a running demo."""
import argparse
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

        def start(value):
            page.locator("#start").fill(value)
            page.locator("#reset").click()
            expect(page.locator("#step")).to_be_enabled()
            return page.request.get(url+"/api/state").json()

        def step(final=False):
            page.locator("#step").click()
            if final:
                expect(page.locator("#playState")).to_have_text("Đã đi hết dữ liệu tháng 9.")
                expect(page.locator("#reset")).to_be_enabled()
            else:
                expect(page.locator("#step")).to_be_enabled()
            return page.request.get(url+"/api/state").json()

        def no_future(state):
            assert all(row["time"] <= state["clock"] for row in state["history"])
            assert all(row["actual"] is None for row in state["forecasts"] if row["forecast_end"] > state["clock"])

        initial = start("2018-09-01T08:00")
        assert initial["policy_info"]["status"] == "incompatible"
        assert not initial["policy_info"]["alerts_enabled"]
        expect(page.locator("#policyStatus")).to_have_text("Policy khác model đang chạy")
        expect(page.locator("#policyIssues")).to_contain_text("SHA-256")
        assert initial["forecasts"][0]["actual"] is None
        page.locator("#modelDetails").evaluate("el => el.open = true")
        expect(page.locator("#modelName")).to_have_text(initial["model_info"]["name"])
        expect(page.locator("#modelHash")).to_have_text(initial["current_forecast"]["prediction"]["model_sha256"])
        expected_mae = page.evaluate("v => new Intl.NumberFormat('vi-VN', {maximumFractionDigits:2, minimumFractionDigits:2}).format(v)", initial["model_info"]["validation_mae_kWh"])
        expect(page.locator("#modelMae")).to_have_text(expected_mae)
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
        assert recorded["decisions"][0]["context"]["alert_policy"] == "not_configured"
        assert recorded["decisions"][0]["context"]["policy_inspection"] == initial["policy_info"]
        assert page.locator("#decisions b").count() == 0
        page.locator("#decisionNote").fill("Bỏ qua quan sát trong bài kiểm tra giao diện; không kết luận tải an toàn.")
        page.locator("#dismiss").click()
        expect(page.locator("#decisionFeedback")).to_contain_text("Đã lưu")
        page.reload(wait_until="networkidle")
        expect(page.locator("#step")).to_be_enabled()
        assert len(page.request.get(url+"/api/state").json()["decisions"]) == 2
        page.locator("#operator").fill("QA trình duyệt · không phải thao tác nhà máy")
        page.locator("#decisionNote").fill("Đã xem dự báo; chưa kết luận tải an toàn.")
        page.locator("#acknowledge").click()
        expect(page.locator("#decisionFeedback")).to_contain_text("Đã lưu")
        assert len(page.request.get(url+"/api/state").json()["decisions"]) == 3
        measurements = []
        for i in range(24):
            state = step()
            no_future(state)
            measurements.append(page.evaluate("window.lastRenderMeasurement"))
            if i < 3:
                assert state["metrics"]["models"][0]["all_matured"]["rows"] == 0
            if i == 3:
                assert state["metrics"]["models"][0]["all_matured"]["rows"] == 1
                page.locator("#modelDetails").evaluate("el => el.open = true")
                page.screenshot(path=str(output / "desktop.png"), full_page=True)
        # Play advances the clock; pause stops subsequent scheduled steps.
        before = page.locator("#clock").inner_text()
        page.locator("#play").click()
        expect(page.locator("#clock")).not_to_have_text(before)
        page.locator("#play").click()
        expect(page.locator("#step")).to_be_enabled()
        paused = page.locator("#clock").inner_text()
        page.wait_for_timeout(1800)
        assert page.locator("#clock").inner_text() == paused
        reset = start("2018-09-01T08:00")
        assert reset["session"] != initial["session"]
        assert reset["metrics"]["models"][0]["all_matured"]["rows"] == 0
        assert all(row["actual"] is None for row in reset["forecasts"])
        assert not reset["decisions"]
        page.locator("#modelDetails").evaluate("el => el.open = true")
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1080})
        midnight = start("2018-09-01T23:45")
        for i in range(4):
            crossed = step()
            no_future(crossed)
            assert crossed["session"] == midnight["session"]
            assert crossed["metrics"]["models"][0]["all_matured"]["rows"] == int(i == 3)
        assert crossed["clock"] == "2018-09-02T00:45:00"
        # A second tab retains the previous clock/session until refreshed.
        stale = browser.new_page()
        stale.on("pageerror", lambda error: errors.append(str(error)))
        for change in ("clock", "session"):
            stale.goto(url, wait_until="networkidle")
            expect(stale.locator("#inspect")).to_be_enabled()
            stale.locator("#operator").fill("QA trang cũ")
            stale.locator("#decisionNote").fill("Ghi nhận từ trang chưa cập nhật.")
            if change == "clock":
                step()
            else:
                start("2018-09-02T00:45")
            stale.locator("#inspect").click()
            expect(stale.locator("#error")).to_contain_text("Phiên hoặc thời điểm đã đổi")
            assert not page.request.get(url+"/api/state").json()["decisions"]
        stale.close()
        current = page.request.get(url+"/api/state").json()
        command = {"session": current["session"], "clock": current["clock"],
                   "request_id": current["current_forecast"]["request_id"],
                   "operator": "QA retry", "action": "inspect", "note": "Kiểm gửi lại yêu cầu.",
                   "token": "browser-retry-check"}
        for _ in range(2):
            assert page.request.post(url+"/api/decision", data=command).status == 200
        assert len(page.request.get(url+"/api/state").json()["decisions"]) == 1
        assert page.request.post(url+"/api/decision", data={**command, "note": "Khác nội dung"}).status == 400
        assert len(page.request.get(url+"/api/state").json()["decisions"]) == 1
        start("2018-09-30T22:45")
        for i in range(4):
            final = step(final=i==3)
            no_future(final)
        assert final["finished"] and final["metrics"]["models"][0]["pending"] == 0
        expect(page.locator("#step")).to_be_disabled()
        expect(page.locator("#play")).to_be_disabled()
        assert page.request.post(url+"/api/reset", data={"start":"2018-10-01T00:00"}).status == 400
        assert page.request.post(url+"/api/step", data={}, headers={"Origin":"https://example.invalid"}).status == 403
        assert page.request.post(url+"/api/decision", data={}, headers={"Origin":"https://example.invalid"}).status == 403
        assert not errors, errors
        summary = {"checks": "loaded-model metadata and validation MAE; real-data navigation, delayed actuals, midnight, play/pause, isolated reset, month end, mobile overflow, invalid-date and foreign-origin rejection; decision required fields, acknowledge/inspect/dismiss, escaped text and persistence on reload",
                   "model_info": initial["model_info"],
                   "policy_info": initial["policy_info"],
                   "decision_guards": "stale clock/session rejected through UI; identical HTTP retry stored once; changed retry content rejected",
                   "browser": browser.version, "javascript_errors": errors, "measured_step_requests": len(measurements),
                   "request_to_render_p50_ms": float(np.median([r["request_to_render_ms"] for r in measurements])),
                   "request_to_render_p95_ms": float(np.quantile([r["request_to_render_ms"] for r in measurements], .95)),
                   "backend_p95_ms": float(np.quantile([r["backend_ms"] for r in measurements], .95)),
                   "process_cpu_p95_ms": float(np.quantile([r["process_cpu_ms"] for r in measurements], .95)),
                   "server_rss_max_MB": max(r["process_rss_bytes"] for r in measurements)/1048576,
                   "model_bytes": final["performance"]["model_bytes"],
                   "scope": "Headless Edge, local server, 24 measured real September steps. RAM covers server only, not browser. Not a production or sustained-load benchmark."}
        (output / "browser_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "browser_timings.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
        # Leave the user's demo in a fresh, paused session.
        start("2018-09-01T08:00")
        browser.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.url.rstrip("/"), args.output_dir.resolve())
