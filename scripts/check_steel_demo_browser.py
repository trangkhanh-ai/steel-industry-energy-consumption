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
        assert initial["forecasts"][0]["actual"] is None
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
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1080})
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
        summary = {"checks": "real-data navigation, delayed actuals, play/pause, isolated reset, month end, mobile overflow, invalid-date and foreign-origin rejection; decision required fields, inspection/dismissal, escaped text and persistence on reload",
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
