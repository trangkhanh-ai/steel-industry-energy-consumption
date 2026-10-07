"""Browser failure checks using real-row subsets and temporary artifact copies.

Never changes the source CSV/model or adds a production fault-injection endpoint.
"""
import argparse
from contextlib import contextmanager
from http.server import HTTPServer
import json
from pathlib import Path
from queue import Queue
import shutil
from tempfile import TemporaryDirectory
from threading import Thread

import pandas as pd
from playwright.sync_api import sync_playwright, expect

from scripts.serve_steel_demo import ROOT, make_handler
from src.steel.demo import ReplayDemo
from src.steel.inference import DEFAULT_MODEL_SUBDIR


@contextmanager
def scenario_server(scenario, directory):
    """Create/use/close SQLite on its owning server thread."""
    ready = Queue()

    def serve():
        demo = None
        try:
            model = ROOT / DEFAULT_MODEL_SUBDIR
            if scenario == "missing_model":
                model = directory / "absent"
            elif scenario == "checksum_mismatch":
                original = model
                model = directory / "artifact_copy"
                model.mkdir()
                shutil.copy2(original / "model_handoff.json", model)
                manifest = json.loads((model / "model_handoff.json").read_text())
                artifact = model / manifest["model_path"]
                artifact.parent.mkdir(parents=True)
                shutil.copy2(original / manifest["model_path"], artifact)
                with artifact.open("ab") as file:
                    file.write(b"checksum-verification-only")
            demo = ReplayDemo(ROOT, model_dir=model, sessions_dir=directory / "sessions")
            if scenario in ("missing_real_row", "missing_actual_row"):
                missing_time = "08:30" if scenario == "missing_actual_row" else "08:00"
                demo.history = demo.history.loc[
                    demo.history.observation_time.ne(pd.Timestamp("2018-09-01 " + missing_time))]
            with HTTPServer(("127.0.0.1", 0), make_handler(demo)) as server:
                ready.put(server)
                server.serve_forever()
        except Exception as error:
            ready.put(error)
        finally:
            if demo is not None:
                demo.close()

    thread = Thread(target=serve, daemon=True)
    thread.start()
    server = ready.get(timeout=30)
    if isinstance(server, Exception):
        raise server
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("QA server did not stop")


def run(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new check output directory")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        for scenario in ("missing_model", "checksum_mismatch", "missing_real_row", "missing_actual_row"):
            with TemporaryDirectory() as temporary, scenario_server(scenario, Path(temporary)) as url:
                page = browser.new_page(viewport={"width": 1440, "height": 1080})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url, wait_until="networkidle")
                expect(page.locator("#reset")).to_be_enabled()
                page.locator("#start").fill("2018-09-01T08:00")
                page.locator("#reset").click()
                expect(page.locator("#inspect")).to_be_enabled()
                if scenario == "missing_actual_row":
                    # The first forecast is valid. Later, withhold one real actual
                    # in memory and verify elapsed time alone cannot score it.
                    expect(page.locator("#acknowledge")).to_be_enabled()
                    for _ in range(4):
                        page.locator("#step").click()
                        expect(page.locator("#step")).to_be_enabled()
                    row = page.locator("#rows tr").filter(has_text="01/09 08:00")
                    expect(row).to_contain_text("Đã đủ giờ · thiếu bản đo")
                    expect(row.locator("td").nth(3)).to_have_text("—")
                    expect(row.locator("td").nth(4)).to_have_text("—")
                expect(page.locator("#acknowledge")).to_be_disabled()
                expect(page.locator("#prediction")).to_have_text("—")
                expect(page.locator("#mae")).to_have_text("—")
                state = page.request.get(url + "/api/state").json()
                has_model = scenario in ("missing_real_row", "missing_actual_row")
                status = "invalid_history" if has_model else "model_unavailable"
                assert state["current_forecast"]["status"] == status
                assert state["current_forecast"]["prediction"] is None
                if scenario == "missing_actual_row":
                    assert state["metrics"]["models"][0]["all_matured"]["rows"] == 0
                    assert all(row["actual"] is None for row in state["forecasts"])
                else:
                    assert not state["forecasts"]
                assert (state["model_info"] is not None) == has_model
                if scenario == "checksum_mismatch":
                    assert "checksum" in state["current_forecast"]["detail"].lower()
                page.locator("#operator").fill("QA kiểm tra lỗi · không phải người vận hành")
                for action in ("inspect", "dismiss"):
                    page.locator("#decisionNote").fill("Ghi nhận lỗi trong kiểm thử; không xác nhận tải an toàn.")
                    page.locator("#" + action).click()
                    expect(page.locator("#decisionFeedback")).to_contain_text("Đã lưu")
                saved = page.request.get(url + "/api/state").json()
                assert len(saved["decisions"]) == 2
                assert all(row["context"]["predicted_kWh"] is None for row in saved["decisions"])
                assert all(row["context"]["status"] == status for row in saved["decisions"])
                page.reload(wait_until="networkidle")
                expect(page.locator("#acknowledge")).to_be_disabled()
                assert len(page.request.get(url + "/api/state").json()["decisions"]) == 2
                assert not errors, errors
                page.screenshot(path=str(output / (scenario + ".png")), full_page=True)
                results.append({"scenario": scenario, "status": status, "passed": True,
                                "javascript_errors": errors})
                page.close()
        browser.close()
    (output / "failure_checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args().output_dir.resolve())
