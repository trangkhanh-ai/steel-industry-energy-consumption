"""Serve the offline Steel replay UI on loopback only; no extra UI dependencies."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path

from src.steel.demo import ReplayDemo
from src.steel.inference import DEFAULT_MODEL_SUBDIR

ROOT = Path(__file__).resolve().parents[1]


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, kind="application/json; charset=utf-8"):
            payload = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def allowed(self):
            # Keep the local demo unavailable through foreign hosts/origins.
            host = self.headers.get("Host", "")
            expected = f"127.0.0.1:{self.server.server_port}"
            localhost = f"localhost:{self.server.server_port}"
            origin = self.headers.get("Origin")
            return host in (expected, localhost) and (origin is None or origin == f"http://{host}")

        def do_GET(self):
            if not self.allowed():
                return self.send(403, {"error": "Local origin required"})
            if self.path == "/api/state":
                return self.send(200, demo.dispatch("state"))
            files = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
            if self.path not in files:
                return self.send(404, {"error": "Not found"})
            name, kind = files[self.path]
            return self.send(200, (ROOT / "web/steel_demo" / name).read_bytes(), kind)

        def do_POST(self):
            if not self.allowed():
                return self.send(403, {"error": "Local origin required"})
            if self.path not in ("/api/reset", "/api/step", "/api/decision"):
                return self.send(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request size")
                args = json.loads(self.rfile.read(length))
                if not isinstance(args, dict):
                    raise ValueError("Expected a JSON object")
                response = demo.dispatch(self.path.rsplit("/", 1)[-1], args.get("start"), decision=args)
            except (ValueError, TypeError, KeyError) as error:
                return self.send(400, {"error": str(error)})
            except Exception:
                # Keep unexpected failures visible without treating them as predictions.
                import traceback
                traceback.print_exc()
                return self.send(500, {"error": "Không thực hiện được thao tác. Xem lỗi ở cửa sổ chạy server."})
            return self.send(200, response)
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-dir", type=Path, default=ROOT / DEFAULT_MODEL_SUBDIR)
    parser.add_argument("--sessions-dir", type=Path, default=ROOT / "reports/steel/modeling/demo_sessions")
    args = parser.parse_args()
    demo = ReplayDemo(ROOT, args.model_dir, args.sessions_dir)
    try:
        with HTTPServer(("127.0.0.1", args.port), make_handler(demo)) as server:
            print(f"Steel replay: http://127.0.0.1:{server.server_port} — Ctrl+C to stop", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        demo.close()
