"""Local results dashboard for pooria-bench runs.

Standalone on purpose: it depends only on the Python standard library and on the
run-directory file format (manifest.json, samples.jsonl, results.jsonl,
transcripts.jsonl), never on the pooria_bench package. The UI can change freely
without touching the harness.

Usage:
    uv run python dashboard/serve.py            # http://127.0.0.1:8765
    uv run python dashboard/serve.py --port 9000 --runs path/to/runs
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
DEFAULT_RUNS = HERE.parent / "runs"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line still being written by a live run
    return rows


def latest_results(path: Path) -> list[dict]:
    """One record per (model, sample, repeat); later lines win (resumes, rescoring)."""
    latest: dict[tuple, dict] = {}
    for r in read_jsonl(path):
        latest[(r.get("model"), r.get("sample_id"), r.get("repeat", 0))] = r
    return list(latest.values())


class Runs:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, run_id: str) -> Path | None:
        path = (self.root / run_id).resolve()
        if not path.is_relative_to(self.root) or not (path / "results.jsonl").exists():
            return None
        return path

    def list(self) -> list[dict]:
        out = []
        for results in sorted(self.root.glob("**/results.jsonl")):
            run_dir = results.parent
            try:
                manifest = json.loads((run_dir / "manifest.json").read_text())
            except (OSError, json.JSONDecodeError):
                manifest = {}
            rows = latest_results(results)
            models: list[str] = []
            for inv in [manifest, *manifest.get("resumed", [])]:
                models += [m for m in (inv.get("models") or {}) if m not in models]
            out.append(
                {
                    "id": run_dir.relative_to(self.root).as_posix(),
                    "task": manifest.get("task"),
                    "models": models,
                    "n_samples": manifest.get("n_samples"),
                    "n_results": len(rows),
                    "created_at": manifest.get("created_at"),
                    "mtime": results.stat().st_mtime,
                }
            )
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    def load(self, run_dir: Path) -> dict:
        try:
            manifest = json.loads((run_dir / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            manifest = {}
        return {
            "manifest": manifest,
            "samples": read_jsonl(run_dir / "samples.jsonl"),
            "results": latest_results(run_dir / "results.jsonl"),
        }

    def transcript(self, run_dir: Path, model: str, sample: str, repeat: int) -> dict | None:
        match = None
        for t in read_jsonl(run_dir / "transcripts.jsonl"):
            if t.get("model") == model and t.get("sample_id") == sample and t.get("repeat", 0) == repeat:
                match = t  # keep the last one (a resumed sample overwrites earlier attempts)
        return match


def make_handler(runs: Runs):
    class Handler(BaseHTTPRequestHandler):
        server_version = "pooria-bench-dashboard"

        def log_message(self, fmt, *args):  # quieter console
            if "/api/" in (args[0] if args else ""):
                return
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _static(self, rel: str):
            path = (STATIC / (rel or "index.html")).resolve()
            if not path.is_relative_to(STATIC) or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/api/runs":
                return self._json(runs.list())
            if url.path in ("/api/run", "/api/transcript"):
                run_dir = runs.resolve(q.get("id", ""))
                if run_dir is None:
                    return self._json({"error": "unknown run"}, HTTPStatus.NOT_FOUND)
                if url.path == "/api/run":
                    return self._json(runs.load(run_dir))
                t = runs.transcript(run_dir, q.get("model", ""), q.get("sample", ""), int(q.get("repeat", 0)))
                if t is None:
                    return self._json({"error": "no transcript"}, HTTPStatus.NOT_FOUND)
                return self._json(t)
            return self._static(url.path.lstrip("/"))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="directory holding run folders")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    runs = Runs(args.runs)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runs))
    print(f"Dashboard: http://{args.host}:{args.port}  (runs from {runs.root})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
