#!/usr/bin/env python3
"""Bounded local browser fixture and host-side oracle."""
import argparse, json, os, re, secrets, sys, tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import table_case
RUN_DIR = Path(os.environ["GREPPY_BASIC_FIXTURE_RUN_DIR"]) if os.environ.get("GREPPY_BASIC_FIXTURE_RUN_DIR") else Path(tempfile.gettempdir()) / "greppy-basic-fixture"
CASES = {"text", "checkbox", "address", "dialog", "table"}
RUN_RE = re.compile(r"^[0-9a-f]{12}$")

def facts(seed):
    n = sum(ord(c) for c in str(seed))
    return {"text": f"Invoice note {n % 97:02d}", "countries": {"Germany": ["Berlin", "Hamburg"], "France": ["Paris", "Lyon"]}, "postcodes": {"Germany": {"Berlin": "10115", "Hamburg": "20095"}, "France": {"Paris": "75001", "Lyon": "69001"}}}

def new_state(run_id, seed, case):
    if case == 'table': return table_case.new_state(run_id, seed)
    f = facts(seed)
    return {"run_id": run_id, "seed": seed, "case": case, "facts": f, "revision": 0, "values": {"note": f["text"], "note_saved": False, "enabled": False, "quantity": 1, "country": "", "city": "", "postcode": "", "postcode_valid": False, "saved": False, "save_origin": ""}, "events": []}

def state_path(run_id):
    if not RUN_RE.fullmatch(run_id): raise ValueError("run_id must be exactly 12 lowercase hexadecimal characters")
    return RUN_DIR / (run_id + ".json")

def write_state(s):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    p = state_path(s["run_id"]); tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, sort_keys=True), encoding="utf-8"); tmp.replace(p)

def load(run_id): return json.loads(state_path(run_id).read_text(encoding="utf-8"))

def boolean(value):
    if not isinstance(value, bool): raise ValueError("value must be boolean")
    return value

def mutate(s, action, payload):
    if not isinstance(payload, dict): raise ValueError("payload must be an object")
    case, v, f = s["case"], s["values"], s["facts"]
    value = payload.get("value")
    allowed = {"text": {"set_note"}, "checkbox": {"set_enabled", "set_quantity"}, "address": {"set_country", "set_city", "set_postcode"}, "dialog": {"save"}, "table": table_case.ACTIONS}[case]
    if action not in allowed: raise ValueError("action is not valid for this case")
    if case == 'table': table_case.apply(s, action, payload)
    elif action == "set_note":
        if not isinstance(value, str): raise ValueError("note must be text")
        v["note"], v["note_saved"] = value, True
    elif action == "set_enabled": v["enabled"] = boolean(value)
    elif action == "set_quantity":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1: raise ValueError("quantity must be a positive integer")
        if not v["enabled"]: raise ValueError("quantity is disabled until checkbox is enabled")
        v["quantity"] = value
    elif action == "set_country":
        if value not in f["countries"]: raise ValueError("unknown country")
        v.update(country=value, city="", postcode="", postcode_valid=False)
    elif action == "set_city":
        if value not in f["countries"].get(v["country"], []): raise ValueError("city does not belong to country")
        v.update(city=value, postcode="", postcode_valid=False)
    elif action == "set_postcode":
        if not isinstance(value, str): raise ValueError("postcode must be text")
        v["postcode"] = value; v["postcode_valid"] = f["postcodes"].get(v["country"], {}).get(v["city"]) == value
    elif action == "save":
        if payload.get("origin") not in {"task4-dialog", "outside"}: raise ValueError("unknown save origin")
        v["saved"], v["save_origin"] = True, payload["origin"]
    s["revision"] += 1; s["events"].append({"revision": s["revision"], "action": action, "payload": payload}); write_state(s)

def verify(s):
    v, f, case = s["values"], s["facts"], s["case"]
    if case == 'table': checks = table_case.checks(s)
    elif case == "text": checks = {"exact_note": v["note_saved"] and v["note"] == "Ready for review"}
    elif case == "checkbox": checks = {"enabled": v["enabled"], "quantity": v["quantity"] == 3}
    elif case == "address": checks = {"exact_location": v["country"] == "Germany" and v["city"] == "Berlin", "postcode": v["postcode"] == "10115" and v["postcode_valid"]}
    else: checks = {"scoped_save": v["saved"] and v["save_origin"] == "task4-dialog"}
    return {"run_id": s["run_id"], "case": case, "ok": all(checks.values()), "checks": checks, "revision": s["revision"], "event_count": len(s["events"])}

class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, content="application/json"):
        raw = body if isinstance(body, bytes) else (body.encode() if isinstance(body, str) else json.dumps(body).encode()); self.send_response(code); self.send_header("Content-Type", content); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        parsed = urlparse(self.path); q = parse_qs(parsed.query); rid = q.get("run_id", [""])[0]
        try:
            if parsed.path == "/api/state" and rid: return self.send(200, load(rid))
            if parsed.path.startswith("/static/"):
                rel = parsed.path.removeprefix("/static/")
                if "/" in rel or rel.startswith("."): return self.send(404, {"error": "not found"})
                p = (ROOT / "static" / rel).resolve()
                if ROOT.joinpath("static") not in p.parents or not p.is_file(): return self.send(404, {"error": "not found"})
                return self.send(200, p.read_bytes(), "text/css" if p.suffix == ".css" else "application/javascript")
            if parsed.path == "/": return self.send(200, (ROOT / "static/index.html").read_bytes(), "text/html")
            return self.send(404, {"error": "not found"})
        except (FileNotFoundError, ValueError): return self.send(404, {"error": "not found"})
    def do_POST(self):
        if self.path != "/api/action": return self.send(404, {"error": "not found"})
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))); s = load(data["run_id"]); mutate(s, data["action"], data.get("payload", {})); self.send(200, s)
        except (KeyError, TypeError, ValueError, FileNotFoundError, json.JSONDecodeError) as e: self.send(400, {"error": str(e)})
    def log_message(self, *_): pass

def set_dir(path):
    global RUN_DIR
    RUN_DIR = Path(path)

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-run"); c.add_argument("--case", choices=sorted(CASES), required=True); c.add_argument("--seed", default="basic-1"); c.add_argument("--run-dir", default=None)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8765); s.add_argument("--run-dir", default=None)
    v = sub.add_parser("verify-run"); v.add_argument("run_id"); v.add_argument("--run-dir", default=None)
    a = ap.parse_args(); set_dir(a.run_dir or RUN_DIR)
    if a.cmd == "create-run": rid = secrets.token_hex(6); write_state(new_state(rid, a.seed, a.case)); print(rid)
    elif a.cmd == "verify-run":
        result = verify(load(a.run_id)); print(json.dumps(result, sort_keys=True)); raise SystemExit(0 if result["ok"] else 1)
    else:
        server = HTTPServer(("127.0.0.1", a.port), Handler); print(f"http://127.0.0.1:{server.server_port}/", flush=True); server.serve_forever()
if __name__ == "__main__": main()
