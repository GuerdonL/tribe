"""Serve the visualizer: the static site in viz/site and the recorded data in viz/data.

    .venv/bin/python viz/serve.py                    # 0.0.0.0:8080, password from ~/.config/tribe-viz/auth
    .venv/bin/python viz/serve.py --host 127.0.0.1 --no-auth

Standard library only. Every request needs HTTP Basic auth (the results are unpublished); the file
~/.config/tribe-viz/auth holds one line, `user:password`, and is created with a random password on first
start. The connection is plain HTTP, so the password is not protected against someone on the network path.
"""
import argparse
import base64
import gzip
import json
import hmac
import os
import secrets
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = {"data": os.path.join(HERE, "data"), "": os.path.join(HERE, "site")}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon"}
AUTH_FILE = os.path.expanduser("~/.config/tribe-viz/auth")
AUTH = None          # "user:password", or None for no auth
LIVE = None          # live.Live poller, or None with --no-live


def load_auth():
    if not os.path.exists(AUTH_FILE):
        os.makedirs(os.path.dirname(AUTH_FILE), exist_ok=True)
        fd = os.open(AUTH_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"tribe:{secrets.token_urlsafe(12)}\n")
    return open(AUTH_FILE).read().strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "tribe-viz"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if "/live/state.json" not in (args[0] if args else ""):          # polled every few seconds
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _authorized(self):
        if AUTH is None:
            return True
        h = self.headers.get("Authorization", "")
        if not h.startswith("Basic "):
            return False
        try:
            given = base64.b64decode(h[6:]).decode()
        except Exception:
            return False
        return hmac.compare_digest(given.encode(), AUTH.encode())

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._authorized():
            return self._send(401, b"password required\n", extra=[("WWW-Authenticate", 'Basic realm="tribe", charset="UTF-8"')])
        path = unquote(self.path.split("?", 1)[0].split("#", 1)[0])
        if path == "/live/state.json":
            return self._live()
        if path == "/":
            path = "/index.html"
        parts = [p for p in path.split("/") if p]
        if any(p in (".", "..") or p.startswith(".") or "\\" in p or "\0" in p for p in parts):
            return self._send(404, b"not found\n")
        root = ROOTS["data"] if parts and parts[0] == "data" else ROOTS[""]
        rel = parts[1:] if parts and parts[0] == "data" else parts
        full = os.path.realpath(os.path.join(root, *rel))
        if not full.startswith(os.path.realpath(root) + os.sep) or not os.path.isfile(full):
            return self._send(404, b"not found\n")
        ext = os.path.splitext(full)[1]
        if ext not in TYPES:
            return self._send(404, b"not found\n")
        st = os.stat(full)
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        cache = [("ETag", etag), ("Cache-Control", "no-cache")]      # always revalidate: the data is rebuilt in place
        if self.headers.get("If-None-Match") == etag:
            return self._send(304, extra=cache)
        body = open(full, "rb").read()
        if "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024:
            body = gzip.compress(body, 5)
            cache.append(("Content-Encoding", "gzip"))
        cache.append(("Vary", "Accept-Encoding"))
        self._send(200, body, TYPES[ext], cache)


    def _live(self):
        """Live training logs: only the rows the page does not have yet (?have=run:rows,run:rows)."""
        if LIVE is None:
            body = json.dumps(dict(runs=[], error="live data is switched off", disabled=True)).encode()
        else:
            have = {}
            for part in parse_qs(urlsplit(self.path).query).get("have", [""])[0].split(","):
                name, _, n = part.rpartition(":")
                if name and n.isdigit():
                    have[name] = int(n)
            body = json.dumps(LIVE.state(have), separators=(",", ":")).encode()
        extra = [("Cache-Control", "no-store")]
        if "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024:
            body = gzip.compress(body, 5)
            extra.append(("Content-Encoding", "gzip"))
        self._send(200, body, "application/json", extra)


def main():
    global AUTH, LIVE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-auth", action="store_true", help="no password (only sensible with --host 127.0.0.1)")
    ap.add_argument("--no-live", action="store_true", help="do not read live logs from the running instances over SSH")
    a = ap.parse_args()
    if a.no_auth and a.host not in ("127.0.0.1", "localhost", "::1"):
        sys.exit("--no-auth is only allowed with --host 127.0.0.1")
    AUTH = None if a.no_auth else load_auth()
    if not a.no_live:
        sys.path.insert(0, HERE)
        import live
        LIVE = live.Live().start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"serving on http://{a.host}:{a.port}" + ("" if a.no_auth else f" (login in {AUTH_FILE})"), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
