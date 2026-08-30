#!/usr/bin/env python3
import socket
import ssl
import sys
import threading

CERT, KEY, HOST, PORT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
BOUND = [PORT]


def handle(conn):
    try:
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 16384:
            chunk = conn.recv(512)
            if not chunk:
                break
            data += chunk
        raw = data.decode("latin1", "replace")
        first = raw.split("\n", 1)[0]
        parts = first.split(" ")
        path = parts[1] if len(parts) > 1 else "/"
        req = raw.lower()
        tagged = "x-greppy-test: yes" in req
        has_ctx = "x-greppy-ctx: yes" in req
        sys.stderr.write(f"tls-origin {path} tagged={tagged} ctx={has_ctx}\n")
        sys.stderr.flush()
        if path.startswith("/jump"):
            loc = f"https://{HOST}:{BOUND[0]}/landed"
            conn.sendall(
                f"HTTP/1.1 302 Found\r\nLocation: {loc}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode()
            )
            return
        if path.startswith("/sub.js") or "/sub.js" in path:
            body = b"window.__greppySub='ok';" if tagged else b"window.__greppySub='missing';"
            conn.sendall(
                f"HTTP/1.1 200 OK\r\nContent-Type: application/javascript\r\nCache-Control: no-store\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            return
        marker = "HEADER_OK" if tagged else "HEADER_MISSING"
        ctx = "<span id=ctx>CTX_OK</span>" if has_ctx else ""
        html = f"<!DOCTYPE html><html><body>{marker}{ctx}<script src=\"/sub.js\"></script></body></html>".encode()
        conn.sendall(
            f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nContent-Length: {len(html)}\r\nConnection: close\r\n\r\n".encode()
            + html
        )
    finally:
        try:
            conn.shutdown(socket.SHUT_WR)
        except Exception:
            pass
        conn.close()


ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(CERT, KEY)
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))
sock.listen(16)
BOUND[0] = sock.getsockname()[1]
print(f"ready {HOST} {BOUND[0]}", flush=True)
while True:
    conn, addr = sock.accept()
    sys.stderr.write(f"tls-accept {addr}\n")
    sys.stderr.flush()
    try:
        conn = ctx.wrap_socket(conn, server_side=True)
    except Exception as error:
        sys.stderr.write(f"tls-wrap {error}\n")
        sys.stderr.flush()
        conn.close()
        continue
    threading.Thread(target=handle, args=(conn,), daemon=True).start()
