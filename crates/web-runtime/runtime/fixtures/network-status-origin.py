#!/usr/bin/env python3
import socket
import ssl
import sys
import threading

CERT, KEY, HOST, PORT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
BOUND = [PORT]
REPEATED = [0]
LOCK = threading.Lock()


def send(conn, status, body=b"", extra=b""):
    conn.sendall(
        f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\nConnection: close\r\n".encode()
        + extra
        + b"\r\n"
        + body
    )


def handle(conn):
    try:
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 16384:
            chunk = conn.recv(512)
            if not chunk:
                break
            data += chunk
        first = data.decode("latin1", "replace").split("\n", 1)[0]
        parts = first.split(" ")
        path = parts[1] if len(parts) > 1 else "/"
        if path == "/jump":
            location = f"https://{HOST}:{BOUND[0]}/landed"
            send(conn, "302 Found", extra=f"Location: {location}\r\n".encode())
        elif path == "/chunked":
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
            )
        elif path == "/missing":
            send(conn, "404 Not Found", b"missing")
        elif path == "/repeat":
            with LOCK:
                occurrence = REPEATED[0]
                REPEATED[0] += 1
            if occurrence == 0:
                send(conn, "200 OK", b"repeated ok")
            else:
                send(conn, "404 Not Found", b"repeated missing")
        elif path == "/empty":
            send(conn, "204 No Content")
        elif path == "/landed":
            send(conn, "200 OK", b"landed")
        else:
            send(conn, "200 OK", b"ok")
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
    conn, _ = sock.accept()
    try:
        conn = ctx.wrap_socket(conn, server_side=True)
    except Exception:
        conn.close()
        continue
    threading.Thread(target=handle, args=(conn,), daemon=True).start()
