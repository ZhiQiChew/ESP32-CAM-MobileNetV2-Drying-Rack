"""Same-LAN UDP discovery so camera clients do not need a fixed server IP."""

import os
import socket
import threading

DISCOVERY_PORT = 4210
DISCOVERY_REQUEST = b"DISCOVER_DRYING_RACK_SERVER"
DISCOVERY_PREFIX = "DRYING_RACK_SERVER"


def _serve():
    server_port = int(os.environ.get("PORT", "8080"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", DISCOVERY_PORT))
    except OSError as exc:
        print(f"[DISCOVERY] UDP port {DISCOVERY_PORT} unavailable: {exc}")
        sock.close()
        return
    print(f"[DISCOVERY] Listening on UDP {DISCOVERY_PORT}")
    while True:
        try:
            payload, client = sock.recvfrom(256)
            if payload.strip() == DISCOVERY_REQUEST:
                reply = f"{DISCOVERY_PREFIX}:{server_port}".encode("ascii")
                sock.sendto(reply, client)
        except OSError as exc:
            print(f"[DISCOVERY] Request failed: {exc}")


def start_discovery_service():
    thread = threading.Thread(target=_serve, daemon=True, name="udp-discovery")
    thread.start()
    return thread
