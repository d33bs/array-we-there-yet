"""Loopback S3 proxy: measure actual HTTP requests and response payload bytes."""

from __future__ import annotations

import http.client
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import urlsplit

HOP_HEADERS = {"connection", "keep-alive", "proxy-connection", "transfer-encoding"}


class Meter:
    """Counters for one timed read, including Zarr metadata requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latency_ms = 0
        self.reset()

    def reset(self, *, latency_ms: int | None = None) -> None:
        with self._lock:
            if latency_ms is not None:
                self.latency_ms = latency_ms
            self.counts: Counter[str] = Counter()
            self.bytes_received = 0
            self.chunk_bytes = 0

    def record(self, method: str, path: str, size: int) -> None:
        parsed = urlsplit(path)
        is_chunk = "/features/c/" in parsed.path
        with self._lock:
            self.counts[method] += 1
            if method == "GET":
                self.bytes_received += size
                if is_chunk:
                    self.counts["CHUNK_GET"] += 1
                    self.chunk_bytes += size
                elif "list-type=" in parsed.query:
                    self.counts["LIST_GET"] += 1
                else:
                    self.counts["METADATA_GET"] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "get_requests": self.counts["GET"],
                "head_requests": self.counts["HEAD"],
                "chunk_gets": self.counts["CHUNK_GET"],
                "metadata_gets": self.counts["METADATA_GET"],
                "list_gets": self.counts["LIST_GET"],
                "response_bytes": self.bytes_received,
                "chunk_bytes": self.chunk_bytes,
            }


class Proxy(ThreadingHTTPServer):
    """A minimal HTTP/1.1 forwarding server bound to localhost only."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, meter: Meter, *, upstream_port: int = 9000) -> None:
        self.meter = meter
        self.upstream_port = upstream_port
        super().__init__(("127.0.0.1", 0), RequestHandler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def __enter__(self) -> Proxy:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
        self.server_close()
        self.thread.join(timeout=5)


class RequestHandler(BaseHTTPRequestHandler):
    """Forward requests without changing signed S3 headers or request path."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        self._forward()

    def do_HEAD(self) -> None:
        self._forward()

    def _forward(self) -> None:
        proxy = cast("Proxy", self.server)
        time.sleep(proxy.meter.latency_ms / 1000)
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_HEADERS
        }
        upstream = http.client.HTTPConnection(
            "127.0.0.1", proxy.upstream_port, timeout=60
        )
        try:
            upstream.request(self.command, self.path, headers=headers)
            response = upstream.getresponse()
            body = response.read()
            self.send_response_only(response.status, response.reason)
            for key, value in response.getheaders():
                name = key.lower()
                if name in HOP_HEADERS or (
                    name == "content-length" and self.command != "HEAD"
                ):
                    continue
                self.send_header(key, value)
            if self.command != "HEAD":
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            proxy.meter.record(self.command, self.path, len(body))
        finally:
            upstream.close()
