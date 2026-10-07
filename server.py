import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BUILDINGS_FILE = Path(__file__).with_name("buildings.json")
PARKING_FILE = Path(__file__).with_name("parking.json")

class StoreHandler(BaseHTTPRequestHandler):

    def _add_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._add_cors_headers()
        self.end_headers()

    def do_GET(self):

        path = urlparse(self.path).path

        # BUILDINGS endpoint
        if path == "/buildings":
            with BUILDINGS_FILE.open(encoding="utf-8") as file:
                data = json.load(file)

        # PARKING endpoint
        elif path == "/parking":
            with PARKING_FILE.open(encoding="utf-8") as file:
                data = json.load(file)

        # Endpoint does not exist
        else:
            self.send_error(404, "Endpoint not found")
            return

        body = json.dumps(data).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )
        self.send_header("Content-Length", str(len(body)))
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", "8000"))

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        StoreHandler
    )

    print(f"Server listening on port {port}", flush=True)

    server.serve_forever()