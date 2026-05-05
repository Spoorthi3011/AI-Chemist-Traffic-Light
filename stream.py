import socketserver
import http.server
import threading
import time

class StreamServer:
    def __init__(self, port=8765):
        self._frame = None
        self._lock = threading.Lock()
        self._port = port
        self._server = None
        self._thread = None

    def update_frame(self, jpeg_bytes):
        with self._lock:
            self._frame = jpeg_bytes

    def get_frame(self):
        with self._lock:
            return self._frame

    def start(self):
        stream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"""
                        <html>
                        <head>
                          <title>AI-CHEMIST Live Feed</title>
                          <style>
                            body { background:#111; margin:0; display:flex;
                                   flex-direction:column; align-items:center; }
                            h2   { color:#0f0; font-family:monospace; margin:10px; }
                            img  { width:100%; max-width:900px; border:2px solid #0f0; }
                            p    { color:#888; font-family:monospace; font-size:12px; }
                          </style>
                        </head>
                        <body>
                          <h2>AI-CHEMIST LIVE FEED | GREEN -> RED -> YELLOW -> GREEN</h2>
                          <img src='/stream'>
                          <p>MJPEG stream | YOLO + Color Detection</p>
                        </body>
                        </html>""")

                elif self.path == '/stream':
                    # ── MJPEG multipart push ──────────────────────────────
                    self.send_response(200)
                    self.send_header(
                        'Content-type',
                        'multipart/x-mixed-replace; boundary=--jpgboundary'
                    )
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'keep-alive')
                    self.end_headers()
                    try:
                        while True:
                            frame = stream.get_frame()
                            if frame:
                                self.wfile.write(b'--jpgboundary\r\n')
                                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                                self.wfile.write(
                                    f'Content-Length: {len(frame)}\r\n\r\n'.encode()
                                )
                                self.wfile.write(frame)
                                self.wfile.write(b'\r\n')
                                self.wfile.flush()
                            time.sleep(0.04)   # ~25 fps cap
                    except (BrokenPipeError, ConnectionResetError):
                        pass   # client disconnected — normal

        # ── ThreadingTCPServer so clients don't block each other ──────
        socketserver.TCPServer.allow_reuse_address = True
        self._server = socketserver.ThreadingTCPServer(('0.0.0.0', self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        print(f"[Stream] Live preview at http://{HOST}:{self._port}")
        print(f"[Stream] Open that URL in any browser on your network!")

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            print("[Stream] Closed cleanly")
