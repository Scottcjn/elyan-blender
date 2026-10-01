# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Local HTTP bridge.

Requests arrive on worker threads, but ``bpy`` may only be used from the main
thread. Every request is therefore queued and executed by ``pump()``, which a
timer calls in the UI and which ``serve()`` calls in a loop when headless.
"""

import collections
import hmac
import json
import os
import queue
import secrets
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bpy

from . import commands, session

MAX_BODY = 32 * 1024 * 1024
DEFAULT_TIMEOUT = 600.0
MAX_TIMEOUT = 6 * 3600.0
PUMP_INTERVAL = 0.05

_httpd = None
_thread = None
_token = ""
_jobs = queue.Queue()
_serving = False

# Recent requests, newest last: (clock time, command, short summary, ok).
log = collections.deque(maxlen=20)


class _Job:
    __slots__ = ("cmd", "args", "done", "response", "cancelled")

    def __init__(self, cmd, args):
        self.cmd = cmd
        self.args = args
        self.done = threading.Event()
        self.response = None
        self.cancelled = False


class _Handler(BaseHTTPRequestHandler):
    server_version = "ElyanBlenderLLM/1"

    def log_message(self, format, *args):
        pass

    def _reply(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        # Browsers always send Origin on cross-site POST; no real client does.
        if self.headers.get("Origin"):
            return self._reply(403, {"ok": False, "error": "browser requests are refused"})
        sent = self.headers.get("Authorization", "")
        if not hmac.compare_digest(sent.encode(), ("Bearer " + _token).encode()):
            return self._reply(401, {"ok": False, "error": "bad or missing token"})
        if self.path != "/rpc":
            return self._reply(404, {"ok": False, "error": "unknown path"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:
            return self._reply(413, {"ok": False, "error": "bad request size"})
        try:
            message = json.loads(self.rfile.read(length))
            cmd = message["cmd"]
            args = message.get("args") or {}
            if not isinstance(cmd, str) or not isinstance(args, dict):
                raise TypeError("cmd must be a string and args an object")
        except (ValueError, KeyError, TypeError) as ex:
            return self._reply(400, {"ok": False, "error": "bad request: {!s}".format(ex)})

        try:
            timeout = min(max(float(args.get("timeout") or DEFAULT_TIMEOUT), 1.0), MAX_TIMEOUT)
        except (TypeError, ValueError):
            return self._reply(400, {"ok": False, "error": "bad request: timeout must be a number"})
        job = _Job(cmd, args)
        _jobs.put(job)
        if not job.done.wait(timeout):
            job.cancelled = True
            return self._reply(504, {
                "ok": False,
                "error": (
                    "timed out after {:g}s waiting for Blender's main thread; if the request had "
                    "already started it will still finish, check before sending it again"
                ).format(timeout),
            })
        self._reply(200, job.response)


def pump():
    """Run one queued request. Main thread only; one per call keeps the UI breathing between them."""
    while True:
        try:
            job = _jobs.get_nowait()
        except queue.Empty:
            return
        if not job.cancelled:
            break
    try:
        job.response = commands.dispatch(job.cmd, job.args)
        log.append((time.strftime("%H:%M:%S"), job.cmd, commands.summarize(job.cmd, job.args), job.response["ok"]))
        _tag_redraw()
    except Exception as ex:
        # Never leave the caller waiting, and never let the timer die.
        job.response = job.response or {"ok": False, "error": "bridge error: {!r}".format(ex)}
    finally:
        job.done.set()


def _pump_timer():
    if _httpd is None:
        return None
    pump()
    return PUMP_INTERVAL


def _tag_redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def is_running():
    return _httpd is not None


def port():
    return _httpd.server_address[1] if _httpd else 0


def start(port=0):
    """Start listening on localhost. ``port`` 0 lets the system pick a free one."""
    global _httpd, _thread, _token
    if _httpd is not None:
        return _httpd.server_address[1]
    _token = secrets.token_urlsafe(32)
    _httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    _httpd.daemon_threads = True
    _thread = threading.Thread(target=_httpd.serve_forever, name="elyan_llm", daemon=True)
    _thread.start()
    write_session()
    if not bpy.app.background:
        bpy.app.timers.register(_pump_timer, persistent=True)
    return _httpd.server_address[1]


def write_session():
    if _httpd is None:
        return
    session.write({
        "pid": os.getpid(),
        "port": _httpd.server_address[1],
        "token": _token,
        "blend": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "background": bpy.app.background,
    })


def stop():
    global _httpd, _thread, _serving
    _serving = False
    if _httpd is None:
        return
    httpd, _httpd = _httpd, None
    httpd.shutdown()
    httpd.server_close()
    # Requests still waiting must not run against whatever session starts next.
    while True:
        try:
            job = _jobs.get_nowait()
        except queue.Empty:
            break
        job.response = {"ok": False, "error": "the bridge was stopped before this request ran"}
        job.done.set()
    _thread = None
    session.remove(os.getpid())
    if bpy.app.timers.is_registered(_pump_timer):
        bpy.app.timers.unregister(_pump_timer)


def serve(port=0):
    """
    Block and answer requests until a ``quit`` command arrives.

    For headless use, where no event loop runs timers::

       blender -b file.blend --python-expr "import elyan_llm; elyan_llm.serve()"
    """
    global _serving
    start(port)
    _serving = True
    commands.quit_callback = _request_quit
    try:
        while _serving:
            pump()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        commands.quit_callback = None
        stop()


def _request_quit():
    global _serving
    _serving = False
