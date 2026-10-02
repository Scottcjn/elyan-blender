# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Local HTTP bridge.

Requests arrive on worker threads, but ``bpy`` may only be used from the main
thread. Every request is therefore queued and executed by ``pump()``, which a
timer calls in the UI and which ``serve()`` calls in a loop when headless.

Every request is a job with an id. A caller either waits for it on the same
connection, or sends it with ``submit`` and collects it later with ``result``.
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
# Finished jobs nobody has fetched yet; beyond this the oldest results are dropped.
MAX_KEPT_RESULTS = 50

_httpd = None
_thread = None
_token = ""
_jobs = queue.Queue()
_serving = False

# Every job by id until its result has been delivered, oldest first.
_registry = collections.OrderedDict()
_registry_lock = threading.Lock()

# Recent requests, newest last: (clock time, command, short summary, ok).
log = collections.deque(maxlen=20)


class _Job:
    __slots__ = ("id", "cmd", "args", "done", "response", "cancelled", "state", "detached", "created", "started",
                 "finished")

    def __init__(self, cmd, args, detached):
        # Random, so an id from an earlier session can never name a job of this one.
        self.id = secrets.token_hex(6)
        self.cmd = cmd
        self.args = args
        self.done = threading.Event()
        self.response = None
        self.cancelled = False
        self.state = "queued"
        # Nobody is waiting on the connection: the result is kept until fetched.
        self.detached = detached
        self.created = time.time()
        self.started = None
        self.finished = None

    def info(self):
        now = time.time()
        info = {
            "job": self.id,
            "cmd": self.cmd,
            "summary": commands.summarize(self.cmd, self.args),
            "state": self.state,
            "queued_seconds": round((self.started or self.finished or now) - self.created, 2),
        }
        if self.started is not None:
            info["run_seconds"] = round((self.finished or now) - self.started, 2)
        return info


def _enqueue(cmd, args, detached):
    job = _Job(cmd, args, detached)
    with _registry_lock:
        _registry[job.id] = job
    _jobs.put(job)
    return job


def _forget(job):
    with _registry_lock:
        _registry.pop(job.id, None)


def _trim_results():
    with _registry_lock:
        kept = [job for job in _registry.values() if job.state == "done" and job.detached]
        for job in kept[:max(0, len(kept) - MAX_KEPT_RESULTS)]:
            del _registry[job.id]


def _job_command(cmd, args):
    """``submit`` and friends. Runs on the request's own thread, so it answers while Blender is busy."""
    if cmd == "submit":
        inner, inner_args = args.get("cmd"), args.get("args") or {}
        if not isinstance(inner, str) or not isinstance(inner_args, dict):
            return {"ok": False, "error": "submit needs 'cmd' (a string) and optionally 'args' (an object)"}
        if inner in commands.JOB_COMMANDS:
            return {"ok": False, "error": "{:s} is answered directly, it cannot be submitted".format(inner)}
        if inner not in commands.COMMANDS:
            # Said now, rather than discovered when the result is fetched.
            return commands.dispatch(inner, inner_args)
        job = _enqueue(inner, inner_args, True)
        return {"ok": True, **job.info(), "queue_length": _jobs.qsize()}
    if cmd == "jobs":
        with _registry_lock:
            return {"ok": True, "jobs": [job.info() for job in _registry.values()]}

    with _registry_lock:
        job = _registry.get(args.get("job"))
    if job is None:
        return {"ok": False, "error": (
            "no job {!r}: ids are forgotten once the result is fetched, when the bridge stops, "
            "and when more than {:d} results are waiting"
        ).format(args.get("job"), MAX_KEPT_RESULTS)}
    if cmd == "status":
        return {"ok": True, **job.info()}
    if cmd == "cancel":
        # Only the main thread may touch Blender, and it cannot be interrupted once it has begun.
        with _registry_lock:
            if job.state != "queued":
                return {"ok": False, "error": "job {:s} is {:s}; only queued jobs can be cancelled".format(
                    job.id, job.state), **job.info()}
            job.cancelled = True
            job.state = "cancelled"
            _registry.pop(job.id, None)
        job.response = {"ok": False, "error": "cancelled before it ran"}
        job.done.set()
        return {"ok": True, **job.info()}
    # result
    try:
        wait = min(max(float(args.get("wait") or 0.0), 0.0), MAX_TIMEOUT)
    except (TypeError, ValueError):
        return {"ok": False, "error": "bad request: wait must be a number"}
    if not job.done.wait(wait):
        return {"ok": False, "pending": True, "error": "job {:s} is still {:s}".format(job.id, job.state),
                **job.info()}
    if not args.get("keep"):
        _forget(job)
    return {**job.response, "job": job.id, "state": job.state}


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
        if cmd in commands.JOB_COMMANDS:
            return self._reply(200, _job_command(cmd, args))
        job = _enqueue(cmd, args, False)
        if not job.done.wait(timeout):
            # The work is not cancelled: it becomes a job to poll, so nothing has to be sent twice.
            job.detached = True
            return self._reply(504, {
                "ok": False,
                "error": (
                    "timed out after {:g}s, but the request is still {:s} as job {:s}; "
                    "do not send it again: poll with status, fetch with result, or cancel it while queued"
                ).format(timeout, job.state, job.id),
                "timed_out": True,
                **job.info(),
            })
        _forget(job)
        self._reply(200, job.response)


def pump():
    """Run one queued request. Main thread only; one per call keeps the UI breathing between them."""
    while True:
        try:
            job = _jobs.get_nowait()
        except queue.Empty:
            return
        # Claimed under the lock, so a cancel either wins before this or is refused after.
        with _registry_lock:
            if not job.cancelled:
                job.state = "running"
                break
    job.started = time.time()
    try:
        job.response = commands.dispatch(job.cmd, job.args)
        log.append((time.strftime("%H:%M:%S"), job.cmd, commands.summarize(job.cmd, job.args), job.response["ok"]))
        _tag_redraw()
    except Exception as ex:
        # Never leave the caller waiting, and never let the timer die.
        job.response = job.response or {"ok": False, "error": "bridge error: {!r}".format(ex)}
    finally:
        job.finished = time.time()
        job.state = "done"
        job.done.set()
        _trim_results()


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
        job.state = "done"
        job.done.set()
    with _registry_lock:
        _registry.clear()
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
