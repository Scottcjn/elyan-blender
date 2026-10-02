# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Window self-test: the artist clicks once, the bridge proves itself in her Blender.

The bridge was developed against headless sessions. The parts that only exist
with a window (the timer that answers requests, undo steps, viewport capture,
window screenshots) can only be proven in one, on her machine. This runs them
in order, records pass or fail for each with the reason, and writes the result
where ``state`` reports it, so an assistant knows what it may rely on.

The steps run from a timer, never in one blocking call: the test sends real
HTTP requests to the bridge from a thread, and the main thread must stay free
to answer them. Blocking it on its own request would hang Blender.

Each step is a function taking the shared context dict. A step that has to
wait is a generator: it yields while waiting and is resumed on the next tick.
Raising fails the step with the message; returning passes it.
"""

import inspect
import json
import os
import threading
import time
import urllib.error
import urllib.request

import bpy

from . import commands, server, session

TICK = 0.1

# Created by the test through the bridge, and removed again by one undo.
PROBE = "ElyanBridgeTestProbe"
MARKER = "elyan_bridge_test_marker"

_PROBE_CODE = (
    "{marker} = True\n"
    "bpy.context.scene.collection.objects.link(bpy.data.objects.new({probe!r}, None))\n"
).format(marker=MARKER, probe=PROBE)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_runner = None
_last = None
_last_loaded = False


class Skipped(RuntimeError):
    """A step that could not be tried because an earlier one failed."""


# -----------------------------------------------------------------------------
# Runner. No Blender in here, so it can be tested with made-up steps.

class Runner:
    """
    Runs steps one after another, a little on every ``tick()``.

    ``steps`` is a list of ``(name, label, function, timeout seconds)``. Every
    step runs whatever happened to the ones before it, so one failure does not
    hide the others and the last step can always tidy up.
    """

    def __init__(self, steps, context=None, clock=time.monotonic):
        self.steps = list(steps)
        self.context = {} if context is None else context
        self.results = []
        self._clock = clock
        self._index = 0
        self._waiting = None
        self._started = None

    def current(self):
        """Label of the step being run, or None when finished."""
        return self.steps[self._index][1] if self._index < len(self.steps) else None

    def _record(self, ok, error=""):
        name, label, _function, _timeout = self.steps[self._index]
        self.results.append({
            "name": name,
            "label": label,
            "ok": ok,
            "error": error,
            "seconds": round(self._clock() - self._started, 2),
        })
        self._waiting = None
        self._started = None
        self._index += 1

    def tick(self):
        """Advance by one step, or one wait of a waiting step. Returns True once every step has a result."""
        if self._index >= len(self.steps):
            return True
        _name, _label, function, timeout = self.steps[self._index]
        try:
            if self._waiting is None:
                self._started = self._clock()
                outcome = function(self.context)
                if not inspect.isgenerator(outcome):
                    self._record(True)
                    return self._index >= len(self.steps)
                self._waiting = outcome
            if self._clock() - self._started > timeout:
                self._waiting.close()
                self._record(False, "no answer after {:g} seconds".format(timeout))
            else:
                next(self._waiting)
        except StopIteration:
            self._record(True)
        except Exception as ex:
            # Not only RuntimeError: a step failing in an unforeseen way is still a result.
            prefix = "skipped: " if isinstance(ex, Skipped) else ""
            self._record(False, prefix + (str(ex) or type(ex).__name__))
        return self._index >= len(self.steps)

    def passed(self):
        return len(self.results) == len(self.steps) and all(step["ok"] for step in self.results)


def make_result(runner):
    """What is written to disk and shown by ``state``."""
    return {
        "blender": bpy.app.version_string,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window": not bpy.app.background,
        "passed": runner.passed(),
        "steps": list(runner.results),
    }


def write_result(result, path=None):
    """Write a result where ``session.read_selftest()`` finds it (or to ``path``, for tests)."""
    path = path or session.selftest_path()
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return path


# -----------------------------------------------------------------------------
# Talking to the bridge the way a client does

class _Request(threading.Thread):
    """One HTTP request to the bridge, off the main thread so the timer can answer it."""

    def __init__(self, info, cmd, args):
        super().__init__(name="elyan_llm_selftest", daemon=True)
        self.info = info
        self.body = json.dumps({"cmd": cmd, "args": args}).encode("utf-8")
        self.response = None
        self.error = ""

    def run(self):
        request = urllib.request.Request(
            "http://127.0.0.1:{:d}/rpc".format(self.info["port"]),
            data=self.body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.info["token"]},
        )
        try:
            with urllib.request.urlopen(request, timeout=120.0) as reply:
                self.response = json.loads(reply.read())
        except urllib.error.HTTPError as ex:
            try:
                self.response = json.loads(ex.read())
            except ValueError:
                self.error = "HTTP {:d}".format(ex.code)
        except (OSError, ValueError) as ex:
            self.error = "could not reach the bridge: {!s}".format(ex)


def _ask(context, cmd, args):
    """Send a request and wait for its answer without blocking. Use with ``yield from``."""
    info = context.get("info")
    if info is None:
        raise Skipped("the bridge did not start")
    request = _Request(info, cmd, args)
    request.start()
    while request.is_alive():
        yield
    if request.error:
        raise RuntimeError(request.error)
    response = request.response
    if not response.get("ok"):
        # The last line of a traceback is the part a person can read.
        lines = str(response.get("error") or "the request failed").strip().splitlines()
        raise RuntimeError(lines[-1])
    return response


def _check_png(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(len(_PNG_MAGIC))
        size = os.path.getsize(path)
    except OSError:
        raise RuntimeError("no picture was written") from None
    if size == 0 or head != _PNG_MAGIC:
        raise RuntimeError("the picture file is empty or not a PNG")


def _scratch(context, name):
    folder = os.path.join(session.session_dir(), "selftest")
    os.makedirs(folder, mode=0o700, exist_ok=True)
    path = os.path.join(folder, name)
    context.setdefault("files", []).append(path)
    return path


# -----------------------------------------------------------------------------
# Steps

def step_start(context):
    context["was_running"] = server.is_running()
    port = server.start(context.get("port", 0))
    # Read what a client would read, so a session file that is missing or wrong fails here.
    with open(session.session_path(os.getpid()), encoding="utf-8") as fh:
        info = json.load(fh)
    if info.get("port") != port or not info.get("token"):
        raise RuntimeError("the session file does not describe this bridge")
    # Whatever she did since her last undo step is fixed in a step of its own,
    # so the undo below takes back the test's object and nothing of hers.
    commands._undo_push("Before LLM bridge test")
    context["info"] = info


def step_request(context):
    response = yield from _ask(context, "ping", {})
    if response.get("pid") != os.getpid():
        raise RuntimeError("another Blender answered")


def step_undo(context):
    response = yield from _ask(context, "exec", {"code": _PROBE_CODE, "label": "bridge test"})
    context["generation"] = response.get("generation")
    if PROBE not in bpy.data.objects:
        raise RuntimeError("the request reported success but made nothing")
    if not response.get("undo_pushed"):
        raise RuntimeError("the request did not record an undo step")
    # Undo on a later tick than the request, as her Ctrl+Z would be.
    yield
    override = commands.view3d_context()
    if override:
        with bpy.context.temp_override(**override):
            outcome = bpy.ops.ed.undo()
    else:
        outcome = bpy.ops.ed.undo()
    if 'FINISHED' not in outcome:
        raise RuntimeError("Blender refused to undo")
    yield
    if PROBE in bpy.data.objects:
        raise RuntimeError("one undo did not remove what the request added")


def step_viewport(context):
    path = _scratch(context, "viewport.png")
    yield from _ask(context, "render", {"path": path, "mode": "viewport", "resolution": [320, 240]})
    _check_png(path)


def step_screenshot(context):
    path = _scratch(context, "window.png")
    yield from _ask(context, "screenshot", {"path": path})
    _check_png(path)


def step_namespace(context):
    before = context.get("generation")
    if before is None:
        raise Skipped("the undo step did not get that far")
    response = yield from _ask(context, "exec", {"code": "{!r} in globals()".format(MARKER), "label": "bridge test"})
    if response.get("result"):
        raise RuntimeError("names from before the undo were kept")
    if not response.get("generation", 0) > before:
        raise RuntimeError("the bridge did not notice the undo")


def step_tidy(context):
    problems = []
    probe = bpy.data.objects.get(PROBE)
    if probe is not None:
        # Undo failed above and said so; the object must still not stay in her scene.
        bpy.data.objects.remove(probe, do_unlink=True)
    for path in context.get("files", ()):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as ex:
            problems.append(str(ex))
    try:
        os.rmdir(os.path.join(session.session_dir(), "selftest"))
    except OSError:
        pass
    if not context.get("was_running", True):
        server.stop()
    if problems:
        raise RuntimeError("; ".join(problems))


# name, what the panel shows, function, seconds allowed.
STEPS = (
    ("start", "Bridge starts", step_start, 10.0),
    ("request", "Answers a request", step_request, 15.0),
    ("undo", "Undo takes back a change", step_undo, 30.0),
    ("viewport", "Picture of the viewport", step_viewport, 90.0),
    ("screenshot", "Picture of the window", step_screenshot, 60.0),
    ("namespace", "Starts fresh after undo", step_namespace, 30.0),
    ("tidy", "Scene left as it was", step_tidy, 10.0),
)


# -----------------------------------------------------------------------------
# Running it in a window

def _tag_redraw():
    wm = bpy.context.window_manager
    for window in (wm.windows if wm else ()):
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _tick():
    global _runner, _last, _last_loaded
    if _runner is None:
        return None
    finished = _runner.tick()
    _tag_redraw()
    if not finished:
        return TICK
    result = make_result(_runner)
    _runner = None
    _last, _last_loaded = result, True
    try:
        write_result(result)
    except OSError as ex:
        # Still shown in the panel; only ``state`` will not know.
        print("elyan_llm: could not save the self-test result:", ex)
    return None


def start(port=0):
    """Begin the test. Returns at once; the steps run from a timer."""
    global _runner
    if bpy.app.background:
        raise RuntimeError("This test needs a Blender window; it cannot run in a background session")
    if _runner is not None:
        raise RuntimeError("The test is already running")
    _runner = Runner(STEPS, {"port": port})
    # Persistent, so that nothing the steps do (undo, or her loading a file) can
    # remove the timer and leave the test waiting for ever.
    bpy.app.timers.register(_tick, first_interval=TICK, persistent=True)


def cancel():
    """Drop a running test, e.g. when the add-on is switched off."""
    global _runner
    _runner = None
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)


def running():
    """``(label of the step being run, results so far)``, or None when no test is running."""
    if _runner is None:
        return None
    return _runner.current(), list(_runner.results)


def last():
    """The latest result: this session's, else the one on disk, else None."""
    global _last, _last_loaded
    if not _last_loaded:
        # Read once, not on every redraw of the panel.
        _last, _last_loaded = session.read_selftest(), True
    return _last
