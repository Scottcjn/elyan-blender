#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Command-line client for the Elyan LLM bridge. Standard library only.

   client.py launch [file.blend]        start a headless Blender that serves
   client.py sessions                   list running bridges
   client.py ping | scene | object [NAME]
   client.py exec -c "CODE" | -f FILE | <stdin
   client.py render OUT.png [--mode viewport] [--res 900 1200] [--engine ...]
   client.py screenshot OUT.png
   client.py api QUERY
   client.py backup [--label TEXT]
   client.py quit
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session  # noqa: E402

ADDONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_session(pid=None):
    live = session.list_live()
    if pid is not None:
        live = [s for s in live if s["pid"] == pid]
    if not live:
        sys.exit("no running bridge found; start one in Blender (Sidebar > Elyan) or run: client.py launch")
    if len(live) > 1:
        lines = ["  --pid {pid}  port {port}  {blend}".format(**s) for s in live]
        sys.exit("several bridges are running, choose one:\n" + "\n".join(lines))
    return live[0]


def call(info, cmd, args=None, timeout=660.0):
    body = json.dumps({"cmd": cmd, "args": args or {}}).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:{:d}/rpc".format(info["port"]),
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + info["token"]},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.loads(reply.read())
    except urllib.error.HTTPError as ex:
        try:
            return json.loads(ex.read())
        except ValueError:
            return {"ok": False, "error": "HTTP {:d}".format(ex.code)}
    except (urllib.error.URLError, OSError) as ex:
        return {"ok": False, "error": "cannot reach Blender on port {:d}: {!s}".format(info["port"], ex)}


def launch(blender, blend, wait):
    """Start a headless Blender running ``serve()`` and wait for its session file."""
    # ``sys.path`` insert keeps this working with a stock Blender that lacks the add-on.
    expr = "import sys; sys.path.insert(0, {!r}); import elyan_llm; elyan_llm.serve()".format(ADDONS_DIR)
    command = [blender, "--background"]
    if blend:
        command.append(blend)
    command += ["--python-expr", expr]
    log_path = os.path.join(session.session_dir(), "launch.log")
    os.makedirs(session.session_dir(), mode=0o700, exist_ok=True)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit("Blender exited with code {:d}, see {:s}".format(proc.returncode, log_path))
        if os.path.exists(session.session_path(proc.pid)):
            return proc.pid
        time.sleep(0.2)
    sys.exit("Blender did not start serving within {:g}s, see {:s}".format(wait, log_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid", type=int, help="which Blender, when several bridges are running")
    parser.add_argument("--json", action="store_true", help="print the raw JSON response")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("launch")
    p.add_argument("blend", nargs="?")
    p.add_argument("--blender", default=os.environ.get("BLENDER_BIN", "blender"))
    p.add_argument("--wait", type=float, default=60.0)
    sub.add_parser("sessions")
    sub.add_parser("ping")
    p = sub.add_parser("scene")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("object")
    p.add_argument("name", nargs="?")
    p = sub.add_parser("exec")
    p.add_argument("-c", dest="code")
    p.add_argument("-f", dest="file")
    p.add_argument("--label", help="name shown in Blender's undo history and panel")
    p.add_argument("--reset", action="store_true", help="clear names kept from earlier calls")
    p.add_argument("--timeout", type=float)
    p = sub.add_parser("render")
    p.add_argument("path")
    p.add_argument("--mode", choices=("camera", "viewport"), default="camera")
    p.add_argument("--res", type=int, nargs=2, metavar=("X", "Y"))
    p.add_argument("--engine")
    p.add_argument("--camera")
    p = sub.add_parser("screenshot")
    p.add_argument("path")
    p = sub.add_parser("api")
    p.add_argument("query")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("backup")
    p.add_argument("--label")
    sub.add_parser("quit")
    opts = parser.parse_args()

    if opts.cmd == "sessions":
        for s in session.list_live():
            print("pid {pid}  port {port}  blender {blender}  {mode}  {blend}".format(
                mode="headless" if s["background"] else "ui", **s))
        return 0
    if opts.cmd == "launch":
        pid = launch(opts.blender, opts.blend, opts.wait)
        print("serving, pid {:d}".format(pid))
        return 0

    args = {}
    if opts.cmd == "exec":
        if opts.code is not None:
            args["code"] = opts.code
        elif opts.file:
            with open(opts.file, encoding="utf-8") as fh:
                args["code"] = fh.read()
            args.setdefault("label", os.path.basename(opts.file))
        else:
            args["code"] = sys.stdin.read()
        for key in ("label", "reset", "timeout"):
            if getattr(opts, key):
                args[key] = getattr(opts, key)
    elif opts.cmd == "render":
        # Resolve here: the caller's working directory is not Blender's.
        args = {"path": os.path.abspath(opts.path), "mode": opts.mode}
        if opts.res:
            args["resolution"] = opts.res
        for key in ("engine", "camera"):
            if getattr(opts, key):
                args[key] = getattr(opts, key)
    elif opts.cmd == "screenshot":
        args = {"path": os.path.abspath(opts.path)}
    elif opts.cmd == "object":
        if opts.name:
            args["name"] = opts.name
    else:
        for key in ("limit", "query", "label"):
            if getattr(opts, key, None):
                args[key] = getattr(opts, key)

    timeout = (args.get("timeout") or 600.0) + 60.0
    response = call(pick_session(opts.pid), opts.cmd, args, timeout)

    if opts.json:
        print(json.dumps(response, indent=1))
    elif opts.cmd == "exec":
        if response.get("stdout"):
            sys.stdout.write(response["stdout"])
            if not response["stdout"].endswith("\n"):
                sys.stdout.write("\n")
        if response.get("ok") and response.get("result") is not None:
            result = response["result"]
            print(result if isinstance(result, str) else json.dumps(result, indent=1))
    elif response.get("ok"):
        print(json.dumps({k: v for k, v in response.items() if k != "ok"}, indent=1))
    if not response.get("ok"):
        sys.stderr.write(response.get("error", "failed") + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
