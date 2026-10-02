#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Command-line client for the Elyan LLM bridge. Standard library only.

   client.py launch [file.blend]        start a headless Blender that serves
   client.py sessions                   list running bridges
   client.py ping | scene | object [NAME]
   client.py exec -c "CODE" | -f FILE | <stdin   [--diff]
   client.py rebuild SCRIPT.py [--root DIR] [--diff] [ARGV...]
   client.py checkpoint NAME COLLECTION [--replace] | rollback NAME | checkpoints [--delete NAME]
   client.py check [NAMES...] [--max-tris N] [--max-slots N] [--seam JSON] [--clearance JSON]
   client.py snapshot NAME | diff NAME [--to NAME]
   client.py contact_sheet OUT.png NAMES... [--closeup-bone RIG BONE | --closeup-object NAME]
   client.py submit CMD [--args JSON] | status JOB | result JOB [--wait S] | cancel JOB | jobs
   client.py --detach <any command>     same as submit: prints a job id at once
   client.py render OUT.png [--mode viewport] [--res 900 1200] [--engine ...]
   client.py screenshot OUT.png
   client.py api QUERY
   client.py backup [--label TEXT]
   client.py validate [NAMES...] [--profile web|quest|vrchat_pc|prop]
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


def json_arg(text):
    """A JSON value given on the command line, or ``@file`` to read it from a file."""
    if text.startswith("@"):
        with open(text[1:], encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(text)


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
    parser.add_argument("--detach", action="store_true", help="queue the command as a job and return its id at once")
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
    p.add_argument("--diff", action="store_true", help="also report what changed in the scene")
    p.add_argument("--timeout", type=float)
    p = sub.add_parser("rebuild")
    p.add_argument("path", help="builder script, run in a fresh namespace")
    p.add_argument("argv", nargs="*", help="passed to the script as sys.argv[1:]")
    p.add_argument("--root", help="reload modules under this folder (default: the script's folder)")
    p.add_argument("--diff", action="store_true", help="also report what changed in the scene")
    p.add_argument("--timeout", type=float)
    p = sub.add_parser("checkpoint")
    p.add_argument("name")
    p.add_argument("collection")
    p.add_argument("--replace", action="store_true", help="overwrite a checkpoint of the same name")
    p = sub.add_parser("rollback")
    p.add_argument("name")
    p = sub.add_parser("checkpoints")
    p.add_argument("--delete", metavar="NAME")
    p = sub.add_parser("check")
    p.add_argument("names", nargs="*", help="mesh objects; default is the selection, else every mesh")
    p.add_argument("--max-tris", type=int, help="budget for triangles after modifiers, all objects together")
    p.add_argument("--max-slots", type=int, help="budget for material slots, all objects together")
    p.add_argument("--base", action="store_true", help="count defects before modifiers instead of after")
    p.add_argument("--seam", action="append", type=json_arg, metavar="JSON",
                   help='{"object": A, "points": [[x,y,z],...]} or {"object": A, "group": G, "other": B, '
                        '"other_group": H}, optional "tolerance"; @file reads JSON from a file')
    p.add_argument("--clearance", action="append", type=json_arg, metavar="JSON",
                   help='{"garment": G, "body": B, "threshold": 0.002, "ignore_groups": [...]}')
    p.add_argument("--args", type=json_arg, metavar="JSON", help="further arguments, e.g. boundary_loops")
    p = sub.add_parser("snapshot")
    p.add_argument("name")
    p = sub.add_parser("diff")
    p.add_argument("name", help="snapshot to compare from")
    p.add_argument("--to", help="snapshot to compare with; default is the scene as it is now")
    p = sub.add_parser("contact_sheet")
    p.add_argument("path")
    p.add_argument("names", nargs="+")
    p.add_argument("--views", nargs="+")
    p.add_argument("--closeup-bone", nargs=2, metavar=("RIG", "BONE"))
    p.add_argument("--closeup-object", metavar="NAME")
    p.add_argument("--closeup-size", type=float, help="width shown by the close-up, in scene units")
    p.add_argument("--tile", type=int, help="pixels per view, at most 400")
    p.add_argument("--samples", type=int, help="Cycles samples, at most 24")
    p.add_argument("--columns", type=int)
    p.add_argument("--timeout", type=float)
    p = sub.add_parser("submit")
    p.add_argument("command")
    p.add_argument("--args", type=json_arg, metavar="JSON", default={})
    for name in ("status", "result", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("job")
        if name == "result":
            p.add_argument("--wait", type=float, help="seconds to wait for the job to finish")
            p.add_argument("--keep", action="store_true", help="leave the result fetchable")
    sub.add_parser("jobs")
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
    p = sub.add_parser("validate")
    p.add_argument("names", nargs="*", help="objects to check; default is the selection, else every mesh")
    p.add_argument("--profile", default="web")
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
        for key in ("label", "reset", "diff", "timeout"):
            if getattr(opts, key):
                args[key] = getattr(opts, key)
    elif opts.cmd == "rebuild":
        args = {"path": os.path.abspath(opts.path), "argv": opts.argv}
        if opts.root:
            args["root"] = os.path.abspath(opts.root)
        for key in ("diff", "timeout"):
            if getattr(opts, key):
                args[key] = getattr(opts, key)
    elif opts.cmd == "checkpoint":
        args = {"name": opts.name, "collection": opts.collection, "replace": opts.replace}
    elif opts.cmd == "checkpoints":
        if opts.delete:
            args["delete"] = opts.delete
    elif opts.cmd == "check":
        args = dict(opts.args or {})
        if opts.names:
            args["names"] = opts.names
        budgets = {"triangles": opts.max_tris, "material_slots": opts.max_slots}
        budgets = {key: value for key, value in budgets.items() if value is not None}
        if budgets:
            args["budgets"] = {**args.get("budgets", {}), **budgets}
        if opts.base:
            args["evaluated"] = False
        for key in ("seam", "clearance"):
            if getattr(opts, key):
                args[key] = getattr(opts, key)
    elif opts.cmd == "diff":
        args = {"name": opts.name}
        if opts.to:
            args["to"] = opts.to
    elif opts.cmd == "contact_sheet":
        args = {"path": os.path.abspath(opts.path), "names": opts.names}
        for key in ("views", "tile", "samples", "columns", "timeout"):
            if getattr(opts, key):
                args[key] = getattr(opts, key)
        if opts.closeup_bone:
            args["closeup"] = {"armature": opts.closeup_bone[0], "bone": opts.closeup_bone[1]}
        elif opts.closeup_object:
            args["closeup"] = {"object": opts.closeup_object}
        if opts.closeup_size and "closeup" in args:
            args["closeup"]["size"] = opts.closeup_size
    elif opts.cmd == "submit":
        args = {"cmd": opts.command, "args": opts.args}
    elif opts.cmd in {"status", "result", "cancel"}:
        args = {"job": opts.job}
        if opts.cmd == "result":
            for key in ("wait", "keep"):
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
    elif opts.cmd == "validate":
        args = {"profile": opts.profile}
        if opts.names:
            args["names"] = opts.names
    elif opts.cmd == "object":
        if opts.name:
            args["name"] = opts.name
    else:
        for key in ("limit", "query", "label", "name"):
            if getattr(opts, key, None):
                args[key] = getattr(opts, key)

    cmd = opts.cmd
    if opts.detach and cmd not in {"submit", "status", "result", "cancel", "jobs"}:
        cmd, args = "submit", {"cmd": cmd, "args": args}
    # ``result --wait`` holds the connection open for that long.
    timeout = (args.get("timeout") or args.get("wait") or 600.0) + 60.0
    response = call(pick_session(opts.pid), cmd, args, timeout)
    # A fetched result is printed like the answer of the command that produced it.
    shown = "exec" if cmd == "result" and "stdout" in response else cmd

    if opts.json:
        print(json.dumps(response, indent=1))
    elif shown in {"exec", "rebuild"}:
        if response.get("stdout"):
            sys.stdout.write(response["stdout"])
            if not response["stdout"].endswith("\n"):
                sys.stdout.write("\n")
        if response.get("ok") and response.get("result") is not None:
            result = response["result"]
            print(result if isinstance(result, str) else json.dumps(result, indent=1))
        extra = {key: response[key] for key in ("reloaded", "imported", "diff", "diff_error") if response.get(key)}
        if extra:
            print(json.dumps(extra, indent=1))
    elif response.get("ok"):
        print(json.dumps({k: v for k, v in response.items() if k != "ok"}, indent=1))
    if not response.get("ok"):
        sys.stderr.write(response.get("error", "failed") + "\n")
        if response.get("job"):
            # Timed out or still running: the id is what the caller needs next.
            sys.stderr.write("job {:s} is {:s}\n".format(response["job"], response.get("state", "unknown")))
        return 1
    # Checks answer "ok" when they ran; whether the asset passed is a separate matter.
    if response.get("passed") is False:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
