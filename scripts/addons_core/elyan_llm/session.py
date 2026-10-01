# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Session files: how a client finds a running bridge and its token.

Kept free of ``bpy`` so the stand-alone client can import it too.
"""

import json
import os
import sys


def session_dir():
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "elyan-blender", "llm-bridge")


def session_path(pid):
    return os.path.join(session_dir(), "{:d}.json".format(pid))


def write(info):
    """Write the session file readable by the owner only (it holds the token)."""
    os.makedirs(session_dir(), mode=0o700, exist_ok=True)
    path = session_path(info["pid"])
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    return path


def remove(pid):
    try:
        os.remove(session_path(pid))
    except OSError:
        pass


def _pid_alive(pid):
    if sys.platform == "win32":
        # No cheap portable check; the client finds out when it connects.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def list_live():
    """All sessions whose Blender process still exists, stale files are removed."""
    result = []
    try:
        names = os.listdir(session_dir())
    except OSError:
        return result
    for name in sorted(names):
        if not name.endswith(".json"):
            continue
        path = os.path.join(session_dir(), name)
        try:
            with open(path, encoding="utf-8") as fh:
                info = json.load(fh)
        except (OSError, ValueError):
            continue
        if _pid_alive(int(info.get("pid", -1))):
            result.append(info)
        else:
            try:
                os.remove(path)
            except OSError:
                pass
    return result
