# SPDX-FileCopyrightText: 2026 Elyan Labs
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Read and change the numbers in a contract file.

A contract is a Python module of named constants that builder scripts import:
landmarks, radii, counts, budgets. When the artist says "neckline higher", the
right edit is one of those numbers followed by a rebuild, not a hand-moved
mesh. This lists the constants, and changes one while keeping the file's
comments and layout, recording the change and keeping a copy of the old file.

No ``bpy`` here: it runs in Blender through the bridge or on its own.
"""

import ast
import datetime
import os
import re
import shutil

_VERSION = re.compile(r"^- v(\d+)\b", re.MULTILINE)


class ContractError(ValueError):
    pass


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def read(path):
    """
    The constants of a contract: ``[{name, value, line, comment}]`` in file order.

    Only plain assignments of literal values (numbers, strings, tuples, lists,
    dicts of those) count; anything computed is left out.
    """
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    lines = source.splitlines()
    entries = []
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        value = _literal(node.value)
        if value is None and not (isinstance(node.value, ast.Constant) and node.value.value is None):
            continue
        tail = lines[node.end_lineno - 1][node.end_col_offset:]
        entries.append({
            "name": node.targets[0].id,
            "value": value,
            "line": node.lineno,
            "comment": tail.split("#", 1)[1].strip() if "#" in tail else "",
        })
    return entries


def _same_shape(old, new):
    """Whether ``new`` can stand where ``old`` stood: same kind, and same length for sequences."""
    number = (int, float)
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool)
    if isinstance(old, number) and isinstance(new, number):
        return True
    if isinstance(old, (tuple, list)) and isinstance(new, (tuple, list)):
        return len(old) == len(new) and all(_same_shape(a, b) for a, b in zip(old, new))
    return type(old) is type(new)


def _as(old, new):
    """JSON has no tuples; give sequences the type they had in the file."""
    if isinstance(old, tuple) and isinstance(new, (list, tuple)):
        return tuple(_as(a, b) for a, b in zip(old, new))
    if isinstance(old, list) and isinstance(new, (list, tuple)):
        return [_as(a, b) for a, b in zip(old, new)]
    if isinstance(old, float) and isinstance(new, int) and not isinstance(new, bool):
        return float(new)
    return new


def _log_line(source, text, today):
    """The source with the change recorded in its changelog, wherever the file keeps one."""
    module = ast.parse(source)
    docstring = ast.get_docstring(module, clean=False)
    if docstring is not None and "CHANGELOG" in docstring:
        # Continue the file's own "- vN (date, who): what" list inside the docstring.
        versions = [int(n) for n in _VERSION.findall(docstring)]
        entry = "- v{:d} ({:s}, agent): {:s}\n".format(max(versions, default=1) + 1, today, text)
        node = module.body[0]
        lines = source.splitlines(keepends=True)
        closing = lines[node.end_lineno - 1]
        quote = closing.rstrip()[-3:]
        cut = closing.rfind(quote)
        before = closing[:cut]
        if before.strip():
            before = before.rstrip() + "\n"
        lines[node.end_lineno - 1] = before + entry + quote + closing[cut + 3:]
        return "".join(lines)
    marker = "# CHANGELOG\n"
    if marker not in source:
        source = source.rstrip("\n") + "\n\n" + marker
    return source.rstrip("\n") + "\n# - {:s}: {:s}\n".format(today, text)


def write(path, name, value, note="", force=False):
    """
    Set one constant. Returns ``{name, old, new, line, backup}``.

    The new value must be the same kind of thing as the old one (a number for a
    number, three numbers for three numbers) unless ``force``. The old file is
    copied into ``contract_history/`` beside it first, and the change is added
    to the file's changelog.
    """
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    target = None
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            target = node
    if target is None:
        known = ", ".join(entry["name"] for entry in read(path))
        raise ContractError("no constant named {!r} in {:s}; it has: {:s}".format(name, os.path.basename(path), known))
    old = _literal(target.value)
    if old is None and not force:
        raise ContractError("{:s} is computed, not a plain value; edit the file by hand".format(name))
    if not force and not _same_shape(old, value):
        raise ContractError("{:s} is {!r}; {!r} is a different kind of value".format(name, old, value))
    value = _as(old, value)
    if value == old:
        return {"name": name, "old": old, "new": value, "line": target.lineno, "backup": None, "unchanged": True}

    lines = source.splitlines(keepends=True)
    node = target.value
    first, last = node.lineno - 1, node.end_lineno - 1
    head, tail = lines[first][:node.col_offset], lines[last][node.end_col_offset:]
    lines[first:last + 1] = [head + repr(value) + tail]
    today = datetime.date.today().isoformat()
    text = "{:s} {!r} -> {!r}".format(name, old, value) + ("; " + note if note else "")
    updated = _log_line("".join(lines), text, today)
    # A contract that no longer parses would break every builder at once.
    ast.parse(updated)

    history = os.path.join(os.path.dirname(os.path.abspath(path)), "contract_history")
    os.makedirs(history, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    backup = os.path.join(history, "{:s}_{:s}.py".format(stem, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(updated)
    return {"name": name, "old": old, "new": value, "line": target.lineno, "backup": backup}


def cmd_contract(args):
    """
    List a contract's constants, or change one.

    ``file`` is the contract. With ``name`` and ``value``, that constant is set
    (``note`` says why, ``force`` allows a different kind of value); with only
    ``name``, that one is returned; otherwise all are. After a change, run
    ``rebuild`` on the builder script to see it.
    """
    path = args.get("file")
    if not path:
        raise ContractError("contract needs 'file'")
    path = os.path.abspath(os.path.expanduser(path))
    if "value" in args:
        if not args.get("name"):
            raise ContractError("setting a value needs 'name'")
        return {"changed": write(path, args["name"], args["value"], args.get("note", ""), bool(args.get("force")))}
    entries = read(path)
    if args.get("name"):
        entries = [entry for entry in entries if entry["name"] == args["name"]]
        if not entries:
            raise ContractError("no constant named {!r}".format(args["name"]))
    return {"file": path, "constants": entries}


COMMANDS = {"contract": cmd_contract}
