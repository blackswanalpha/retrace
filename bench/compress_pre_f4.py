"""Output compression for code edits: the model writes edit ops, a decompressor applies them to the source.

Ops inside one ```edit block. Payload lines follow their op and are taken raw (no quoting):
    rename OLD NEW      rename identifier OLD to NEW everywhere (whole word)
    replace             replace every occurrence of a text
    old: <text>
    new: <text>
    insert_start NAME   add lines at the start of function or class NAME (after its docstring)
    line: <code>
    insert_end NAME     add lines at the end of function or class NAME
    line: <code>
    insert_after        add lines after the one line that contains <text>
    anchor: <text>
    line: <code>
    delete NAME         remove function or class NAME
    rewrite             the edit does not fit these ops; the caller regenerates the whole file

GRAMMAR constrains decoding so the output always parses. apply_ops raises ApplyError when an op cannot be
applied unambiguously, the result does not parse, or nothing changed. The caller then falls back to a
full rewrite, so a failed compression costs time, not correctness.
"""

import ast
import re
import textwrap

FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.S)

GRAMMAR = r"""
root ::= "```edit\n" ops "```"
ops ::= "rewrite\n" | edit+
edit ::= rename | replace | insstart | insend | insafter | delete
rename ::= "rename " name " " name "\n"
replace ::= "replace\nold: " text "\nnew: " opt "\n"
insstart ::= "insert_start " name "\n" lines
insend ::= "insert_end " name "\n" lines
insafter ::= "insert_after\nanchor: " text "\n" lines
delete ::= "delete " name "\n"
lines ::= ("line: " opt "\n")+
name ::= [A-Za-z_] [A-Za-z0-9_.]*
text ::= [^\n]+
opt ::= [^\n]*
"""


class ApplyError(Exception):
    pass


def extract_code(text: str) -> str:
    """Content of the last fenced block, or the whole text when there is none.

    Models often echo the original in one block and give the edit in a later one.
    """
    blocks = FENCE_RE.findall(text)
    return blocks[-1] if blocks else text


def normalize(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip("\n").splitlines())


def _payload(line: str, key: str) -> str:
    if not line.startswith(f"{key}:"):
        raise ApplyError(f"expected '{key}:' line, got {line!r}")
    p = line[len(key) + 1 :]
    p = p[1:] if p.startswith(" ") else p
    if len(p) > 1 and p[0] == p[-1] == "`":
        p = p[1:-1]
    return p


def parse_ops(text: str):
    lines = extract_code(text).splitlines()
    ops, i = [], 0
    while i < len(lines):
        head = lines[i].split()
        i += 1
        if not head:
            continue
        kind = head[0]
        if kind == "rename" and len(head) == 3:
            ops.append(("rename", head[1], head[2]))
        elif kind == "replace" and i + 1 < len(lines):
            ops.append(("replace", _payload(lines[i], "old"), _payload(lines[i + 1], "new")))
            i += 2
        elif kind in ("insert_start", "insert_end", "insert_after") and len(head) <= 2:
            if kind == "insert_after":
                target = _payload(lines[i], "anchor") if i < len(lines) else ""
                i += 1
            else:
                target = head[1] if len(head) == 2 else ""
            body = []
            while i < len(lines) and lines[i].startswith("line:"):
                body.append(_payload(lines[i], "line"))
                i += 1
            if not target or not body:
                raise ApplyError(f"{kind} needs a target and lines")
            ops.append((kind, target, body))
        elif kind == "delete" and len(head) == 2:
            ops.append(("delete", head[1]))
        elif kind == "rewrite":
            ops.append(("rewrite",))
        else:
            raise ApplyError(f"bad op line {lines[i - 1]!r}")
    if not ops:
        raise ApplyError("no ops")
    return ops


def _indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _scopes(lines):
    """(qualified name, node) for every function and class."""
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}{child.name}"
                out.append((q, child))
                walk(child, q + ".")
            else:
                walk(child, prefix)

    walk(ast.parse("\n".join(lines)), "")
    return out


def _scope(lines, name, renames):
    names = {name} | {n for o, n in renames if o == name} | {o for o, n in renames if n == name}
    hits = [node for q, node in _scopes(lines) if any(q == n or q.endswith("." + n) for n in names)]
    if len(hits) != 1:
        raise ApplyError(f"scope {name!r} matched {len(hits)} definitions")
    return hits[0]


def _place(body, indent):
    block = textwrap.dedent("\n".join(body)).splitlines()
    return [indent + b if b.strip() else "" for b in block]


def apply_ops(ops, source: str) -> str:
    lines = source.splitlines()
    renames = []
    for op in ops:
        kind = op[0]
        if kind == "rewrite":
            raise ApplyError("model chose rewrite")
        if kind == "rename":
            _, old, new = op
            pat = re.compile(rf"\b{re.escape(old)}\b")
            if not any(pat.search(x) for x in lines):
                raise ApplyError(f"rename: {old!r} not found")
            lines = [pat.sub(lambda _: new, x) for x in lines]
            renames.append((old, new))
        elif kind == "replace":
            _, old, new = op
            text = "\n".join(lines)
            if old not in text:
                raise ApplyError(f"replace: {old!r} not found")
            lines = text.replace(old, new).split("\n")
        elif kind in ("insert_start", "insert_end"):
            _, name, body = op
            node = _scope(lines, name, renames)
            first = node.body[0]
            indent = _indent(lines[first.lineno - 1])
            if kind == "insert_end":
                at = node.end_lineno
            elif isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                at = first.end_lineno
            else:
                at = first.lineno - 1
            lines[at:at] = _place(body, indent)
        elif kind == "insert_after":
            _, anchor, body = op
            hits = [k for k, x in enumerate(lines) if anchor.strip() in x]
            if len(hits) != 1:
                raise ApplyError(f"anchor {anchor!r} matched {len(hits)} lines")
            k = hits[0]
            indent = _indent(lines[k])
            if lines[k].rstrip().endswith(":"):
                nxt = next((x for x in lines[k + 1 :] if x.strip()), None)
                indent = _indent(nxt) if nxt is not None and len(_indent(nxt)) > len(indent) else indent + "    "
            lines[k + 1 : k + 1] = _place(body, indent)
        elif kind == "delete":
            node = _scope(lines, op[1], renames)
            a = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            b = node.end_lineno
            while b < len(lines) and not lines[b].strip():
                b += 1
            if b == len(lines):
                while a > 0 and not lines[a - 1].strip():
                    a -= 1
            del lines[a:b]
    result = "\n".join(lines)
    try:
        ast.parse(result)
    except SyntaxError as e:
        raise ApplyError(f"result does not parse: {e}") from None
    if normalize(result) == normalize(source):
        raise ApplyError("ops changed nothing")
    return result


def decompress(text: str, source: str) -> str:
    """Parse and apply model output. Raises ApplyError when the caller should fall back."""
    return apply_ops(parse_ops(text), source)


FORMAT_RULES = """You edit code by writing edit operations, never the edited code itself. Reply with one ```edit block using only these operations:
rename OLD NEW         rename the identifier OLD to NEW everywhere
replace                replace every occurrence of a piece of text
old: <exact text from the code>
new: <replacement text>
insert_start NAME      add lines at the start of function or class NAME
line: <one code line>
insert_end NAME        add lines at the end of function or class NAME
line: <one code line>
insert_after           add lines right after the one line that contains the anchor text
anchor: <exact text from one line>
line: <one code line>
delete NAME            remove function or class NAME
rewrite                use this alone when the change cannot be written with the operations above
Repeat `line:` for several lines. Indent lines only relative to each other."""

_EX_SOURCE = """def area(w, h):
    return w * h


def debug(x):
    print(x)


def report(shapes):
    for s in shapes:
        print(area(s.w, s.h))"""

_EXAMPLES = [
    ("Rename `area` to `rect_area`.", "rename area rect_area"),
    ("Add `total = 0` as the first line of `report`.", "insert_start report\nline: total = 0"),
    ("Add `print('start')` as the first line of `area`, and `print('done')` as the last line of `report`.",
     "insert_start area\nline: print('start')\ninsert_end report\nline: print('done')"),
    ("Remove the `debug` function.", "delete debug"),
    ("Make `area` return a float.", "replace\nold: return w * h\nnew: return float(w * h)"),
    ("Print each shape before its area is printed.", "insert_after\nanchor: for s in shapes:\nline: print(s)"),
    ("Rewrite `report` as a single list comprehension.", "rewrite"),
]


def code_message(task: str, source: str) -> str:
    return f"{task}\n\n```python\n{source.rstrip(chr(10))}\n```"


def compressed_messages(task: str, source: str):
    """(role, text) messages asking for edit ops, after one worked example per op."""
    msgs = [("system", FORMAT_RULES)]
    for ex_task, ex_ops in _EXAMPLES:
        msgs += [("user", code_message(ex_task, _EX_SOURCE)), ("assistant", f"```edit\n{ex_ops}\n```")]
    return msgs + [("user", code_message(task, source))]
