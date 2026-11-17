"""Output compression for code edits: the model writes edit ops, a decompressor applies them to the source.

Ops inside one ```edit block. Payload lines follow their op and are taken raw (no quoting).
SCOPE is a function or class name, or `module` for the lines outside every top-level function and class:
    rename OLD NEW         rename identifier OLD to NEW everywhere (whole word)
    replace SCOPE          replace every occurrence of a text inside SCOPE
    old: <text>
    new: <text>
    insert_start NAME      add lines at the start of function or class NAME (after its docstring)
    line: <code>
    insert_end NAME        add lines at the end of function or class NAME
    line: <code>
    insert_before SCOPE    add lines before the one line in SCOPE that contains <text>
    anchor: <text>
    line: <code>
    insert_after SCOPE     add lines after the one line in SCOPE that contains <text>
    anchor: <text>
    line: <code>
    delete NAME            remove function or class NAME
    rewrite                the edit does not fit these ops; the caller regenerates the whole file

GRAMMAR constrains decoding so the output always parses. apply_ops raises ApplyError when an op cannot be
applied unambiguously, inserted lines redefine an existing function or class, the result does not parse,
nothing changed, or a function now reads a name that nothing defines. The caller then falls back to a
full rewrite, so a failed compression costs time, not correctness.
"""

import ast
import builtins
import re
import symtable
import textwrap

FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.S)

GRAMMAR = r"""
root ::= "```edit\n" ops "```"
ops ::= "rewrite\n" | edit+
edit ::= rename | replace | insstart | insend | insbefore | insafter | delete
rename ::= "rename " name " " name "\n"
replace ::= "replace " name "\nold: " text "\nnew: " opt "\n"
insstart ::= "insert_start " name "\n" lines
insend ::= "insert_end " name "\n" lines
insbefore ::= "insert_before " name "\nanchor: " text "\n" lines
insafter ::= "insert_after " name "\nanchor: " text "\n" lines
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
        elif kind == "replace" and len(head) == 2 and i + 1 < len(lines):
            ops.append(("replace", head[1], _payload(lines[i], "old"), _payload(lines[i + 1], "new")))
            i += 2
        elif kind in ("insert_start", "insert_end", "insert_before", "insert_after") and len(head) == 2:
            anchor = None
            if kind in ("insert_before", "insert_after"):
                anchor = _payload(lines[i], "anchor") if i < len(lines) else ""
                i += 1
            body = []
            while i < len(lines) and lines[i].startswith("line:"):
                body.append(_payload(lines[i], "line"))
                i += 1
            if anchor == "" or not body:
                raise ApplyError(f"{kind} needs an anchor and lines")
            ops.append((kind, head[1], body) if anchor is None else (kind, head[1], anchor, body))
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


def _span(node):
    """0-based [start, end) line range of a def or class, decorators included."""
    return min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1, node.end_lineno


def _region(lines, scope, renames):
    """Line indexes an op may touch: one function or class, or `module` for lines outside all of them."""
    if scope == "module":
        inside = set()
        for node in ast.parse("\n".join(lines)).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inside.update(range(*_span(node)))
        return [k for k in range(len(lines)) if k not in inside]
    return list(range(*_span(_scope(lines, scope, renames))))


DEF_RE = re.compile(r"\s*(?:async\s+)?(?:def|class)\s+(\w+)")


def _check_redefine(body, lines):
    """Inserted lines must not define a name that already exists, such as a copy of the target function."""
    existing = {q.rsplit(".", 1)[-1] for q, _ in _scopes(lines)}
    for b in body:
        m = DEF_RE.match(b)
        if m and m.group(1) in existing:
            raise ApplyError(f"inserted lines redefine {m.group(1)!r}")


def _undefined(code):
    """(function, name) pairs read inside a function that no scope, import or builtin defines."""
    top = symtable.symtable(code, "<edit>", "exec")
    known = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()} | set(dir(builtins))
    out = set()

    def walk(t):
        for c in t.get_children():
            if c.get_type() == "function":
                out.update((c.get_name(), s.get_name()) for s in c.get_symbols()
                           if s.is_referenced() and s.is_global() and not s.is_declared_global()
                           and s.get_name() not in known)
            walk(c)

    walk(top)
    return out


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
            _, scope, old, new = op
            hits = [k for k in _region(lines, scope, renames) if old in lines[k]]
            if not hits:
                raise ApplyError(f"replace: {old!r} not found in {scope}")
            for k in hits:
                lines[k] = lines[k].replace(old, new)
        elif kind in ("insert_start", "insert_end"):
            _, name, body = op
            _check_redefine(body, lines)
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
        elif kind in ("insert_before", "insert_after"):
            _, scope, anchor, body = op
            _check_redefine(body, lines)
            hits = [k for k in _region(lines, scope, renames) if anchor.strip() in lines[k]]
            if len(hits) != 1:
                raise ApplyError(f"anchor {anchor!r} matched {len(hits)} lines in {scope}")
            k = hits[0]
            indent = _indent(lines[k])
            if kind == "insert_before":
                lines[k:k] = _place(body, indent)
                continue
            if lines[k].rstrip().endswith(":"):
                nxt = next((x for x in lines[k + 1 :] if x.strip()), None)
                indent = _indent(nxt) if nxt is not None and len(_indent(nxt)) > len(indent) else indent + "    "
            lines[k + 1 : k + 1] = _place(body, indent)
        elif kind == "delete":
            a, b = _span(_scope(lines, op[1], renames))
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
    new_names = _undefined(result) - _undefined(source)
    if new_names:
        raise ApplyError(f"edit reads undefined names {sorted(new_names)}")
    return result


def decompress(text: str, source: str) -> str:
    """Parse and apply model output. Raises ApplyError when the caller should fall back."""
    return apply_ops(parse_ops(text), source)


FORMAT_RULES = """You edit code by writing edit operations, never the edited code itself. Reply with one ```edit block using only these operations. SCOPE is the name of the function or class the change belongs in, or `module` for lines outside every function and class.
rename OLD NEW         rename the identifier OLD to NEW everywhere
replace SCOPE          replace every occurrence of a piece of text inside SCOPE
old: <exact text from the code>
new: <replacement text>
insert_start NAME      add lines at the start of function or class NAME
line: <one code line>
insert_end NAME        add lines at the end of function or class NAME
line: <one code line>
insert_before SCOPE    add lines right before the one line in SCOPE that contains the anchor text
anchor: <exact text from one line>
line: <one code line>
insert_after SCOPE     add lines right after the one line in SCOPE that contains the anchor text
anchor: <exact text from one line>
line: <one code line>
delete NAME            remove function or class NAME
rewrite                use this alone when the change cannot be written with the operations above
Repeat `line:` for several lines. Indent lines only relative to each other."""

_EX_SOURCE = """SCALE = 1


def area(w, h):
    return w * h * SCALE


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
    ("Make `area` return a float.", "replace area\nold: return w * h * SCALE\nnew: return float(w * h * SCALE)"),
    ("Use a scale of 2.", "replace module\nold: SCALE = 1\nnew: SCALE = 2"),
    ("Print each shape right before its area is printed.",
     "insert_before report\nanchor: print(area(s.w, s.h))\nline: print(s)"),
    ("Skip shapes that are None.", "insert_after report\nanchor: for s in shapes:\nline: if s is None:\nline:     continue"),
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
