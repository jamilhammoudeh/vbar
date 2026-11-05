"""The package reads no model key: the only variables it looks at are its own and JAVA_HOME."""

import ast
from pathlib import Path

from vbar.eval.results import HIDDEN_LOG_VARIABLE
from vbar.scenarios.seeds import HIDDEN_KEY_VARIABLE

SOURCE = Path(__file__).resolve().parents[1] / "src" / "vbar"
CONSTANTS = {
    "HIDDEN_LOG_VARIABLE": HIDDEN_LOG_VARIABLE,
    "HIDDEN_KEY_VARIABLE": HIDDEN_KEY_VARIABLE,
}


def is_os_attribute(node, attribute):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attribute
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def variable_name(key):
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    if isinstance(key, ast.Name):
        return CONSTANTS.get(key.id)
    return None


def read_key(node, parent, grandparent):
    """The expression naming the variable that `node`, an os.environ or os.getenv use, reads."""
    if is_os_attribute(node, "getenv"):
        if isinstance(parent, ast.Call) and parent.func is node and parent.args:
            return parent.args[0]
    elif isinstance(parent, ast.Subscript) and parent.value is node:
        return parent.slice
    elif isinstance(parent, ast.Attribute) and parent.attr == "get" and parent.value is node:
        if isinstance(grandparent, ast.Call) and grandparent.func is parent and grandparent.args:
            return grandparent.args[0]
    elif isinstance(parent, ast.Compare) and parent.comparators == [node]:
        if isinstance(parent.ops[0], ast.In | ast.NotIn):
            return parent.left
    return None


def scan(source):
    """The variable names a module looks up, and the places it uses the environment otherwise."""
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    names, unresolved = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            unresolved.extend(
                node.lineno for alias in node.names if alias.name in ("environ", "getenv")
            )
        if not (is_os_attribute(node, "environ") or is_os_attribute(node, "getenv")):
            continue
        parent = parents[node]
        key = read_key(node, parent, parents.get(parent))
        name = None if key is None else variable_name(key)
        if name is None:
            unresolved.append(node.lineno)
        else:
            names.add(name)
    return names, sorted(unresolved)


def test_scan_resolves_every_form_a_variable_can_be_read_in():
    source = (
        "import os\n"
        "a = os.environ.get('A', '')\n"
        "b = os.getenv('B')\n"
        "c = os.environ['C']\n"
        "d = 'D' in os.environ\n"
        "e = 'E' not in os.environ\n"
        "f = os.environ.get(HIDDEN_KEY_VARIABLE)\n"
    )
    names, unresolved = scan(source)
    assert names == {"A", "B", "C", "D", "E", HIDDEN_KEY_VARIABLE}
    assert unresolved == []


def test_scan_reports_a_read_whose_name_it_cannot_resolve():
    source = "import os\nsecrets = dict(os.environ)\nkey = os.getenv(name)\n"
    assert scan(source) == (set(), [2, 3])


def test_the_package_reads_only_java_home_and_the_hidden_split_variables():
    names, unresolved = set(), []
    for path in sorted(SOURCE.rglob("*.py")):
        found, lines = scan(path.read_text(encoding="utf-8"))
        names |= found
        unresolved += [f"{path.relative_to(SOURCE)}:{line}" for line in lines]
    assert unresolved == []
    assert names == {"JAVA_HOME", HIDDEN_LOG_VARIABLE, HIDDEN_KEY_VARIABLE}
