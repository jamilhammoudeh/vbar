"""The environment variables the package reads: its own, JAVA_HOME and the config folder's."""

import ast
from pathlib import Path

from vbar.eval.results import HIDDEN_LOG_VARIABLE
from vbar.scenarios.seeds import HIDDEN_KEY_VARIABLE

SOURCE = Path(__file__).resolve().parents[1] / "src" / "vbar"
CONSTANTS = {
    "HIDDEN_LOG_VARIABLE": HIDDEN_LOG_VARIABLE,
    "HIDDEN_KEY_VARIABLE": HIDDEN_KEY_VARIABLE,
}


ACCESSORS = ("environ", "getenv")


def accessor(node, aliases):
    """The os attribute, environ or getenv, that `node` fetches by name or through getattr."""
    if isinstance(node, ast.Attribute):
        if node.attr in ACCESSORS and isinstance(node.value, ast.Name) and node.value.id in aliases:
            return node.attr
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "getattr" and len(node.args) >= 2:
            module, attribute = node.args[:2]
            if isinstance(module, ast.Name) and module.id in aliases:
                if isinstance(attribute, ast.Constant) and attribute.value in ACCESSORS:
                    return attribute.value
    return None


def os_aliases(tree):
    """The names the os module is bound to: `os`, and each `import os as name`."""
    aliases = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(a.asname for a in node.names if a.name == "os" and a.asname)
    return aliases


def variable_name(key):
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    if isinstance(key, ast.Name):
        return CONSTANTS.get(key.id)
    return None


def read_key(kind, node, parent, grandparent):
    """The expression naming the variable that `node`, an os.environ or os.getenv use, reads."""
    if kind == "getenv":
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
    aliases = os_aliases(tree)
    names, unresolved = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            unresolved.extend(
                node.lineno for alias in node.names if alias.name in ("environ", "getenv")
            )
        kind = accessor(node, aliases)
        if kind is None:
            continue
        parent = parents[node]
        key = read_key(kind, node, parent, parents.get(parent))
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


def test_scan_reports_a_variable_read_through_an_alias_of_os():
    source = "import os as _o\na = _o.environ.get('A')\nb = _o.getenv('B')\nc = dict(_o.environ)\n"
    assert scan(source) == ({"A", "B"}, [4])


def test_scan_reports_a_variable_read_through_getattr():
    source = (
        "import os\n"
        "a = getattr(os, 'environ').get('A')\n"
        "b = getattr(os, 'getenv')('B')\n"
        "c = dict(getattr(os, 'environ'))\n"
    )
    assert scan(source) == ({"A", "B"}, [4])


def test_scan_reports_a_variable_read_through_getattr_on_an_alias():
    source = "import os as _o\na = getattr(_o, 'environ')['A']\nb = 'B' in getattr(_o, 'environ')\n"
    assert scan(source) == ({"A", "B"}, [])


def test_the_package_reads_only_its_own_variables_java_home_and_the_config_folders():
    names, unresolved = set(), []
    for path in sorted(SOURCE.rglob("*.py")):
        found, lines = scan(path.read_text(encoding="utf-8"))
        names |= found
        unresolved += [f"{path.relative_to(SOURCE)}:{line}" for line in lines]
    assert unresolved == []
    # The stored key's folder follows APPDATA on Windows and XDG_CONFIG_HOME elsewhere; the
    # only key the package reads from the environment is its own VBAR_LLM_API_KEY.
    assert names == {
        "JAVA_HOME",
        HIDDEN_LOG_VARIABLE,
        HIDDEN_KEY_VARIABLE,
        "VBAR_CONFIG_DIR",
        "VBAR_LLM_API_KEY",
        "APPDATA",
        "XDG_CONFIG_HOME",
    }
