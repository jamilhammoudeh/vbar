"""Source provenance: a digest of the local changes tells edited source trees apart."""

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

SHA = "a" * 40
TEXT, TEXT_BYTES = "dv.txt", b"one\ntwo\n"
DATA, DATA_BYTES = "table.bin", b"\x00\x01\x02\x03"
MANY, MANY_LINES = "many.txt", "".join("\n" if k % 7 == 0 else f"line {k}\n" for k in range(40))


def run_git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def head(root):
    return run_git(root, "rev-parse", "HEAD").decode().strip()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@pytest.fixture
def isolated_git(tmp_path, monkeypatch):
    """Git without the user's and the system's configuration, and no repository above tmp_path."""
    empty = tmp_path / "empty.gitconfig"
    empty.write_bytes(b"")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    # Set when the tests run from inside a git hook, and then they would override -C.
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def checkout(isolated_git, tmp_path):
    """A repository with one commit holding a text file and a binary file."""
    root = tmp_path / "tree"
    root.mkdir()
    run_git(root, "init", "-q")
    run_git(root, "config", "user.name", "t")
    run_git(root, "config", "user.email", "t@example.com")
    (root / TEXT).write_bytes(TEXT_BYTES)
    (root / DATA).write_bytes(DATA_BYTES)
    run_git(root, "add", ".")
    run_git(root, "commit", "-q", "-m", "first")
    return root


@needs_git
def test_a_clean_checkout_gives_its_commit_and_no_digest(checkout):
    from vbar.eval.provenance import _revision

    sha = head(checkout)
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert _revision(checkout) == (sha, False, None)


@needs_git
def test_an_edit_to_a_tracked_file_gives_the_same_commit_and_a_digest(checkout):
    from vbar.eval.provenance import _revision

    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    sha, dirty, digest = _revision(checkout)
    assert (sha, dirty) == (head(checkout), True)
    assert is_digest(digest)


@needs_git
def test_two_different_edits_give_different_digests(checkout):
    from vbar.eval.provenance import _revision

    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    first = _revision(checkout)[2]
    (checkout / TEXT).write_bytes(b"one\nthree\n")
    second = _revision(checkout)[2]
    assert is_digest(first) and is_digest(second) and first != second


@needs_git
def test_the_same_edit_made_twice_gives_the_same_digest(checkout):
    from vbar.eval.provenance import _revision

    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    first = _revision(checkout)
    assert _revision(checkout) == first
    (checkout / TEXT).write_bytes(TEXT_BYTES)
    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    assert _revision(checkout) == first


@needs_git
def test_editing_a_file_back_to_its_committed_content_makes_the_tree_clean(checkout):
    from vbar.eval.provenance import _revision

    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    assert _revision(checkout)[1] is True
    (checkout / TEXT).write_bytes(TEXT_BYTES)
    assert _revision(checkout) == (head(checkout), False, None)


@needs_git
def test_committing_an_edit_gives_a_clean_tree_at_the_new_commit(checkout):
    from vbar.eval.provenance import _revision

    before = head(checkout)
    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    run_git(checkout, "add", TEXT)
    run_git(checkout, "commit", "-q", "-m", "second")
    sha, dirty, digest = _revision(checkout)
    assert sha == head(checkout) != before
    assert (dirty, digest) == (False, None)


@needs_git
def test_staged_and_unstaged_edits_of_the_same_content_give_the_same_digest(checkout):
    from vbar.eval.provenance import _revision

    # The digest is of the files as they are on disk against the commit, not of the index.
    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    unstaged = _revision(checkout)
    run_git(checkout, "add", TEXT)
    staged = _revision(checkout)
    assert unstaged[:2] == staged[:2] == (head(checkout), True)
    assert is_digest(unstaged[2]) and staged == unstaged


@needs_git
def test_an_untracked_file_leaves_the_tree_clean_until_it_is_added(checkout):
    from vbar.eval.provenance import _revision

    (checkout / "extra.txt").write_bytes(b"new\n")
    assert _revision(checkout) == (head(checkout), False, None)
    run_git(checkout, "add", "extra.txt")
    sha, dirty, digest = _revision(checkout)
    assert (sha, dirty) == (head(checkout), True) and is_digest(digest)


@needs_git
def test_deleting_a_tracked_file_gives_a_digest_that_differs_from_an_edit(checkout):
    from vbar.eval.provenance import _revision

    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    edited = _revision(checkout)[2]
    (checkout / TEXT).unlink()
    sha, dirty, deleted = _revision(checkout)
    assert (sha, dirty) == (head(checkout), True)
    assert is_digest(deleted) and deleted != edited


@needs_git
def test_edits_to_a_binary_file_give_different_digests(checkout):
    from vbar.eval.provenance import _revision

    (checkout / DATA).write_bytes(b"\x00\x01\x02\xff")
    first = _revision(checkout)
    (checkout / DATA).write_bytes(b"\x00\x01\x02\xfe")
    second = _revision(checkout)
    assert first[:2] == second[:2] == (head(checkout), True)
    assert is_digest(first[2]) and is_digest(second[2]) and first[2] != second[2]


@needs_git
def test_the_digest_is_the_sha256_of_the_diff_against_head(checkout):
    from vbar.eval.provenance import _revision

    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    (checkout / DATA).write_bytes(b"\x00\x01\x02\xff\xfe")
    (checkout / "extra.txt").write_bytes(b"new\n")
    run_git(checkout, "add", "extra.txt")
    from vbar.eval.provenance import _DIFF, _GIT_SETTINGS

    expected = run_git(checkout, *_GIT_SETTINGS, *_DIFF)
    # Without --binary a changed binary file is one line that does not show the change.
    assert b"GIT binary patch" in expected
    assert _revision(checkout) == (head(checkout), True, hashlib.sha256(expected).hexdigest())


@needs_git
@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("color.ui", "always", id="color"),
        pytest.param("diff.external", "echo external", id="external-diff"),
        pytest.param("diff.upper.textconv", "echo converted", id="textconv"),
    ],
)
def test_the_users_diff_settings_do_not_change_the_digest(checkout, key, value):
    from vbar.eval.provenance import _revision

    run_git(checkout, "config", key, value)
    (checkout / ".git" / "info" / "attributes").write_bytes(f"{TEXT} diff=upper\n".encode())
    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    from vbar.eval.provenance import _DIFF, _GIT_SETTINGS

    expected = hashlib.sha256(run_git(checkout, *_GIT_SETTINGS, *_DIFF)).hexdigest()
    # The setting changes what a plain diff prints, so the test can tell if the option is missing.
    assert hashlib.sha256(run_git(checkout, "diff", "HEAD")).hexdigest() != expected
    assert _revision(checkout) == (head(checkout), True, expected)


@needs_git
@pytest.mark.parametrize(
    ("key", "value", "changes_plain_diff"),
    [
        # Abbreviated object names grow with the repository: 7 digits become 8 past 16384 objects.
        pytest.param("core.abbrev", "4", True, id="abbreviation"),
        pytest.param("diff.noprefix", "true", True, id="no-prefix"),
        pytest.param("diff.mnemonicPrefix", "true", True, id="mnemonic-prefix"),
        pytest.param("diff.context", "10", True, id="context"),
        pytest.param("diff.suppressBlankEmpty", "true", True, id="blank-lines"),
        pytest.param("diff.renames", "false", None, id="renames"),
        pytest.param("diff.algorithm", "patience", None, id="algorithm"),
        pytest.param("diff.interHunkContext", "20", None, id="hunk-merging"),
    ],
)
def test_the_digest_of_one_edit_does_not_depend_on_the_git_settings(
    checkout, key, value, changes_plain_diff
):
    from vbar.eval.provenance import _revision

    (checkout / MANY).write_bytes(MANY_LINES.encode())
    run_git(checkout, "add", MANY)
    run_git(checkout, "commit", "-q", "-m", "second")
    (checkout / MANY).write_bytes(MANY_LINES.replace("line 9\n", "line nine\n").encode())
    plain_before = run_git(checkout, "diff", "HEAD")
    before = _revision(checkout)
    run_git(checkout, "config", key, value)
    if changes_plain_diff:
        # The setting changes what a plain diff prints, so a missing option would show.
        assert run_git(checkout, "diff", "HEAD") != plain_before
    assert _revision(checkout) == before


@needs_git
def test_the_digest_of_one_edit_does_not_depend_on_a_diff_order_file(checkout):
    from vbar.eval.provenance import _revision

    (checkout / MANY).write_bytes(MANY_LINES.encode())
    run_git(checkout, "add", MANY)
    run_git(checkout, "commit", "-q", "-m", "second")
    (checkout / TEXT).write_bytes(b"one\nTWO\n")
    (checkout / MANY).write_bytes(MANY_LINES.replace("line 9\n", "line nine\n").encode())
    plain_before = run_git(checkout, "diff", "HEAD")
    before = _revision(checkout)
    order = checkout / ".git" / "order"
    order.write_bytes(f"{MANY}\n{TEXT}\n".encode())
    run_git(checkout, "config", "diff.orderFile", str(order))
    # Files print in the order of the file, not alphabetically, so a missing option would show.
    assert run_git(checkout, "diff", "HEAD") != plain_before
    assert _revision(checkout) == before


@needs_git
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    first=st.one_of(st.just(DATA_BYTES), st.binary(max_size=24)),
    second=st.one_of(st.just(DATA_BYTES), st.binary(max_size=24)),
)
def test_the_digest_follows_the_file_content_exactly(checkout, first, second):
    from vbar.eval.provenance import _revision

    (checkout / DATA).write_bytes(first)
    digest_first = _revision(checkout)[2]
    (checkout / DATA).write_bytes(second)
    digest_second = _revision(checkout)[2]
    assert (digest_first is None) == (first == DATA_BYTES)
    assert (digest_first == digest_second) == (first == second)


def test_a_folder_that_is_not_a_checkout_gives_no_revision(isolated_git, tmp_path):
    from vbar.eval.provenance import _revision

    (tmp_path / "plain").mkdir()
    assert _revision(tmp_path / "plain") == (None, False, None)


def test_a_damaged_checkout_gives_no_revision(isolated_git, tmp_path):
    from vbar.eval.provenance import _revision

    (tmp_path / "damaged" / ".git").mkdir(parents=True)
    assert _revision(tmp_path / "damaged") == (None, False, None)


@needs_git
def test_a_repository_without_a_commit_gives_no_revision(isolated_git, tmp_path):
    from vbar.eval.provenance import _revision

    root = tmp_path / "fresh"
    root.mkdir()
    run_git(root, "init", "-q")
    (root / TEXT).write_bytes(TEXT_BYTES)
    run_git(root, "add", TEXT)
    assert _revision(root) == (None, False, None)


def test_a_missing_git_gives_no_revision(isolated_git, tmp_path, monkeypatch):
    from vbar.eval.provenance import _revision

    (tmp_path / "tree" / ".git").mkdir(parents=True)
    monkeypatch.setenv("PATH", "")
    assert _revision(tmp_path / "tree") == (None, False, None)


def fields_under(monkeypatch, revision):
    from vbar.agents import PDAgent
    from vbar.eval import provenance

    monkeypatch.setattr(provenance, "source_revision", lambda: revision)
    return provenance.provenance_fields(
        task="space",
        identity={"scenario_id": "vbar:approach-vbar:cw:box:dv:v1", "dynamics": "cw"},
        seed=3,
        agent_seed=5,
        agent=PDAgent(),
    )


def test_the_fields_carry_the_digest_beside_the_commit_and_the_dirty_flag(monkeypatch):
    edited = fields_under(monkeypatch, (SHA, True, "1" * 64))
    assert (edited["git"], edited["dirty"], edited["diff"]) == (SHA, True, "1" * 64)
    clean = fields_under(monkeypatch, (SHA, False, None))
    assert (clean["git"], clean["dirty"]) == (SHA, False)
    assert "diff" in clean and clean["diff"] is None


def test_two_edits_of_one_commit_hash_differently(monkeypatch):
    from vbar.eval.provenance import provenance_hash

    first = fields_under(monkeypatch, (SHA, True, "1" * 64))
    second = fields_under(monkeypatch, (SHA, True, "2" * 64))
    assert provenance_hash(first) != provenance_hash(second)


def test_the_same_edit_of_one_commit_hashes_the_same(monkeypatch):
    from vbar.eval.provenance import provenance_hash

    first = fields_under(monkeypatch, (SHA, True, "1" * 64))
    second = fields_under(monkeypatch, (SHA, True, "1" * 64))
    assert provenance_hash(first) == provenance_hash(second)


def test_an_edited_tree_hashes_differently_from_the_clean_commit(monkeypatch):
    from vbar.eval.provenance import provenance_hash

    clean = fields_under(monkeypatch, (SHA, False, None))
    edited = fields_under(monkeypatch, (SHA, True, "1" * 64))
    assert provenance_hash(clean) != provenance_hash(edited)


def fly_under(monkeypatch, revision, out):
    from eval_cases import ShortTask

    from vbar.agents import PDAgent
    from vbar.eval import Job, provenance, run_jobs
    from vbar.scenarios import seeds

    monkeypatch.setattr(provenance, "source_revision", lambda: revision)
    pairs = seeds("approach-vbar", "validation", 2).pairs()
    work = [
        Job("approach-vbar", "validation", k, s, a, "test-suite") for k, (s, a) in enumerate(pairs)
    ]
    return run_jobs(work, PDAgent, task=ShortTask(), out=out)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param((SHA, False, None), (SHA, True, "1" * 64), id="clean-then-edited"),
        pytest.param((SHA, True, "1" * 64), (SHA, True, "2" * 64), id="edited-then-edited-again"),
        pytest.param((SHA, True, "1" * 64), (SHA, False, None), id="edited-then-reverted"),
    ],
)
def test_a_resumed_run_flies_again_when_the_source_changed(tmp_path, monkeypatch, before, after):
    from vbar.eval import read_results
    from vbar.eval.report import latest_results

    out = tmp_path / "results.jsonl"
    first = fly_under(monkeypatch, before, out)
    second = fly_under(monkeypatch, after, out)
    stored = read_results(out)
    assert len(stored) == 4
    assert not {r["provenance"] for r in first} & {r["provenance"] for r in second}
    assert [r["provenance"] for r in stored] == [r["provenance"] for r in first + second]
    assert [r["environment"]["diff"] for r in stored] == [before[2]] * 2 + [after[2]] * 2
    assert [r["provenance"] for r in latest_results(stored)] == [r["provenance"] for r in second]


@pytest.mark.parametrize(
    "revision",
    [
        pytest.param((SHA, False, None), id="clean"),
        pytest.param((SHA, True, "1" * 64), id="edited"),
    ],
)
def test_a_resumed_run_flies_nothing_again_when_the_source_is_the_same(
    tmp_path, monkeypatch, revision
):
    from vbar.eval import read_results

    out = tmp_path / "results.jsonl"
    first = fly_under(monkeypatch, revision, out)
    again = fly_under(monkeypatch, revision, out)
    assert len(read_results(out)) == 2
    assert [r["provenance"] for r in again] == [r["provenance"] for r in first]


def test_the_environment_records_the_digest_of_the_local_changes(monkeypatch):
    from vbar.eval import provenance

    monkeypatch.setattr(provenance, "source_revision", lambda: (SHA, True, "c" * 64))
    recorded = provenance.environment()
    assert (recorded["git"], recorded["dirty"], recorded["diff"]) == (SHA, True, "c" * 64)


def test_the_environment_of_a_clean_tree_records_no_digest(monkeypatch):
    from vbar.eval import provenance

    monkeypatch.setattr(provenance, "source_revision", lambda: (SHA, False, None))
    recorded = provenance.environment()
    assert "diff" in recorded and recorded["diff"] is None


def test_the_environment_matches_the_revision_of_the_package_checkout():
    from vbar.eval.provenance import environment, source_revision

    sha, dirty, digest = source_revision()
    recorded = environment()
    assert (recorded["git"], recorded["dirty"], recorded["diff"]) == (sha, dirty, digest)
    assert (digest is not None) == dirty


def test_the_revision_of_the_package_checkout_is_read_once_per_process(monkeypatch):
    from vbar.eval import provenance

    roots = []

    def fake(root):
        roots.append(Path(root))
        return SHA, True, "d" * 64

    monkeypatch.setattr(provenance, "_revision", fake)
    provenance.source_revision.cache_clear()
    try:
        assert provenance.source_revision() == (SHA, True, "d" * 64)
        provenance.source_revision()
    finally:
        provenance.source_revision.cache_clear()
    assert roots == [Path(provenance.__file__).resolve().parents[3]]
