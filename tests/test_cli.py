"""`vbar setup` stores a tested key; `vbar run` flies a mission and prints its scorecard."""

import codecs
import getpass
import io
import json
import logging
import os
import re
import stat
import subprocess
import sys
import time
import tomllib
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pytest
from provider_fakes import FAKE_KEY, FakeProvider, Response, chat_error, chat_reply, closed_port_url

from vbar import __version__, metrics, oracles
from vbar import cli as cli_module
from vbar import providers as providers_module
from vbar.agents.llm import Price
from vbar.cli import main
from vbar.metrics.efficiency import FuelBounds
from vbar.oracles import fuel as fuel_module
from vbar.providers import MESSAGES
from vbar.providers import catalog as catalog_module
from vbar.providers.catalog import PROVIDERS, ListedModel, Provider
from vbar.providers.store import (
    config_dir,
    forget_key,
    load_key,
    load_settings,
    save_key,
    save_settings,
)
from vbar.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
# Menu numbers: entries are listed in catalogue order, from 1.
OTHER = list(PROVIDERS).index("other") + 1
READY = Response(200, chat_reply("ready", prompt_tokens=12, completion_tokens=2))
COAST = '{"ax": 0.0, "ay": 0.0, "az": 0.0}'
CONTROLLERS = ("drift", "pd", "lqr", "llm")
# Drawn seeds are 128-bit, so at least 20 digits but for a chance below 1e-18.
SEED = re.compile(r"\b\d{20,}\b")
# The commit hooks' vendor words, rot13-encoded so that this file passes those hooks.
VENDOR_WORDS = [
    codecs.decode(w, "rot13") for w in ("pynhqr", "naguebcvp", "bcranv", "pungtcg", "pbcvybg")
]
VENDORS = re.compile(
    "|".join(VENDOR_WORDS) + r"|(^|[^a-z])" + codecs.decode("phefbe", "rot13") + r"([^a-z]|$)",
    re.I | re.M,
)


@dataclass(frozen=True)
class Ran:
    code: int
    out: str
    err: str

    @property
    def text(self):
        return self.out + self.err

    def lines(self):
        return [line for line in self.text.splitlines() if line.strip()]

    def stopped(self):
        (line,) = [line for line in self.lines() if line.strip().startswith("stopped: ")]
        return line.strip()


class Terminal(io.StringIO):
    """A stream that says it is a terminal."""

    def isatty(self):
        return True


def call(*argv, stdin="", tty=False):
    """Run `vbar argv` in this process with `stdin` as its input; `tty` makes both terminals."""
    stream = Terminal if tty else io.StringIO
    out, err = stream(), io.StringIO()
    saved = sys.stdin
    sys.stdin = stream(stdin)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as stop:
                code = stop.code
    finally:
        sys.stdin = saved
    if isinstance(code, str):
        err.write(code)
        code = 1
    ran = Ran(code or 0, out.getvalue(), err.getvalue())
    assert FAKE_KEY not in ran.text
    return ran


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    folder = tmp_path / "config"
    monkeypatch.setenv("VBAR_CONFIG_DIR", str(folder))
    monkeypatch.delenv("VBAR_LLM_API_KEY", raising=False)
    # Anything a run saves lands here, not in the repository.
    monkeypatch.chdir(tmp_path)

    def no_prompt(prompt="Password: ", stream=None):
        raise AssertionError("the key was asked for")

    monkeypatch.setattr(getpass, "getpass", no_prompt)
    return folder


@pytest.fixture
def typed_key(monkeypatch):
    """Answers the hidden key prompt with the fake key; records each prompt."""
    prompts = []

    def typed(prompt="Password: ", stream=None):
        prompts.append(prompt)
        return FAKE_KEY

    monkeypatch.setattr(getpass, "getpass", typed)
    return prompts


def stored_nothing(folder):
    return not (folder / "credentials.json").exists() and not (folder / "config.json").exists()


def other_flags(url):
    return ("setup", "--provider", "other", "--base-url", url, "--model", "test-model")


def test_setup_from_the_menu_tests_the_key_once_and_stores_it(config, typed_key, caplog):
    with caplog.at_level(logging.DEBUG), FakeProvider([READY]) as fake:
        # Provider number, the address, then the model; the key comes through getpass.
        ran = call("setup", stdin=f"{OTHER}\n{fake.url}\ntest-model\n")
    assert ran.code == 0
    assert len(typed_key) == 1
    (request,) = fake.requests
    assert request.method == "POST"
    body = request.json()
    assert "Reply with the word ready" in json.dumps(body["messages"])
    assert body["max_completion_tokens"] == 64
    assert load_key() == FAKE_KEY
    settings = load_settings()
    assert (settings["provider"], settings["model"]) == ("other", "test-model")
    assert settings["base_url"] == fake.url
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in (config / "config.json").read_text(encoding="utf-8")


def test_setup_reads_the_key_from_stdin_when_asked(config):
    with FakeProvider([READY]) as fake:
        ran = call(*other_flags(fake.url), "--key-stdin", stdin=FAKE_KEY + "\n")
    assert ran.code == 0
    assert len(fake.requests) == 1
    assert load_key() == FAKE_KEY


@POSIX
def test_setup_keeps_the_key_in_a_private_file_and_folder(config):
    with FakeProvider([READY]) as fake:
        call(*other_flags(fake.url), "--key-stdin", stdin=FAKE_KEY + "\n")
    assert stat.S_IMODE(config.stat().st_mode) == 0o700
    assert stat.S_IMODE((config / "credentials.json").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (
            401,
            chat_error("Incorrect API key provided", "invalid_request_error", "invalid_api_key"),
            "key",
        ),
        (429, chat_error("You exceeded your current quota", "insufficient_quota", None), "credit"),
    ],
)
def test_a_failed_test_call_stores_nothing(config, typed_key, status, body, kind):
    with FakeProvider([Response(status, body)]) as fake:
        ran = call(*other_flags(fake.url))
    assert ran.code != 0
    assert len(fake.requests) == 1
    assert MESSAGES[kind] in ran.text
    assert stored_nothing(config)
    assert load_key() is None


def test_echoed_key_input_is_refused_naming_key_stdin(config, monkeypatch):
    def echoing(prompt="Password: ", stream=None):
        warnings.warn("Can not control echo on the terminal.", getpass.GetPassWarning, stacklevel=2)
        return FAKE_KEY

    monkeypatch.setattr(getpass, "getpass", echoing)
    with FakeProvider([READY]) as fake:
        ran = call(*other_flags(fake.url))
    assert ran.code != 0
    assert "--key-stdin" in ran.text
    assert fake.requests == []
    assert stored_nothing(config)


def test_a_key_flag_is_an_argparse_error_that_does_not_echo_the_key(config):
    # `call` also checks the key is not in the error, which argparse would otherwise repeat.
    ran = call("setup", "--key", FAKE_KEY)
    assert ran.code == 2
    assert stored_nothing(config)


def test_skip_test_stores_the_key_without_calling_the_provider(config):
    with FakeProvider([READY]) as fake:
        ran = call(*other_flags(fake.url), "--key-stdin", "--skip-test", stdin=FAKE_KEY + "\n")
    assert ran.code == 0
    assert fake.requests == []
    assert load_key() == FAKE_KEY


def test_a_local_server_is_set_up_without_asking_for_a_key(config):
    with FakeProvider([READY]) as fake:
        ran = call("setup", "--provider", "local", "--base-url", fake.url, "--model", "test-model")
    assert ran.code == 0
    # Only the placeholder key of spec section 11 is sent.
    assert len(fake.requests) == 1
    assert load_settings()["provider"] == "local"
    assert not (config / "credentials.json").exists()


def test_show_prints_provider_model_folder_and_whether_a_key_is_stored(config):
    save_settings(
        {"provider": "other", "model": "test-model", "base_url": "https://api.example.com/v1"}
    )
    save_key(FAKE_KEY)
    ran = call("setup", "--show")
    assert ran.code == 0
    assert len(ran.lines()) == 4
    assert PROVIDERS["other"].label in ran.out
    assert "test-model" in ran.out
    assert str(config_dir()) in ran.out
    assert "key stored: yes" in ran.out
    assert "api.example.com" not in ran.out

    forget_key()
    assert "key stored: no" in call("setup", "--show").out


def test_show_makes_no_call_and_changes_no_file(config):
    save_settings({"provider": "other", "model": "test-model", "base_url": closed_port_url()})
    save_key(FAKE_KEY)
    before = {p.name: p.read_bytes() for p in config.iterdir()}
    call("setup", "--show")
    assert {p.name: p.read_bytes() for p in config.iterdir()} == before


def test_forget_removes_the_stored_key(config):
    save_settings({"provider": "other", "model": "test-model", "base_url": closed_port_url()})
    save_key(FAKE_KEY)
    ran = call("setup", "--forget")
    assert ran.code == 0
    assert load_key() is None
    assert not (config / "credentials.json").exists()


def test_setup_list_names_every_catalogue_entry():
    ran = call("setup", "--list")
    assert ran.code == 0
    for provider in PROVIDERS.values():
        assert provider.label in ran.out


@pytest.fixture(scope="module")
def pd_runs(tmp_path_factory):
    """PD on approach-vbar: two drawn starts, then the same two given back with `--seeds`."""
    folder = tmp_path_factory.mktemp("pd")
    record = folder / "run.json"
    flags = ("run", "--scenario", "approach-vbar", "--controller", "pd", "--json", str(record))
    # A saved folder's name holds the time, so the rerun comparison runs without one.
    flags += ("--no-fuel-bound", "--no-save")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(folder)
        patch.setenv("VBAR_CONFIG_DIR", str(folder / "config"))
        first = call(*flags, "--episodes", "2")
        first_json = record.read_text(encoding="utf-8")
        seeds = list(dict.fromkeys(SEED.findall(first.out)))
        second = call(*flags, "--seeds", *seeds)
    return first, first_json, seeds, second


def test_a_pd_run_prints_a_scorecard_with_the_two_drawn_seeds(pd_runs):
    first, _, seeds, _ = pd_runs
    assert first.code == 0
    assert len(seeds) == 2
    assert "Passed" in first.out or "Did not pass" in first.out


def test_rerunning_with_the_printed_seeds_gives_the_same_scorecard(pd_runs):
    first, _, _, second = pd_runs
    assert second.code == 0
    assert second.out == first.out


def test_the_json_record_is_valid_and_holds_the_seeds(pd_runs):
    _, first_json, seeds, _ = pd_runs
    json.loads(first_json)
    for seed in seeds:
        assert seed in first_json


@pytest.mark.parametrize(
    "flags",
    [
        ("--scenario", "no-such-mission", "--controller", "pd"),
        ("--scenario", "approach-vbar", "--controller", "warp-drive"),
    ],
)
def test_an_unknown_name_is_one_line_pointing_to_list(flags):
    ran = call("run", *flags)
    assert ran.code != 0
    (line,) = ran.lines()
    assert "--list" in line


def test_run_list_shows_every_registered_scenario_and_controller():
    ran = call("run", "--list")
    assert ran.code == 0
    for name in [*SCENARIOS, *CONTROLLERS]:
        assert name in ran.out


@pytest.mark.parametrize("settings", [None, {"provider": "other", "model": "test-model"}])
def test_an_llm_run_without_a_key_says_to_run_setup(settings):
    if settings is not None:
        save_settings({**settings, "base_url": closed_port_url()})
    ran = call("run", "--scenario", "approach-vbar", "--controller", "llm", "--episodes", "1")
    assert ran.code != 0
    assert "vbar setup" in ran.text
    assert "Traceback" not in ran.text


def connect_to(fake):
    save_settings({"provider": "other", "model": "test-model", "base_url": fake.url})
    save_key(FAKE_KEY)


LLM_RUN = ("run", "--scenario", "approach-vbar", "--controller", "llm", "--episodes", "1")


def test_over_reported_tokens_stop_the_run_at_the_dollar_cap(tmp_path):
    # 1e9 tokens each way at $3 and $15 per million is far past the $1.00 default cap.
    huge = chat_reply(COAST, prompt_tokens=10**9, completion_tokens=10**9)
    record = tmp_path / "llm.json"
    with FakeProvider([Response(200, huge)]) as fake:
        connect_to(fake)
        ran = call(*LLM_RUN, "--price", "3.00/15.00", "--json", str(record))
    assert len(fake.requests) == 1
    assert "$1.00" in ran.stopped()
    if record.exists():
        assert FAKE_KEY not in record.read_text(encoding="utf-8")


def test_the_run_stops_at_the_call_cap():
    with FakeProvider([Response(200, chat_reply(COAST))]) as fake:
        connect_to(fake)
        ran = call(*LLM_RUN, "--price", "3.00/15.00", "--max-calls", "3")
    assert len(fake.requests) == 3
    assert "3" in ran.stopped()


def test_a_dollar_cap_on_a_model_without_a_listed_price_needs_price():
    with FakeProvider([Response(200, chat_reply(COAST))]) as fake:
        connect_to(fake)
        ran = call(*LLM_RUN)
    assert ran.code != 0
    assert "--price" in ran.text
    assert fake.requests == []


def test_a_provider_failure_is_shown_as_its_sentence():
    refused = chat_error("Incorrect API key provided", "invalid_request_error", "invalid_api_key")
    with FakeProvider([Response(401, refused)]) as fake:
        connect_to(fake)
        ran = call(*LLM_RUN, "--price", "3.00/15.00")
    assert len(fake.requests) == 1
    assert ran.stopped() == f"stopped: {MESSAGES['key']}"
    assert "Traceback" not in ran.text


def test_python_m_vbar_prints_the_version():
    done = subprocess.run(
        [sys.executable, "-m", "vbar", "--version"], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0
    assert __version__ in done.stdout


def test_the_package_installs_a_vbar_command():
    with open(ROOT / "pyproject.toml", "rb") as file:
        project = tomllib.load(file)["project"]
    assert project["scripts"]["vbar"] == "vbar.cli:main"


def test_vendor_names_appear_only_in_the_provider_catalogue():
    catalogue = ROOT / "src" / "vbar" / "providers" / "catalog.py"
    named = []
    for base in (ROOT / "src", ROOT / "docs", ROOT / "examples"):
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path == catalogue:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if VENDORS.search(text):
                named.append(path.relative_to(ROOT).as_posix())
    assert VENDORS.search((ROOT / "README.md").read_text(encoding="utf-8")) is None
    assert named == []


def test_a_plain_http_address_off_this_computer_is_refused_before_the_key_prompt(config):
    # The autouse fixture fails the test if the key prompt is reached.
    ran = call(*other_flags("http://api.example.com/v1"))
    assert ran.code != 0
    assert stored_nothing(config)


def test_a_bad_address_typed_in_the_menu_is_refused_before_the_key_prompt(config):
    ran = call("setup", stdin=f"{OTHER}\nhttp://api.example.com/v1\ntest-model\n")
    assert ran.code != 0
    assert stored_nothing(config)


def test_key_stdin_with_a_value_is_refused_without_echoing_it(config):
    ran = call(*other_flags(closed_port_url()), f"--key-stdin={FAKE_KEY}")
    assert ran.code == 2
    assert stored_nothing(config)


def test_a_call_cap_alone_flies_a_model_with_no_listed_price():
    with FakeProvider([Response(200, chat_reply(COAST))]) as fake:
        connect_to(fake)
        ran = call(*LLM_RUN, "--max-calls", "2")
    assert len(fake.requests) == 2
    assert "2" in ran.stopped()
    assert "--price" not in ran.text


def test_ctrl_c_during_a_run_prints_one_line_and_exits_130(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(metrics, "record_episode", interrupted)
    ran = call("run", "--scenario", "approach-vbar", "--controller", "pd", "--seeds", "5")
    assert ran.code == 130
    assert "Traceback" not in ran.text
    # Besides the line naming the flight it was about to fly.
    assert len([line for line in ran.lines() if not line.startswith("Flying")]) == 1


def test_show_says_when_the_environment_key_replaces_the_stored_one(config, monkeypatch):
    save_settings({"provider": "other", "model": "test-model", "base_url": closed_port_url()})
    save_key("stored-key-0000")
    monkeypatch.setenv("VBAR_LLM_API_KEY", FAKE_KEY)
    ran = call("setup", "--show")
    assert ran.code == 0
    assert "VBAR_LLM_API_KEY" in ran.out
    assert "stored-key-0000" not in ran.out


def test_the_model_is_sent_the_noisy_navigation_not_the_true_state(monkeypatch):
    # Criterion 17: the requests carry the agent's logged prompts, built from the sensor only.
    scenario = "approach-vbar:nav1"
    seen = {}
    record = metrics.record_episode

    def watched(env, agent, *args, **kwargs):
        seen["agent"] = agent
        seen["truth"] = truth = []
        step, reset = env.step, env.reset

        def reset_watched(*a, **k):
            obs, info = reset(*a, **k)
            truth.append(info["true_state"][:3])
            return obs, info

        def step_watched(*a, **k):
            out = step(*a, **k)
            truth.append(out[-1]["true_state"][:3])
            return out

        monkeypatch.setattr(env, "reset", reset_watched)
        monkeypatch.setattr(env, "step", step_watched)
        return record(env, agent, *args, **kwargs)

    monkeypatch.setattr(metrics, "record_episode", watched)
    with FakeProvider([Response(200, chat_reply(COAST))]) as fake:
        connect_to(fake)
        call("run", "--scenario", scenario, "--controller", "llm", "--max-calls", "4")
    prompts = [entry["prompt"] for entry in seen["agent"].log]
    assert len(fake.requests) == len(prompts) == 4
    for request, prompt in zip(fake.requests, prompts, strict=True):
        assert request.json()["messages"][0]["content"] == prompt
    sent = "".join(r.raw.decode("utf-8") for r in fake.requests)
    assert seen["truth"]
    for x, y, z in seen["truth"]:
        assert f"[{x:.2f}, {y:.2f}, {z:.2f}]" not in sent


def toolless_catalogue(monkeypatch):
    listed = ListedModel(
        "test-model", Price(1.0, 2.0), "2026-11-05", "https://example.com/prices", tools=False
    )
    entry = Provider(
        key="test-entry",
        label="Test entry",
        summary="a chat server for tests",
        wire="chat",
        base_url=None,
        models=(listed,),
    )
    patched = {**PROVIDERS, "test-entry": entry}
    for module in (cli_module, providers_module, catalog_module):
        monkeypatch.setattr(module, "PROVIDERS", patched, raising=False)


def test_tools_with_a_model_listed_without_tools_stop_before_any_call(monkeypatch):
    toolless_catalogue(monkeypatch)
    with FakeProvider([Response(200, chat_reply(COAST))]) as fake:
        save_settings({"provider": "test-entry", "model": "test-model", "base_url": fake.url})
        save_key(FAKE_KEY)
        ran = call(*LLM_RUN, "--tools")
    assert ran.code != 0
    assert fake.requests == []
    assert len(ran.lines()) == 1
    assert "Traceback" not in ran.text


def fuel_blocks(text):
    """The Fuel check of each card in `text`: its heading line and the indented lines below."""
    blocks, block = [], None
    for line in text.splitlines():
        if line.startswith("Fuel:"):
            block = [line]
            blocks.append(block)
        elif block is not None and line.startswith("  "):
            block.append(line)
        else:
            block = None
    return [" ".join(" ".join(b).split()) for b in blocks]


def test_without_a_fuel_bound_the_card_gives_the_fuel_used_and_no_ratio(pd_runs):
    first, _, _, _ = pd_runs
    blocks = fuel_blocks(first.out)
    assert len(blocks) == 2
    for block in blocks:
        assert re.search(r"Used [0-9.]+ k?g", block)
        assert "not compared with the least possible" in block.lower()
        assert "times" not in block


# Spec section 13: fuel bounds, interactive menus, colour and the runs folder.
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PD_RUN = ("run", "--scenario", "approach-vbar", "--controller", "pd")
BOUND_NOTICE = "Working out the least fuel possible from this start"


@pytest.fixture(scope="module")
def pd_episode():
    """One recorded PD flight on approach-vbar, flown once and handed to every patched run."""
    from vbar.agents import PDAgent
    from vbar.scenarios import make

    return metrics.record_episode(make("approach-vbar"), PDAgent(), seed=5, agent_seed=6)


@pytest.fixture
def replayed(monkeypatch, pd_episode):
    """Runs return the recorded PD flight instead of flying it."""
    monkeypatch.setattr(metrics, "record_episode", lambda *args, **kwargs: pd_episode)


@pytest.fixture
def bounds(monkeypatch):
    """A stand-in for the fuel oracle; returns the list of episodes it was asked about."""
    asked = []

    def bounds_for(ep, oracle=None):
        asked.append(ep)
        norm = "l2" if ep.view.thrust_limit == "ball" else "l1"
        return FuelBounds(lower=0.2, upper=0.25, norm=norm, rung="cw")

    for module in (oracles, fuel_module, cli_module):
        monkeypatch.setattr(module, "bounds_for", bounds_for, raising=False)
    return asked


def fuel_block(ran):
    (block,) = fuel_blocks(ANSI.sub("", ran.out))
    return block


def interrupt_flights(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(metrics, "record_episode", interrupted)


def test_a_start_is_bounded_once_and_then_read_from_the_cache(replayed, bounds):
    first = call(*PD_RUN, "--seeds", "5", "--no-save")
    assert first.code == 0
    assert len(bounds) == 1
    assert BOUND_NOTICE in first.out
    second = call(*PD_RUN, "--seeds", "5", "--no-save")
    assert second.code == 0
    assert len(bounds) == 1
    assert fuel_block(second) == fuel_block(first)


def test_a_bounded_card_compares_the_fuel_with_the_least_possible(replayed, bounds):
    block = fuel_block(call(*PD_RUN, "--seeds", "5", "--no-save"))
    assert "not compared" not in block.lower()
    assert "times the least possible" in block or "No controller can use less than" in block


def test_the_bound_cache_is_kept_per_start_and_package_version(replayed, bounds, monkeypatch):
    call(*PD_RUN, "--seeds", "5", "--no-save")
    call(*PD_RUN, "--seeds", "7", "--no-save")
    assert len(bounds) == 2
    for module in (sys.modules["vbar"], cli_module):
        monkeypatch.setattr(module, "__version__", "0.0.0.test", raising=False)
    call(*PD_RUN, "--seeds", "5", "--no-save")
    assert len(bounds) == 3


def test_no_fuel_bound_skips_the_oracle_and_the_card_says_not_compared(replayed, bounds):
    ran = call(*PD_RUN, "--seeds", "5", "--no-save", "--no-fuel-bound")
    assert ran.code == 0
    assert bounds == []
    assert BOUND_NOTICE not in ran.out
    assert "not compared with the least possible" in fuel_block(ran).lower()


@pytest.mark.parametrize(("starts", "warned"), [(5, False), (6, True)])
def test_more_than_five_unbounded_starts_state_the_expected_time_first(
    bounds, monkeypatch, starts, warned
):
    interrupt_flights(monkeypatch)
    seeds = [str(seed) for seed in range(1, starts + 1)]
    ran = call(*PD_RUN, "--seeds", *seeds, "--no-save")
    before = []
    for line in ran.lines():
        if line.startswith("Flying"):
            break
        before.append(line)
    assert any(re.search(r"\d", line) and "about" in line for line in before) == warned


def test_without_flags_a_terminal_run_asks_through_numbered_menus(monkeypatch):
    answers = iter(["approach-vbar", "PD"])

    def menu_choice(prompt=""):
        # The number of the latest menu line naming the wanted entry; Enter for anything else.
        wanted = next(answers, None)
        if wanted is None:
            return ""
        printed = sys.stdout.getvalue()
        return re.findall(rf"^\s*(\d+)\s+.*{re.escape(wanted)}", printed, re.M)[-1]

    monkeypatch.setattr("builtins.input", menu_choice)
    interrupt_flights(monkeypatch)
    ran = call("run", tty=True)
    text = ANSI.sub("", ran.out)
    assert "--scenario approach-vbar" in text
    assert "--controller pd" in text
    assert "Traceback" not in ran.text


def test_without_flags_and_without_a_terminal_a_run_refuses_in_one_line():
    ran = call("run")
    assert ran.code != 0
    (line,) = ran.lines()
    assert "--list" in line


def test_colour_only_on_a_terminal_without_no_color_and_the_words_stay(replayed, monkeypatch):
    flags = (*PD_RUN, "--seeds", "5", "--no-save", "--no-fuel-bound")
    monkeypatch.delenv("NO_COLOR", raising=False)
    coloured = call(*flags, tty=True).out
    plain = call(*flags).out
    monkeypatch.setenv("NO_COLOR", "1")
    asked_plain = call(*flags, tty=True).out
    assert ANSI.search(coloured)
    assert not ANSI.search(plain)
    assert not ANSI.search(asked_plain)
    assert ANSI.sub("", coloured) == plain
    assert asked_plain == plain


FIXED_CLOCK = time.struct_time((2026, 11, 5, 14, 32, 0, 3, 309, -1))


@pytest.fixture
def clock(monkeypatch):
    """`time.localtime()` reads 2026-11-05 14:32; given a time, it converts as before."""
    real = time.localtime
    monkeypatch.setattr(time, "localtime", lambda *args: real(*args) if args else FIXED_CLOCK)


def test_a_run_saves_its_folder_without_the_key(tmp_path, clock):
    with FakeProvider([Response(200, chat_reply(COAST))]) as fake:
        connect_to(fake)
        ran = call(*LLM_RUN, "--max-calls", "2", "--no-fuel-bound")
    folder = tmp_path / "vbar-runs" / "2026-11-05-1432-approach-vbar"
    assert folder.is_dir()
    names = {p.name for p in folder.iterdir()}
    assert {"scorecard.txt", "results.json"} <= names
    assert any("settings" in name for name in names - {"scorecard.txt", "results.json"})
    json.loads((folder / "results.json").read_text(encoding="utf-8"))
    for path in folder.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text(encoding="utf-8")
    assert folder.name in ran.lines()[-1]


def test_no_save_writes_no_folder(tmp_path, replayed, clock):
    ran = call(*PD_RUN, "--seeds", "5", "--no-save", "--no-fuel-bound")
    assert ran.code == 0
    assert not (tmp_path / "vbar-runs").exists()
