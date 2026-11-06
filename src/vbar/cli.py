"""The `vbar` command: `vbar setup` stores a tested API key, `vbar run` flies a mission and prints
its scorecard.

The key is read only through a hidden prompt or `--key-stdin`, never from the command line, where
other users can see it in the process list, and it is never printed.
"""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import math
import os
import re
import sys
import textwrap
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from . import __version__
from .agents.llm import BudgetedModel, BudgetExceeded, Price, as_reply
from .conventions import G0
from .metrics.efficiency import FuelBounds
from .providers import MESSAGES, PROVIDERS, Provider, ProviderError, connect
from .providers.http import check_url
from .providers.store import (
    CREDENTIALS,
    SETTINGS,
    config_dir,
    forget_key,
    load_key,
    load_settings,
    save_key,
    save_settings,
)

CONTROLLERS = {
    "drift": "no thrust at all, as a reference",
    "pd": "PD controller toward the hold point, built in, free",
    "lqr": "LQR regulator on the relative-motion model, built in, free",
    "llm": "the AI model set up with `vbar setup`",
}
BUILT_IN = {"drift": "the drift reference", "pd": "the PD controller", "lqr": "the LQR controller"}
READY_PROMPT = "Reply with the word ready"
# Enough for the one-word answer, and small enough that the test call costs a fraction of a cent.
READY_TOKENS = 64
DEFAULT_CAP_USD = 1.00
TOOL_ROUNDS = 4
BOUNDS_CACHE = "fuel-bounds.json"
# Seconds to work out one start's fuel bound, measured on approach-vbar starts (6.8 s and 10.1 s).
BOUND_S = 10.0
# More new starts than this are announced with the time they will take.
MAX_QUIET_BOUNDS = 5
RESET = "\x1b[0m"
COLOURS = {
    "Passed": "\x1b[1;32m",
    "Did not pass": "\x1b[1;31m",
    "met": "\x1b[32m",
    "not met": "\x1b[31m",
}
STATUS = re.compile(r"^([A-Z][\w -]*: )(met|not met)$")
INTRO = (
    "V-BAR has no AI of its own. To fly an AI model, connect a provider you have an account "
    "with. The built-in controllers need no setup."
)


def _ask(prompt: str) -> str:
    """One line typed in answer to `prompt`, without surrounding spaces."""
    try:
        return input(prompt).strip()
    except EOFError:
        raise SystemExit("stopped: no answer was given") from None


def _choose_provider() -> Provider:
    entries = list(PROVIDERS.values())
    width = max(len(p.label) for p in entries)
    print("Provider")
    for number, provider in enumerate(entries, 1):
        print(f"  {number}  {provider.label:<{width}}  {provider.summary}")
    while True:
        answer = _ask(f"Choose 1 to {len(entries)}: ")
        if answer.isdigit() and 1 <= int(answer) <= len(entries):
            print()
            return entries[int(answer) - 1]


def _read_key(key_stdin: bool, host: str) -> str:
    print("API key")
    if key_stdin:
        key = sys.stdin.readline().strip()
    else:
        print(f"  Paste the key here. It is stored on this computer and sent only to {host}.")
        print("  Nothing shows while you paste.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", getpass.GetPassWarning)
            key = getpass.getpass("Key: ").strip()
        if any(issubclass(w.category, getpass.GetPassWarning) for w in caught):
            raise SystemExit(
                "This terminal cannot hide what is typed, so no key was read; pipe the key in "
                "with --key-stdin instead."
            )
    if not key:
        raise SystemExit("No key was given; nothing was saved.")
    print(f"  Received a key ending in {key[-4:]}.")
    print()
    return key


def _choose_model(provider: Provider) -> str:
    print("Model")
    if provider.models:
        default = provider.models[0].name
        print(f"  Press Enter for {default}, or type another name.")
        model = _ask("Model: ") or default
    else:
        print("  Type the model name the server knows.")
        model = _ask("Model: ")
    if not model:
        raise SystemExit("No model name was given; nothing was saved.")
    print()
    return model


def _host(url: str) -> str:
    return urlsplit(url).hostname or url


def _setup_show() -> int:
    settings = load_settings()
    provider = PROVIDERS.get(settings.get("provider", ""))
    stored = (config_dir() / CREDENTIALS).exists()
    key_line = "key stored: yes" if stored else "key stored: no"
    if os.environ.get("VBAR_LLM_API_KEY"):
        key_line += ", and VBAR_LLM_API_KEY is set and used instead"
    print(f"provider: {provider.label if provider else settings.get('provider', 'not set')}")
    print(f"model: {settings.get('model', 'not set')}")
    print(f"folder: {config_dir()}")
    print(key_line)
    return 0


def _setup_list() -> int:
    width = max(len(p.label) for p in PROVIDERS.values())
    for provider in PROVIDERS.values():
        print(f"{provider.key:<8}{provider.label:<{width}}  {provider.summary}")
        for listed in provider.models:
            print(
                f"  {listed.name}: ${listed.price.input_per_mtok:.2f} input and "
                f"${listed.price.output_per_mtok:.2f} output per million tokens, as of "
                f"{listed.as_of} ({listed.source})"
            )
    return 0


def setup(args: argparse.Namespace) -> int:
    """`vbar setup`: choose a provider and model, test the key with one call, then store both."""
    if args.list:
        return _setup_list()
    if args.show:
        return _setup_show()
    if args.forget:
        path = config_dir() / CREDENTIALS
        if path.exists():
            forget_key()
            print(f"Removed the stored key from {config_dir()}.")
        else:
            print("No key was stored.")
        return 0
    if args.provider is None:
        print(textwrap.fill(INTRO, 64))
        print()
        provider = _choose_provider()
    else:
        provider = PROVIDERS[args.provider]
    base_url = args.base_url
    if base_url is None and provider.base_url is None:
        print("Address")
        print("  The API root the server answers at, such as https://host/v1.")
        base_url = _ask("Address: ")
        print()
    url = base_url or provider.base_url
    if not url:
        raise SystemExit("No address was given; nothing was saved.")
    # Refuse the address before the key is typed, not after.
    check_url(url)
    key = _read_key(args.key_stdin, _host(url)) if provider.needs_key else None
    model = args.model or _choose_model(provider)
    complete = connect(provider, model, key, max_output_tokens=READY_TOKENS, base_url=url)
    if not args.skip_test:
        print("Test")
        print(f"  One short request to {model} ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            reply = as_reply(complete([{"role": "user", "content": READY_PROMPT}], []))
        except ProviderError as error:
            print("failed")
            raise SystemExit(f"{error} Nothing was saved.") from None
        print(f"answered in {time.perf_counter() - t0:.1f} s")
        listed = provider.listed(model)
        if listed is not None:
            cost = listed.price.cost(reply.input_tokens, reply.output_tokens)
            print("  It cost less than $0.01." if cost < 0.01 else f"  It cost about ${cost:.2f}.")
        print()
    if key is not None:
        save_key(key)
    settings: dict[str, Any] = {"provider": provider.key, "model": model}
    if base_url is not None:
        settings["base_url"] = base_url
    save_settings(settings)
    print("Saved")
    if key is not None:
        print(f"  Key       {config_dir() / CREDENTIALS}")
    print(f"  Settings  {config_dir() / SETTINGS}")
    print()
    print("Next")
    print("  vbar run --list   the missions and controllers to fly")
    return 0


def _price(text: str) -> Price:
    """`--price IN/OUT`: US dollars per million input and output tokens (an argparse type)."""
    parts = text.split("/")
    try:
        rates = [float(part) for part in parts]
    except ValueError:
        rates = []
    if len(rates) != 2 or not all(math.isfinite(r) and r >= 0.0 for r in rates):
        raise argparse.ArgumentTypeError(
            f"expected dollars per million input/output tokens, such as 3.00/15.00, got {text!r}"
        )
    return Price(rates[0], rates[1])


def _list_run_choices() -> int:
    from .scenarios import SCENARIOS

    print("Missions (--scenario)")
    for name in SCENARIOS:
        print(f"  {name}")
    print()
    print("Controllers (--controller)")
    width = max(len(name) for name in CONTROLLERS)
    for name, summary in CONTROLLERS.items():
        print(f"  {name:<{width}}  {summary}")
    return 0


def _menu(title: str, entries: Sequence[tuple[str, str]]) -> int:
    """The index of the entry chosen from a numbered menu; Enter takes the first."""
    width = max(len(name) for name, _ in entries)
    print(title)
    for number, (name, note) in enumerate(entries, 1):
        print(f"  {number:>2}  {name:<{width}}  {note}".rstrip())
    while True:
        answer = _ask(f"Choose 1 to {len(entries)} [1]: ")
        if not answer:
            answer = "1"
        if answer.isdigit() and 1 <= int(answer) <= len(entries):
            print()
            return int(answer) - 1


def _ask_run(args: argparse.Namespace) -> None:
    """Fill in the mission, the controller and, for a model, the dollar cap from menus."""
    from .scenarios import SCENARIOS

    if args.scenario is None:
        names = list(SCENARIOS)
        args.scenario = names[_menu("Mission", [(name, "") for name in names])]
    if args.controller is None:
        settings = load_settings()
        provider = PROVIDERS.get(settings.get("provider", ""))
        choices = [
            ("pd", "PD controller", "built in, free"),
            ("lqr", "LQR controller", "built in, free"),
            ("drift", "Drift", "no thrust, as a reference"),
        ]
        if provider is not None and settings.get("model"):
            choices.insert(0, ("llm", settings["model"], f"AI model, your {provider.label} key"))
        index = _menu("Flown by", [(name, note) for _, name, note in choices])
        args.controller = choices[index][0]
        if args.controller == "llm" and args.max_usd is None:
            print("Spending cap")
            print("  V-BAR stops before any call that could pass the cap.")
            answer = _ask(f"Cap in US dollars [{DEFAULT_CAP_USD:.2f}]: ")
            try:
                cap = float(answer) if answer else DEFAULT_CAP_USD
            except ValueError:
                cap = math.nan
            if not (math.isfinite(cap) and cap > 0.0):
                raise SystemExit("The cap must be a positive number of dollars.")
            args.max_usd = cap
            print()
    command = f"vbar run --scenario {args.scenario} --controller {args.controller}"
    if args.max_usd is not None:
        command += f" --max-usd {args.max_usd:.2f}"
    print(f"Same as: {command}")
    print()


def _language_model(args: argparse.Namespace, flights: int, max_steps: int):
    """The budgeted model of an `llm` run, its name, price, dollar cap and tool rounds; refuses
    before any call."""
    settings = load_settings()
    provider = PROVIDERS.get(settings.get("provider", ""))
    model = settings.get("model")
    if provider is None or not model:
        raise SystemExit("No AI model is set up; run `vbar setup` first.")
    listed = provider.listed(model)
    if args.tools and listed is not None and not listed.tools:
        raise SystemExit(
            f"{model} takes no tool calls in the request format V-BAR speaks; run without --tools."
        )
    key = load_key() if provider.needs_key else None
    if provider.needs_key and key is None:
        raise SystemExit("No API key is stored; run `vbar setup` and paste one.")
    price = args.price or (listed.price if listed is not None else None)
    max_usd = args.max_usd
    if price is None:
        if max_usd is not None or args.max_calls is None:
            raise SystemExit(
                f"V-BAR has no listed price for {model}, so it cannot hold a dollar cap; give one "
                "with --price 3.00/15.00 (dollars per million input/output tokens) or cap the "
                "calls with --max-calls."
            )
    elif max_usd is None:
        max_usd = DEFAULT_CAP_USD
    # A zero rate makes every call free to the budget, which switches the dollar cap off; only a
    # server on this computer costs nothing.
    if (
        price is not None
        and provider.needs_key
        and not (price.input_per_mtok > 0.0 and price.output_per_mtok > 0.0)
    ):
        raise SystemExit(
            "A price of zero would switch off the dollar cap; give both rates above zero with "
            "--price, such as 3.00/15.00."
        )
    rounds = TOOL_ROUNDS if args.tools else 0
    max_calls = args.max_calls or flights * max_steps * (rounds + 1)
    try:
        complete = connect(
            provider,
            model,
            key,
            max_output_tokens=args.max_output_tokens,
            base_url=settings.get("base_url"),
        )
        budget = BudgetedModel(
            complete,
            price=price,
            max_calls=max_calls,
            max_usd=max_usd,
            max_output_tokens=args.max_output_tokens if max_usd is not None else None,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    return budget, model, price, max_usd, rounds


def _cap_sentence(budget: BudgetedModel, max_usd: float | None) -> str:
    if budget.max_calls is not None and budget.calls >= budget.max_calls:
        return (
            f"The cap of {budget.max_calls} model calls was reached; the remaining runs were not "
            "flown."
        )
    return MESSAGES["cap"].format(x=f"{max_usd:.2f}")


def _bound_key(scenario_id: str, seed: int) -> str:
    """The cache key of a start's fuel bound; a new package version solves it again."""
    return f"{__version__} {scenario_id} {seed}"


def _read_bounds() -> dict[str, FuelBounds]:
    """The cached fuel bounds; a missing or unreadable cache is empty."""
    from .eval.results import from_json

    try:
        entries = json.loads((config_dir() / BOUNDS_CACHE).read_text(encoding="utf-8"))
    # A deeply nested file makes the decoder recurse past the interpreter's limit.
    except (OSError, ValueError, RecursionError):
        return {}
    if not isinstance(entries, dict):
        return {}
    cached = {}
    for key, fields in entries.items():
        try:
            cached[key] = FuelBounds(**from_json(fields))
        except (TypeError, ValueError, RecursionError):
            continue
    return cached


def _write_bounds(cached: Mapping[str, FuelBounds]) -> None:
    from .eval.results import to_json

    folder = config_dir()
    folder.mkdir(parents=True, exist_ok=True)
    entries = {key: to_json(dataclasses.asdict(bounds)) for key, bounds in cached.items()}
    (folder / BOUNDS_CACHE).write_text(json.dumps(entries, indent=1), encoding="utf-8")


def _least_fuel(bounds: FuelBounds | None, episode) -> tuple[float, float] | None:
    """The bound's floor and best plan as propellant mass (kg), or None when it is not usable."""
    if bounds is None or bounds.infeasible:
        return None
    if not (bounds.lower > 0.0 and math.isfinite(bounds.upper)):
        return None
    exhaust = episode.isp * G0
    floor, best = (
        -episode.view.mass_kg * math.expm1(-dv / exhaust) for dv in (bounds.lower, bounds.upper)
    )
    return floor, best


def _paint(text: str, colour: bool) -> str:
    """`text` with its verdicts and met or not met statuses coloured; the words stay."""
    if not colour:
        return text
    lines = []
    for line in text.split("\n"):
        if line in COLOURS:
            line = f"{COLOURS[line]}{line}{RESET}"
        else:
            status = STATUS.match(line)
            if status is not None:
                head, word = status.groups()
                line = f"{head}{COLOURS[word]}{word}{RESET}"
        lines.append(line)
    return "\n".join(lines)


def _save_run(scenario: str, transcript: str, record: Mapping[str, Any], used: Mapping) -> Path:
    """Write the run's folder under `vbar-runs` in the current folder and return it."""
    stamp = time.strftime("%Y-%m-%d-%H%M", time.localtime())
    base = Path("vbar-runs") / f"{stamp}-{re.sub(r'[^A-Za-z0-9._-]', '-', scenario)}"
    folder, copy = base, 2
    while folder.exists():
        folder, copy = base.with_name(f"{base.name}-{copy}"), copy + 1
    folder.mkdir(parents=True)
    (folder / "scorecard.txt").write_text(transcript, encoding="utf-8")
    (folder / "results.json").write_text(
        json.dumps(record, indent=1, allow_nan=False), encoding="utf-8"
    )
    (folder / "settings.json").write_text(json.dumps(used, indent=1), encoding="utf-8")
    return folder


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """`vbar run`: fly the controller from each start and print each flight's scorecard."""
    from . import metrics, oracles
    from .agents import DriftAgent, LLMAgent, LQRAgent, PDAgent
    from .agents.remote import session_seeds
    from .agents.tools import CW_TOOLS
    from .eval.results import to_json
    from .metrics.agent import AgentReport
    from .scenarios import make
    from .scorecard import flight_card, graded_card

    if args.list:
        return _list_run_choices()
    if args.scenario is None or args.controller is None:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise SystemExit(
                "vbar run needs --scenario and --controller; `vbar run --list` shows them."
            )
        _ask_run(args)
    try:
        env = make(args.scenario)
    except (KeyError, ValueError):
        raise SystemExit(
            f"unknown scenario {args.scenario!r}; `vbar run --list` shows the missions."
        ) from None
    if args.controller not in CONTROLLERS:
        raise SystemExit(
            f"unknown controller {args.controller!r}; `vbar run --list` shows the controllers."
        )
    seeds = session_seeds(parser, args)
    budget = None
    price: Price | None = None
    max_usd: float | None = None
    rounds = 0
    if args.controller == "llm":
        budget, who, price, max_usd, rounds = _language_model(args, len(seeds), env.cfg.max_steps)
    else:
        who = BUILT_IN[args.controller]
    # Scenarios without declared metric parameters use the defaults.
    declared = getattr(env, "metric_settings", None)
    settings = declared() if declared else None
    report = AgentReport(
        token_prices=None if price is None else (price.input_per_mtok, price.output_per_mtok)
    )
    colour = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    transcript: list[str] = []

    def say(text: str = "") -> None:
        transcript.append(text)
        print(_paint(text, colour))

    def controller():
        if budget is not None:
            return LLMAgent(
                budget, price=price, tools=CW_TOOLS if args.tools else (), max_tool_rounds=rounds
            )
        return {"drift": DriftAgent, "pd": PDAgent, "lqr": LQRAgent}[args.controller]()

    cached = _read_bounds() if args.fuel_bound else {}
    unbounded = sum(_bound_key(env.scenario_id, seed) not in cached for seed in seeds)
    if args.fuel_bound and unbounded > MAX_QUIET_BOUNDS:
        say(
            f"Working out the least fuel possible takes about {BOUND_S:.0f} s for each new "
            f"start, so about {_minutes(unbounded * BOUND_S)} for these {unbounded}; "
            "--no-fuel-bound skips it."
        )
    flights: list[dict[str, Any]] = []
    flown: list[dict[str, Any]] = []
    stopped = None
    interrupted = False
    try:
        for seed in seeds:
            say(f"Flying {args.scenario}, start {seed}, with {who}.")
            agent = controller()
            episode = metrics.record_episode(env, agent, seed=seed)
            bounds = None
            if args.fuel_bound:
                key = _bound_key(env.scenario_id, seed)
                bounds = cached.get(key)
                if bounds is None:
                    say("Working out the least fuel possible from this start.")
                    try:
                        bounds = oracles.bounds_for(episode)
                    except (ValueError, RuntimeError):
                        say("It could not be worked out, so the fuel is not compared with it.")
                    else:
                        cached[key] = bounds
                        _write_bounds(cached)
            measured = metrics.evaluate(episode, settings, bounds=bounds, report=report)
            usage = agent.usage() if isinstance(agent, LLMAgent) else None
            card = flight_card(
                measured,
                episode.view,
                least_fuel=_least_fuel(bounds, episode),
                calls=None if usage is None else usage["model_calls"],
                cap=max_usd,
                soft=env.soft_constraints,
                decisions=None if usage is None else usage["decisions"],
            )
            say()
            say(card.text())
            flown.append(measured)
            flights.append(
                {
                    "seed": str(seed),
                    "agent_seed": str(episode.agent_seed),
                    "verdict": card.verdict,
                    "scorecard": card.text(),
                    "usage": to_json(usage),
                    "metrics": to_json(measured),
                }
            )
    except BudgetExceeded:
        assert budget is not None
        stopped = _cap_sentence(budget, max_usd)
    except ProviderError as error:
        stopped = str(error)
    except KeyboardInterrupt:
        interrupted = True
        stopped = "interrupted; the flight in progress was not scored and nothing was saved."
    if len(flights) > 1:
        say(graded_card(flown).text().rstrip("\n"))
    if stopped is not None:
        say(f"stopped: {stopped}")
    record = {
        "scenario": args.scenario,
        "scenario_id": env.scenario_id,
        "controller": args.controller,
        "model": who if budget is not None else None,
        "seeds": [str(seed) for seed in seeds],
        "flights": flights,
        "stopped": stopped,
    }
    if args.json is not None:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(record, file, indent=1, allow_nan=False)
    if args.save and not interrupted:
        stored = load_settings() if budget is not None else {}
        used = {
            "version": __version__,
            "scenario": args.scenario,
            "controller": args.controller,
            "seeds": [str(seed) for seed in seeds],
            "fuel_bound": args.fuel_bound,
            "provider": stored.get("provider"),
            "model": stored.get("model"),
            "base_url": _plain_address(stored.get("base_url")),
            "max_usd": max_usd,
            "max_calls": None if budget is None else budget.max_calls,
            "price": None if price is None else [price.input_per_mtok, price.output_per_mtok],
            "max_output_tokens": args.max_output_tokens if budget is not None else None,
            "tools": bool(args.tools) if budget is not None else None,
        }
        folder = _save_run(args.scenario, "\n".join(transcript) + "\n", record, used)
        print(f"Saved in {folder}")
    if interrupted:
        return 130
    return 1 if stopped is not None else 0


def _plain_address(url: str | None) -> str | None:
    """`url` as scheme, host, port and path: a user name, query or fragment can hold a secret."""
    if not url:
        return url
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _minutes(seconds: float) -> str:
    if seconds < 120.0:
        return f"{seconds:.0f} s"
    return f"{seconds / 60.0:.0f} min"


def _parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    from .agents.remote import add_seed_options

    parser = argparse.ArgumentParser(
        prog="vbar", description="Fly and grade spacecraft controllers.", allow_abbrev=False
    )
    parser.add_argument("--version", action="version", version=f"vbar {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    setup_parser = commands.add_parser(
        "setup", help="connect an AI provider and store a tested key", allow_abbrev=False
    )
    setup_parser.add_argument("--provider", choices=list(PROVIDERS), help="catalogue entry")
    setup_parser.add_argument("--model", help="model name the provider knows")
    setup_parser.add_argument("--base-url", help="API root, replacing the entry's own")
    setup_parser.add_argument(
        "--key-stdin", action="store_true", help="read the key from the first line of stdin"
    )
    setup_parser.add_argument(
        "--skip-test", action="store_true", help="store without the one test call"
    )
    only = setup_parser.add_mutually_exclusive_group()
    only.add_argument("--show", action="store_true", help="show the stored settings")
    only.add_argument("--forget", action="store_true", help="delete the stored key")
    only.add_argument("--list", action="store_true", help="list the providers and their models")

    run_parser = commands.add_parser(
        "run", help="fly a mission and print its scorecard", allow_abbrev=False
    )
    run_parser.add_argument("--scenario", help="mission name, see --list")
    run_parser.add_argument("--controller", help="drift, pd, lqr or llm")
    add_seed_options(run_parser)
    run_parser.add_argument(
        "--max-usd", type=float, help=f"dollar cap over all runs (default {DEFAULT_CAP_USD:.2f})"
    )
    run_parser.add_argument("--max-calls", type=int, help="cap on model calls over all runs")
    run_parser.add_argument(
        "--price", type=_price, help="IN/OUT dollars per million tokens, for an unlisted model"
    )
    run_parser.add_argument(
        "--max-output-tokens", type=int, default=2048, help="reply limit of each model call"
    )
    run_parser.add_argument(
        "--tools", action="store_true", help="offer the model the relative-motion tools"
    )
    run_parser.add_argument("--json", help="write the seeds, scorecards and metrics here")
    run_parser.add_argument(
        "--no-fuel-bound",
        dest="fuel_bound",
        action="store_false",
        help="skip working out the least fuel possible from each start",
    )
    run_parser.add_argument(
        "--no-save", dest="save", action="store_false", help="write no folder under vbar-runs"
    )
    run_parser.add_argument("--list", action="store_true", help="list the missions and controllers")
    return parser, {"setup": setup_parser, "run": run_parser}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `vbar` command.

    Parameters
    ----------
    argv : sequence of str, optional
        The arguments after `vbar`; `sys.argv[1:]` by default.

    Returns
    -------
    int
        The exit status: 0 on success, 1 when a run stopped early, 130 when it was interrupted
        with Ctrl+C. Refusals exit through
        `SystemExit` with a one-line message.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser, commands = _parser()
    # argparse quotes the value of a flag that takes none, and a key typed there would show.
    if any(a.startswith("--key-stdin=") for a in argv):
        commands["setup"].error("--key-stdin takes no value; pipe the key in on stdin")
    args, extras = parser.parse_known_args(argv)
    if extras:
        # Only flag names are repeated: a stray value could be a key.
        names = sorted({e.split("=", 1)[0] for e in extras if e.startswith("-")})
        hint = (
            "; the key is never taken on the command line, use the prompt or --key-stdin"
            if "--key" in names
            else ""
        )
        found = ", ".join(names) if names else f"{len(extras)} extra values"
        commands[args.command].error(f"unrecognized arguments: {found}{hint}")
    if args.command == "setup":
        try:
            return setup(args)
        except ValueError as error:
            raise SystemExit(f"{error}; nothing was saved.") from None
        except KeyboardInterrupt:
            raise SystemExit("setup stopped; nothing was saved.") from None
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be a positive integer")
    if args.max_calls is not None and args.max_calls < 1:
        parser.error("--max-calls must be a positive integer")
    if args.max_usd is not None and not (math.isfinite(args.max_usd) and args.max_usd > 0.0):
        parser.error("--max-usd must be a positive number of dollars")
    try:
        return run(commands["run"], args)
    except KeyboardInterrupt:
        print("stopped: interrupted; nothing was flown.")
        return 130
