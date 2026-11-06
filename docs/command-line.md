# Command line

V-BAR ships no AI of its own and grades the controller it is given. The `vbar` command flies a
built-in controller, or an AI model you connect with `vbar setup`, on one of the scenarios and
prints a plain-language scorecard for each flight.

Code: `vbar.cli` (`main`), installed as the `vbar` command. `python -m vbar` runs the same command
where the scripts folder is not on `PATH`.

```bash
vbar --version
vbar setup                 # connect an AI provider and store a tested key
vbar run --list            # the missions and controllers
vbar run --scenario approach-vbar --controller pd --episodes 2
```

## vbar setup

Asks for the provider, its address when the entry has none, the API key and the model, makes one
test call ("Reply with the word ready", with a reply limit of 64 tokens) and stores the key and
settings only if that call succeeds. [setup.md](setup.md#vbar-setup) describes the steps, where
the files go and the providers on offer.

| Flag | Meaning |
|---|---|
| `--provider KEY` | the catalogue entry, skipping the menu (`vbar setup --list` shows the keys) |
| `--model NAME` | the model name, skipping the question |
| `--base-url URL` | the API root, replacing the entry's own; `https`, or `http` to this computer only |
| `--key-stdin` | read the key from the first line of standard input instead of the hidden prompt |
| `--skip-test` | store without the test call |
| `--show` | print the provider, the model, the settings folder and whether a key is stored |
| `--forget` | delete the stored key; the settings stay |
| `--list` | list the providers, and the models each lists with its price, date and source |

There is no `--key` flag: anything on the command line shows in the process list other users can
read. A terminal that cannot hide typing makes Python's `getpass` warn; `vbar setup` then reads
nothing and points to `--key-stdin`. Whatever goes wrong, the key is never printed, and a failed
test call stores nothing.

## vbar run

Flies `--controller` on `--scenario` from each start and prints one scorecard per flight, then,
for two or more flights, the pass count with its exact 95% range.

| Flag | Default | Meaning |
|---|---|---|
| `--scenario NAME` | | a catalogue name or `name:variant`, as `vbar.scenarios.make` takes |
| `--controller NAME` | | `drift` (no thrust), `pd`, `lqr`, or `llm` for the model set up with `vbar setup` |
| `--seeds S ...` | | starts to repeat |
| `--seeds-file PATH` | | starts to repeat, one per line, kept out of the process list |
| `--episodes N` | 1 | starts drawn at random (128-bit seeds), when no seeds are given |
| `--max-usd X` | 1.00 | dollar cap over all flights, for `llm` |
| `--max-calls N` | flights x steps x (tool rounds + 1) | cap on model calls over all flights, for `llm` |
| `--price IN/OUT` | the listed price | dollars per million input and output tokens, such as `3.00/15.00` |
| `--max-output-tokens N` | 2048 | the reply limit of each model call |
| `--tools` | off | offer the model the relative-motion tools, up to 4 rounds per decision |
| `--json PATH` | | write the seeds, scorecards, usage and metrics |
| `--no-fuel-bound` | | skip working out the least fuel possible from each start |
| `--no-save` | | write no folder under `vbar-runs` |
| `--list` | | list the missions and controllers |

Run on a terminal without `--scenario` or `--controller`, `vbar run` asks for them through
numbered menus (Enter takes the first entry): the mission, then who flies it, the built-in
controllers and, once `vbar setup` has stored one, the AI model, and for the model the dollar cap.
It then prints the equivalent command line, such as `Same as: vbar run --scenario approach-vbar
--controller pd`. Without a terminal the missing flags are refused in one line pointing to
`--list`, as is an unknown scenario or controller.

Before flying, `vbar run` checks that the controller can fly the mission and otherwise refuses in
one line pointing to `vbar run --list`. PD and LQR fly to a hold point, so they fly hold and
berthing missions only, not docking, inspection, departure or capture; the menu offers only the
controllers that can fly the chosen mission. No controller can be scored on a scenario flown with
Euler steps (docking-afrl-paper), since the metrics need a continuous trajectory. An `llm` run with no provider set up, or no
stored key, says to run `vbar setup`, and `--tools` with a model the catalogue lists as taking no
tool calls is refused before any call. Each flight starts with the line `Flying <scenario>, start
<seed>, with <controller>.`, so a flight can be flown again with `--seeds`; the built-in
controllers then print the same scorecard again. Ctrl+C stops the run with one line and exit
status 130; the flight in progress is not scored and no runs folder is written. A `--json`
file, when asked for, holds the flights finished before it.

On a terminal the verdicts and the met and not met statuses are coloured, unless the `NO_COLOR`
environment variable is set and not empty ([no-color.org](https://no-color.org)); the words are
the same either way, and output to a file or pipe has no colour.

### The scorecard

Each flight's card comes from `vbar.scorecard.flight_card` ([evaluation.md](evaluation.md#scorecard)):
the verdict ("Passed" or "Did not pass"), one sentence on why, then the checks arrival, keep-out
zone (when the scenario has one), fuel, cost and anything else seen, each a measured fact and the
rule it was held to. A built-in controller's cost reads "Free". An AI model's cost is the estimate
from the price in use, against the dollar cap, with the number of calls and the replies that could
not be read as a thrust command; the provider's bill is final. The card has no composite score.

### The least fuel possible

After each flight `vbar run` works out the least fuel any controller could use from that start,
with the fuel oracle's `bounds_for` ([oracles.md](oracles.md)), and prints "Working out the least
fuel possible from this start." first. The oracle gives a certified lower bound and the best plan
it found, both as delta-v; the card turns them into propellant mass with the rocket equation,
m = m0 (1 - exp(-dv / (Isp g0))), with the scenario's initial mass and specific impulse, and
shows the fuel used as a range of times the least possible (on a passed flight) or the floor and
best plan (on a failed one). The floor is claimed only where the metrics certify it for the
flight (`optimality_gap`): the lower bound for a flight that kept within the acceleration the
bound assumes, the impulsive lower bound for one that did not, and none for a flight on other
dynamics than the bound's, against a target that moved otherwise than briefed, or over the limit
without an impulsive bound. Without a certified floor the card gives the best plan alone, says no
least possible is certified for the flight and does not compare the fuel. A start takes about 10 s (`time.perf_counter` around `bounds_for` on
the PD controller's approach-vbar flights from seeds 3 and 11: 6.8 s and 10.1 s on a shared
laptop), so before more than 5 new starts the run states the time it expects. Bounds are cached
in `fuel-bounds.json` in the settings folder ([setup.md](setup.md)), keyed by package version,
scenario identifier and seed, so a start is solved once per version. A cache file that cannot be
read, including one nested too deeply to parse, is ignored and the bounds are worked out again.
With `--no-fuel-bound`, or when the oracle finds no usable bound, the card says the fuel is not
compared with the least possible.

### Dollar and call caps

An `llm` run goes through one `BudgetedModel` ([llm-agents.md](llm-agents.md#budgets)) for all its
flights. Before each call it refuses when the spend so far plus the call's worst case (one input
token per byte of the request and the full reply limit) would pass `--max-usd`, or when
`--max-calls` calls have been made. After a call the spend grows by the price of the tokens the
provider reports. A provider that reports more tokens than the worst case can take the spend past
the cap on that one call; the next call is refused.

When a cap is reached, the flight in progress is not scored, no further flight is flown, and the
run ends with a line such as

```text
stopped: The $1.00 cap was reached; the remaining runs were not flown.
```

and exit status 1. A provider failure ends the run the same way, with its sentence from
`vbar.providers.MESSAGES` (wrong key, no credit, rate limit, no connection, unknown model, server
error, certificate, refused request).

The dollar cap needs a price. The catalogue lists one for each model it names, with the date it
was read and its source; for any other model give `--price IN/OUT`. A rate of zero is refused,
since it would make every call free to the budget and switch the cap off, except for the local
server entry, which costs nothing. Without a price, a run with
`--max-calls` and no `--max-usd` is held to the call cap alone and its cost reads "No listed
price". Keep a spending limit on the provider's side as well: the listed price can go stale.

### The JSON record

`--json PATH` writes one object: `scenario`, `scenario_id`, `controller`, `model` (null for a
built-in controller), `seeds` (decimal strings, since 128-bit integers do not survive every JSON
reader), `flights` and `stopped` (the sentence, or null). Each flight holds its `seed` and
`agent_seed`, the `verdict`, the `scorecard` text, the model `usage` (null for a built-in
controller) and every metric, with non-finite values written as `"nan"`, `"inf"` or `"-inf"` as in
`vbar.eval.results.to_json`. The record holds no key and no prompt.

### The runs folder

Unless `--no-save` is given or the run was interrupted, each `vbar run` saves
`vbar-runs/<YYYY-MM-DD-HHMM>-<scenario>/` in the current folder (with `-2`, `-3` and so on added
when that name is taken) and its last line says where. The folder holds `scorecard.txt` (the
printed output without colour), `results.json` (the JSON record above) and `settings.json` (the
package version, scenario, controller, seeds, whether the fuel bound was worked out and, for a
model, the provider, model, address, caps, price, reply limit and tools). The key is never written,
and the address keeps only its scheme, host, port and path, since a user name, query or fragment
can carry a secret.

## Example

```python
from vbar.cli import main

flags = ["--controller", "pd", "--episodes", "1", "--no-fuel-bound", "--no-save"]
status = main(["run", "--scenario", "approach-vbar", *flags])
assert status == 0
```

## Assumptions and limits

- Each flight runs the scenario's full length; there is no step limit on the command line.
- The agent's random stream is drawn for each flight, so `--seeds` repeats a stochastic controller's
  start but not its choices. The built-in controllers are deterministic.
- The listed prices are copied from the providers' pages on the date given and are not checked
  against the bill.
- The fuel bound is the oracle's, worked out on the relative-motion model from the true start; for
  dynamics other than that model it is a model bound, not a certified one
  ([metrics.md](metrics.md)).
- The test call of `vbar setup` and the flights of `vbar run` wait up to 120 s for each answer;
  rate limits and server errors are retried at most twice ([llm-agents.md](llm-agents.md#providers)).

## How it is tested

`tests/test_cli.py`, against a fake provider on 127.0.0.1 (`tests/provider_fakes.py`) and with the
settings folder in a temporary directory: setup from the menu makes exactly one POST with the
64-token limit and stores the key and settings; `--key-stdin` and `--skip-test` work; on POSIX the
key file is 0600 in a 0700 folder; a refused key or an empty balance stores nothing and prints its
sentence; a terminal that cannot hide input is refused naming `--key-stdin`; `--key` is a usage
error that does not repeat the key, and so is `--key-stdin=` with a value; a plain-http address to
another host is refused before the key is asked for; a local server is set up without asking for a
key; `--show` prints four lines, notes `VBAR_LLM_API_KEY`, and changes no file; `--forget` removes
the key; `--list` names every entry. For `vbar
run`: PD on approach-vbar with two drawn starts prints both seeds, and rerunning with `--seeds`
prints the same scorecards; the JSON record is valid and holds the seeds; unknown names give one
line naming `--list`; `--list` shows every registered scenario; each built-in controller on
every registered scenario either flies it or refuses in one line naming `vbar run --list`, and
the menu does not offer PD or LQR for docking; an `llm` run without a key says to
run `vbar setup`; over-reported tokens stop the run at the dollar cap after one call; `--max-calls 3`
stops it after three, and a call cap alone flies a model with no listed price; a model without a
price and no call cap is refused before any call; `--tools` with a model listed without tools is
refused before any call; a refused key stops the run with its sentence and no traceback; Ctrl+C
gives one line and status 130; the model's requests carry the logged prompts and, with navigation
noise, never the true position. With a stand-in oracle: a start is bounded once and then read
from the cache, the cache is per start and package version, a bounded card compares the fuel with
the least possible, `--no-fuel-bound` skips the oracle and the card says not compared, and more
than five new starts are announced with the expected time. The menus run on a terminal and print
the equivalent command; colour appears only on a terminal without `NO_COLOR` and removing it
leaves the plain output; a run saves its folder without the key or the address's query and
fragment, and `--no-save` writes none; a deeply nested bound cache is ignored; a zero price is
refused in one line before any call, except for a local server.
Every output is checked for the key. `tests/test_catalog.py` checks the catalogue's shape, prices,
dates, sources and tools flags, and `connect`, including a model's extra body fields.
