# Setup

V-BAR ships no AI of its own and grades the controller it is given. To fly a language model, it
needs the provider's API key and the model's name; this page covers where they are kept.

The key store and `vbar setup` come in the next release; they are not in the 0.7.0 release on
PyPI. `pip install git+https://github.com/jamilhammoudeh/vbar` installs the development version,
which has them.

Code: `vbar.providers.store` (`save_key`, `load_key`, `forget_key`, `save_settings`,
`load_settings`, `config_dir`), `vbar.providers.catalog` (`Provider`, `ListedModel`, `PROVIDERS`),
`vbar.providers.connect` and the `vbar setup` command (`vbar.cli`).

## vbar setup

`vbar setup` asks four things, makes one test call and stores the answers:

1. **Provider**, from a numbered menu in the order of `vbar.providers.catalog.PROVIDERS`: first
   the providers listed by `vbar setup --list`, each with its models and their prices, then
   "Local server", a chat-format server on this computer at `http://127.0.0.1:11434/v1`, which is
   sent a placeholder key instead of yours, and last "Other address", any server that takes
   chat-format requests.
2. **Address**, only for an entry without one, such as "Other address": the API root, such as
   `https://host/v1`. Plain `http` is allowed only to this computer.
3. **API key**, through a hidden prompt (`getpass`), or from the first line of standard input with
   `--key-stdin`. Only its last four characters are shown back, and none for a key of 8
   characters or fewer. An entry that takes no key skips
   this step.
4. **Model**: Enter takes the entry's first listed model; an entry without a list needs the name.

The test call sends "Reply with the word ready" with a reply limit of 64 tokens and prints how long
the answer took (and, for a listed model, what it cost). If it fails, the run stops with the
failure's sentence and nothing is written. If it succeeds, the key goes to `credentials.json` and
the provider key, model and any address given to `config.json`. `--skip-test` stores without the
call, `--show` prints the provider, model, folder and whether a key is stored (never the key or
the address), `--forget` deletes the key and `--list` prints the catalogue.
[command-line.md](command-line.md#vbar-setup) lists every flag.

```bash
vbar setup
vbar setup --provider other --base-url https://host/v1 --model my-model --key-stdin < key.txt
vbar setup --show
```

Each catalogue entry is a frozen `Provider`: `key` (the name stored and given to `--provider`),
`label` and `summary` for the menu, `wire` (the request format: `chat`, `messages` or `generate`),
`base_url` (None when the user gives it), `auth_header` (the header carrying the bare key, which
the generate format needs; None sends `Authorization: Bearer`), `extra_headers` (not in the
generate format), `extra_body` (chat format only), `needs_key` (an entry without a key is sent
`PLACEHOLDER_KEY`) and `models`. Each listed model is a `ListedModel` with its `name`, `price` (US
dollars per million input and output tokens), `as_of` (the ISO date the price was read), `source`
(the https page it was read from), `tools` (whether it takes tool calls in its entry's request
format; `vbar run --tools` refuses a model that does not) and `extra_body` (fields sent with every
request to that model, chat format only). `Provider` refuses an entry that breaks these rules.
Where a provider charges less for short prompts or until a date, the catalogue lists the higher
price, so a dollar cap stops early rather than late. Every listed price was read on 2026-11-05 and
can go stale; the provider's bill is final. `connect(provider, model, key, *,
max_output_tokens, base_url=None)` returns the model function for an entry, in the adapter of its
wire format, and raises `ValueError` before any connection for a missing address or key or a
refused address.

```python
from vbar.providers import PROVIDERS, connect

model = connect(
    PROVIDERS["other"], "my-model", "my-key", max_output_tokens=64, base_url="https://host/v1"
)
assert "my-key" not in repr(model)
```

## Where the key is stored

The key goes in `credentials.json` and the provider settings (provider, model, address) in
`config.json`, both in one folder per user:

| System | Folder |
|---|---|
| Windows | `%APPDATA%\vbar`, usually `C:\Users\<name>\AppData\Roaming\vbar` |
| macOS, Linux | `$XDG_CONFIG_HOME/vbar` if that variable holds an absolute path, else `~/.config/vbar` |

`VBAR_CONFIG_DIR` replaces the folder, for tests or a second set of settings. `config.json`
never holds the key, so it can be shown or shared.

Permissions:

- On macOS and Linux the folder is set to 0700 and the files are written as 0600 (owner read and
  write only). Each file is written to a temporary file in the same folder, opened with mode 0600,
  and moved over the old one, so a half-written or briefly readable key file never exists.
- When V-BAR reads a key file that other users can read, it sets it to 0600 and warns through
  `warnings`.
- On Windows `os.chmod` only sets the read-only flag, so the files rely on the access list of the
  user's profile folder. That this access list keeps other standard users out by default has not
  been verified.

`VBAR_LLM_API_KEY`, when set and not empty, is used instead of the stored key, and nothing is
written. It suits CI jobs and shared machines. An empty value is ignored.

`save_key` removes surrounding whitespace and refuses an empty key or one holding a control
character such as a line break, which would split the HTTP header the key is sent in. It also
refuses any character outside ASCII, since look-alike and invisible characters come along with a
key copied from a web page; the error never quotes the key. `forget_key` deletes
`credentials.json` and keeps the settings.

```python
from vbar.providers.store import config_dir

print(config_dir())  # the folder holding credentials.json and config.json
```

### Threat model

The file protects the key from other users of the same computer and from being committed to a
repository by accident, since it lives outside any project folder. It does not protect the key
from programs running as the same user, which can read the file. Keep a spending limit on the
provider's side as well.

The key never appears in V-BAR's output: errors carry a fixed sentence per kind (`key`, `credit`,
`rate`, `network`, `model`, `server`, `tls` and `refused`) instead of the provider's text, log
records and `repr` of the adapters omit it, and recorded transcripts hold only request digests and
replies ([llm-agents.md](llm-agents.md#providers) lists the kinds).

## Certificate errors

The sentence "Python could not verify the provider's certificate" means the TLS check of the
provider's address failed. V-BAR never turns the check off. Common causes:

- Python from the python.org installer on macOS uses its own certificate store, which starts
  empty. Run `Install Certificates.command` once, from the `Python 3.x` folder in Applications.
- A company network that inspects encrypted traffic presents its own certificate. Ask for that
  certificate as a PEM file and point OpenSSL at a bundle holding it with the `SSL_CERT_FILE`
  environment variable, which Python's default TLS context reads.
- The computer's clock is far off, so a valid certificate looks expired or not yet valid.

## How it is tested

`tests/test_provider_store.py`, with `VBAR_CONFIG_DIR` pointed at a temporary folder: a saved key
loads back and replaces the previous one; forgetting removes it and keeps the settings; the key
never reaches `config.json`, log records or warnings; `VBAR_LLM_API_KEY` wins over the stored key
and an empty one does not; empty keys and keys with line breaks are refused; no temporary file is
left behind; on POSIX the file is 0600 in a 0700 folder and a 0644 file is tightened with a
warning; the default folder follows `APPDATA`, `XDG_CONFIG_HOME` or `~/.config`; the module imports
only the standard library. `tests/test_environment_variables.py` checks that these are the only
new environment variables the package reads.
`tests/test_cli.py` runs `vbar setup` against a fake provider on 127.0.0.1: one test call, the
files written only after it succeeds, the key never in any output, `--show`, `--forget` and
`--list`. `tests/test_catalog.py` checks every entry's shape and dated prices and that `connect`
sends the key and reply limit, sends no key to a local server and refuses plain http to another
host.
