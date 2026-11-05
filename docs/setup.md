# Setup

V-BAR ships no AI of its own and grades the controller it is given. To fly a language model, it
needs the provider's API key and the model's name; this page covers where they are kept.

Code: `vbar.providers.store` (`save_key`, `load_key`, `forget_key`, `save_settings`,
`load_settings`, `config_dir`).

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

## How it is tested

`tests/test_provider_store.py`, with `VBAR_CONFIG_DIR` pointed at a temporary folder: a saved key
loads back and replaces the previous one; forgetting removes it and keeps the settings; the key
never reaches `config.json`, log records or warnings; `VBAR_LLM_API_KEY` wins over the stored key
and an empty one does not; empty keys and keys with line breaks are refused; no temporary file is
left behind; on POSIX the file is 0600 in a 0700 folder and a 0644 file is tightened with a
warning; the default folder follows `APPDATA`, `XDG_CONFIG_HOME` or `~/.config`; the module imports
only the standard library. `tests/test_environment_variables.py` checks that these are the only
new environment variables the package reads.
