# Security policy

V-BAR is a research simulator: it runs locally, opens no network connections and handles no
credentials. The parts that run code you did not write are the LLM agent adapter, which calls a
function you supply, and any agent or plugin you load.

## Supported versions

Security fixes go into the latest release on PyPI.

## Reporting a vulnerability

Email j.hammoudeh@outlook.com with a description and, if possible, steps to reproduce. Please do
not open a public issue for a vulnerability. Once a fix is released, the report is credited in
the changelog unless you prefer otherwise.
