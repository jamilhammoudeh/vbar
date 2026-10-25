import os

from hypothesis import HealthCheck, settings

# Both are measured on the wall clock, so a busy machine would fail examples that did not change.
settings.register_profile(
    "load_independent", deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("load_independent")


def pytest_terminal_summary(terminalreporter):
    # Workflow commands put each failure on the CI run page, so it can be read without the log.
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    stats = terminalreporter.stats
    for report in stats.get("failed", []) + stats.get("error", []):
        path, line, _ = report.location
        details = [s[1:].strip() for s in report.longreprtext.splitlines() if s.startswith("E ")]
        message = f"{report.nodeid}: {' | '.join(details[:4]) or 'failed'}"
        terminalreporter.write_line(
            f"::error file={path},line={(line or 0) + 1}::{message.replace('%', '%25')}"
        )
