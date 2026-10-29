"""Reads a Rerun recording back into plain Python, for the replay tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Recording:
    """Rows by entity path: `timed` sorted by `sim_time` (ns), `static` merged."""

    def __init__(self, path: Path):
        import rerun.chunk as chunks

        store = chunks.RrdReader(path).stream().collect()
        self.timed: dict[str, list[dict]] = {}
        self.static: dict[str, dict] = {}
        for chunk in store.stream().to_chunks():
            batch = chunk.to_record_batch()
            names = [n for n in batch.schema.names if n not in ("rerun.controls.RowId", "log_time")]
            columns = {}
            for name in names:
                column = batch.column(name)
                if name == "sim_time":
                    column = column.cast("int64")
                columns[name] = column.to_pylist()
            rows = [{n: columns[n][i] for n in names} for i in range(batch.num_rows)]
            if chunk.is_static:
                for row in rows:
                    self.static.setdefault(chunk.entity_path.removeprefix("/"), {}).update(row)
            else:
                self.timed.setdefault(chunk.entity_path.removeprefix("/"), []).extend(rows)
        for rows in self.timed.values():
            rows.sort(key=lambda row: row["sim_time"])

    def column(self, entity: str, name: str) -> np.ndarray:
        """One component of an entity over time, one row per entry, as an array."""
        return np.array([np.asarray(row[name]).reshape(-1) for row in self.timed[entity]])

    def times(self, entity: str) -> np.ndarray:
        """`sim_time` of an entity's rows in seconds."""
        return np.array([row["sim_time"] for row in self.timed[entity]]) * 1e-9
