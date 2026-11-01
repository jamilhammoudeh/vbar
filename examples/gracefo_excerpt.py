"""Write the GRACE-FO orbit excerpt installed with vbar.flightdata.

Takes the folder holding the extracted files GNV1B_2019-12-03_C_04.txt, GNV1B_2019-12-03_D_04.txt
and KBR1B_2019-12-03_Y_04.txt from the PO.DAAC daily bundle gracefo_1B_2019-12-03_RL04.ascii.noLRI
(collection GRACEFO_L1B_ASCII_GRAV_JPL_RL04, doi:10.5067/GFL1B-ASJ04) and writes every 120th
second of the first 16 hours: both satellites' Earth-fixed states and the K-band range with its
light-time and antenna-offset corrections.

    python examples/gracefo_excerpt.py <folder> src/vbar/flightdata/gracefo_2019-12-03.csv
"""

import sys
from pathlib import Path

import numpy as np

from vbar.flightdata import read_level1b, write_orbit_excerpt

START = 628603200.0  # 2019-12-03 00:00:00 GPS, in GPS seconds past 2000-01-01 12:00:00 GPS
SPAN = 16 * 3600.0
STEP = 120.0


def states(gnv: dict[str, np.ndarray], t: np.ndarray) -> np.ndarray:
    index = np.searchsorted(gnv["gps_time"], t)
    if not np.array_equal(gnv["gps_time"][index], t):
        raise ValueError("an excerpt time is missing from the GNV1B file")
    keys = ("xpos", "ypos", "zpos", "xvel", "yvel", "zvel")
    return np.stack([gnv[key][index] for key in keys], axis=1)


def main(folder: str, out: str) -> None:
    root = Path(folder)
    target = read_level1b(root / "GNV1B_2019-12-03_C_04.txt")
    chaser = read_level1b(root / "GNV1B_2019-12-03_D_04.txt")
    kbr = read_level1b(root / "KBR1B_2019-12-03_Y_04.txt")
    t = START + np.arange(0.0, SPAN + STEP / 2, STEP)
    index = np.searchsorted(kbr["gps_time"], t)
    if not np.array_equal(kbr["gps_time"][index], t):
        raise ValueError("an excerpt time is missing from the KBR1B file")
    corrected = (
        kbr["biased_range"][index] + kbr["lighttime_corr"][index] + kbr["ant_centr_corr"][index]
    )
    header = [
        "GRACE-FO Level-1B, 2019-12-03, every 120 s of the first 16 h.",
        "Source: GRACE-FO (2019), GRACEFO_L1B_ASCII_GRAV_JPL_RL04, ver. 4, PO.DAAC, CA, USA,",
        "doi:10.5067/GFL1B-ASJ04 (product id in the file headers: 10.5067/GFJPL-L1B04).",
        "Licence: CC0 (NASA Earthdata data use policy, NASA-led missions); no endorsement implied.",
        "Files: GNV1B_2019-12-03_C_04 (target), GNV1B_2019-12-03_D_04 (chaser),",
        "KBR1B_2019-12-03_Y_04.",
        "t: GPS seconds past 2000-01-01 12:00:00 GPS. States: Earth-fixed (ITRF2014), m and m/s.",
        "range: KBR biased range plus light-time and antenna-offset corrections, m; its constant",
        "bias is unknown, so only its changes are meaningful.",
    ]
    write_orbit_excerpt(out, t, states(target, t), states(chaser, t), corrected, header)


if __name__ == "__main__":
    main(*sys.argv[1:])
