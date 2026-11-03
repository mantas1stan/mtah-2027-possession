"""Load the tactical ATD into dense numpy arrays, in metres, indexed by video frame.

Parsing 195 MB of text takes a while, so the arrays are cached as .npz next to the data (which is
gitignored and never leaves the machine).
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require  # noqa: E402

# Params
N_SLOTS = 46
HOME, AWAY = 0, 1
CACHE_VERSION = 1


def _meta(meta_path):
    """Team of each slot, position type of each slot, and the home attack direction (1st half)."""
    root = ET.parse(meta_path).getroot()
    strip = lambda el: el.tag.split("}")[-1]  # noqa: E731
    team_ids = [t.get("id") for t in root.iter() if strip(t) == "Team"]
    players = [p for p in root.iter() if strip(p) == "Player"]
    team = np.array([HOME if p.get("teamId") == team_ids[0] else AWAY for p in players], np.int8)

    def params(el):
        out = {}
        for prm in el.iter():
            if strip(prm) == "ProviderParameter":
                name = value = None
                for child in prm:
                    if strip(child) == "Name":
                        name = child.text
                    elif strip(child) == "Value":
                        value = child.text
                out[name] = value
        return out

    position_type = np.array([params(p).get("position_type") or "" for p in players])
    team_params = [params(t) for t in root.iter() if strip(t) == "Team"]
    return team, position_type, team_params


def _parse_raw(raw_path):
    with open(raw_path) as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN", "nan"]).values
    del text
    frame_id = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 3 * N_SLOTS].reshape(-1, N_SLOTS, 3)
    ball = table[:, 1 + 3 * N_SLOTS:1 + 3 * N_SLOTS + 3]
    return frame_id, players, ball


def load(game=GAME, data_dir=None):
    """dict of arrays, all indexed by video frame:

    px, py, ps   (n, 46) player slot position in metres and speed in m/s, NaN if absent
    bx, by       (n,)    ball position in metres, NaN if absent
    team         (46,)   0 home / 1 away for each slot
    ptype        (46,)   provider position type string for each slot
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache = os.path.join(data_dir or DATA_DIR, "cache", "{}_atd_v{}.npz".format(game, CACHE_VERSION))
    if os.path.isfile(cache):
        return dict(np.load(cache, allow_pickle=False))

    length, width = GAMES[game]["pitch"]
    frame_id, players, ball = _parse_raw(raw_path)
    n = GAMES[game]["n_video_frames"]
    if len(frame_id) != n or not np.array_equal(frame_id, np.arange(1, n + 1)):
        raise ValueError("ATD is not one record per video frame numbered from 1")

    team, ptype, _ = _meta(meta_path)
    out = {
        "px": (players[:, :, 0] * length).astype(np.float32),
        "py": (players[:, :, 1] * width).astype(np.float32),
        "ps": players[:, :, 2].astype(np.float32),
        "bx": (ball[:, 0] * length).astype(np.float64),
        "by": (ball[:, 1] * width).astype(np.float64),
        "team": team,
        "ptype": ptype,
    }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    d = load()
    print({k: v.shape for k, v in d.items()})
    print("slots per team:", np.bincount(d["team"]))
    print("position types:", dict(zip(*np.unique(d["ptype"], return_counts=True))))
