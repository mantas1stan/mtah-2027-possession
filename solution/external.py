"""External training/validation data: Metrica Sports' open sample games 1 and 2.

    https://github.com/metrica-sports/sample-data   (anonymised; credit Metrica Sports)

They are other matches, so labels derived from them may be used to tune parameters (RULES.md and
the organisers' clarification). Download them into ../external-data/metrica-sample, or point
MTAH_EXTERNAL_DIR at the folder that holds Sample_Game_1/ and Sample_Game_2/.

Two things are produced per game, both in the shape the competition uses:

  tracking  the same dict solution/atd.py returns (metres, NaN where absent, speeds in m/s, a team
            per slot and no player identity), so possession.py runs on it unchanged;
  labels    H/A/D per frame, derived mechanically from the event data the way the competition's
            ground truth is derived from corrected tracking -- see derive_labels().
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
EXTERNAL_DIR = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
    os.path.dirname(ROOT), "external-data", "metrica-sample")
PITCH = (105.0, 68.0)  # the sample games' pitch, per their README
FPS = 25
SPEED_SMOOTH = 5  # +/- frames when differentiating positions into speeds
HOME, AWAY = 0, 1
GAMES = (1, 2)

# Events whose Team is the side in possession at that moment. CHALLENGE is left out because both
# sides log one; BALL LOST and BALL OUT are logged by the side that had the ball, which keeps the
# label unchanged -- possession only moves when the other side records a controlled action.
POSSESSION_TYPES = ("PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT")
# Events after which play stops until the next SET PIECE.
STOP_TYPES = ("BALL OUT", "FAULT RECEIVED")
STOP_SUBTYPE_WORDS = ("FAULT", "OFFSIDE", "HAND BALL", "-GOAL", "-OUT", "END HALF")


def _game_dir(game):
    return os.path.join(EXTERNAL_DIR, "Sample_Game_{}".format(game))


def _read_team_csv(path):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    player_cols = [c for c in cols if c.startswith("Player")]
    xy = np.stack([raw[[c, cols[cols.index(c) + 1]]].values for c in player_cols], axis=1)
    ball = raw[["Ball", cols[cols.index("Ball") + 1]]].values
    return raw["Period"].values, raw["Frame"].values, xy, ball


def _speed(x, y):
    """m/s from positions in metres, centred differences over +/- SPEED_SMOOTH frames."""
    k = SPEED_SMOOTH
    dx = np.full_like(x, np.nan)
    dy = np.full_like(y, np.nan)
    dx[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2 * k)
    dy[k:-k] = (y[2 * k:] - y[:-2 * k]) / (2 * k)
    return np.hypot(dx, dy) * FPS


def load_tracking(game):
    """dict in solution/atd.py's format, plus 'period' per frame. Index = Frame - 1."""
    folder = _game_dir(game)
    stem = os.path.join(folder, "Sample_Game_{}_RawTrackingData_".format(game))
    period, frame, home_xy, ball = _read_team_csv(stem + "Home_Team.csv")
    _, frame_a, away_xy, _ = _read_team_csv(stem + "Away_Team.csv")
    if not np.array_equal(frame, frame_a) or not np.array_equal(frame, np.arange(1, len(frame) + 1)):
        raise ValueError("game {}: home and away tracking are not frame-aligned".format(game))

    length, width = PITCH
    xy = np.concatenate([home_xy, away_xy], axis=1)
    px, py = xy[:, :, 0] * length, xy[:, :, 1] * width
    team = np.array([HOME] * home_xy.shape[1] + [AWAY] * away_xy.shape[1], np.int8)
    return {
        "px": px.astype(np.float32),
        "py": py.astype(np.float32),
        "ps": np.stack([_speed(px[:, i], py[:, i]) for i in range(px.shape[1])],
                       axis=1).astype(np.float32),
        "bx": ball[:, 0] * length,
        "by": ball[:, 1] * width,
        "team": team,
        "ptype": np.array([""] * len(team)),
        "period": period,
    }


def half_blocks(trk):
    """[(period, first_frame, last_frame)] as 0-based indices, like games.half_video_blocks."""
    out = []
    for p in np.unique(trk["period"]):
        idx = np.flatnonzero(trk["period"] == p)
        out.append((int(p), int(idx[0]), int(idx[-1])))
    return out


def _is_stop(row):
    if row["Type"] in STOP_TYPES:
        return True
    sub = str(row["Subtype"]) if isinstance(row["Subtype"], str) else ""
    if row["Type"] in ("CHALLENGE", "BALL LOST", "SHOT"):
        return any(w in sub for w in STOP_SUBTYPE_WORDS) and "ADVANTAGE" not in sub
    return False


def derive_labels(game, n_frames, blocks):
    """H/A/D/X per frame from the event data.

    Alive frames take the team of the most recent possession event. Play is dead from a stoppage
    event (ball out, foul, offside, goal, end of half) until the next set piece. Frames outside
    the two periods, and before each period's first event, are X.
    """
    ev = pd.read_csv(os.path.join(_game_dir(game), "Sample_Game_{}_RawEventsData.csv".format(game)))
    ev = ev.sort_values(["Start Frame"], kind="stable").reset_index(drop=True)
    labels = np.full(n_frames, "X", dtype="<U1")

    for period, first, last in blocks:
        rows = ev[ev["Period"] == period]
        state, team = "X", None
        cursor = first
        for _, row in rows.iterrows():
            start = int(row["Start Frame"]) - 1
            end = max(start, int(row["End Frame"]) - 1)
            start = min(max(start, first), last + 1)
            labels[cursor:start] = state if state != "A_or_H" else team
            cursor = start
            side = "H" if row["Team"] == "Home" else "A"
            if row["Type"] == "SET PIECE":
                state, team = "A_or_H", side
            elif _is_stop(row):
                # dead from the moment the ball stops being playable: the end of a ball-out
                # flight, else the event itself
                stop = min(max((end if row["Type"] in ("BALL OUT", "SHOT") else start), cursor),
                           last + 1)
                if state == "A_or_H":
                    labels[cursor:stop] = team
                cursor = stop
                state = "D"
            elif row["Type"] in POSSESSION_TYPES:
                if row["Type"] in ("BALL LOST", "BALL OUT"):
                    if state == "D":
                        continue
                else:
                    # a pass or recovery while "dead" means the restart was not logged as a
                    # set piece (a quick free kick): play is alive again from this event
                    team = side
                state = "A_or_H"
        labels[cursor:last + 1] = team if state == "A_or_H" else state
    return labels


def load_game(game):
    trk = load_tracking(game)
    blocks = half_blocks(trk)
    return trk, blocks, derive_labels(game, len(trk["bx"]), blocks)


if __name__ == "__main__":
    for g in GAMES:
        trk, blocks, lab = load_game(g)
        graded = lab[lab != "X"]
        b = np.flatnonzero(graded[1:] != graded[:-1]) + 1
        segs = np.split(graded, b)
        spells = [len(s) for s in segs if s[0] in "HA"]
        print("game {}: frames {}  H {:.1%} A {:.1%} D {:.1%}  possessions {} mean {:.1f} s  "
              "dead stretches {}  ball missing {:.1%}".format(
                  g, len(lab), np.mean(graded == "H"), np.mean(graded == "A"),
                  np.mean(graded == "D"), len(spells), np.mean(spells) / FPS,
                  sum(s[0] == "D" for s in segs), np.mean(np.isnan(trk["bx"]))))
