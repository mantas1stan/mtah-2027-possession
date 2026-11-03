"""External training/validation data: Metrica Sports' open sample games 1, 2 and 3.

    https://github.com/metrica-sports/sample-data   (anonymised; credit Metrica Sports)

They are other matches, so labels derived from them may be used to tune parameters (RULES.md and
the organisers' clarification). Download them into ../external-data/metrica-sample, or point
MTAH_EXTERNAL_DIR at the folder that holds Sample_Game_1/ .. Sample_Game_3/.

Games 1 and 2 (CSV) are the tuning set. Game 3 (FIFA/EPTS tracking, like the competition ATD, and
JSON events) is held out: it is only scored after tuning, to estimate performance on an unseen
match.

Two things are produced per game, both in the shape the competition uses:

  tracking  the same dict solution/atd.py returns (metres, NaN where absent, speeds in m/s, a team
            per slot and no player identity), so possession.py runs on it unchanged;
  labels    H/A/D per frame, derived mechanically from the event data the way the competition's
            ground truth is derived from corrected tracking -- see derive_labels().
"""

import json
import os
import xml.etree.ElementTree as ET

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
GAMES = (1, 2)       # tuning
HOLDOUT_GAMES = (3,)  # never tuned on

# Events whose Team is the side in possession at that moment. CHALLENGE is left out because both
# sides log one; BALL LOST and BALL OUT are logged by the side that had the ball, which keeps the
# label unchanged -- possession only moves when the other side records a controlled action.
POSSESSION_TYPES = ("PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT")
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


def _tag(el):
    return el.tag.split("}")[-1]


def _load_tracking_epts(game):
    """Game 3: FIFA/EPTS text + metadata, x/y only, with the channel layout changing at each
    substitution. Each layout's players are packed into fixed per-team slots (identity is not
    kept, as in the competition ATD)."""
    folder = _game_dir(game)
    root = ET.parse(os.path.join(folder, "Sample_Game_{}_metadata.xml".format(game))).getroot()
    team_ids = [t.get("id") for t in root.iter() if _tag(t) == "Team"]
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter() if _tag(p) == "Player"}
    player_of_channel = {c.get("id"): c.get("playerId")
                         for c in root.iter() if _tag(c) == "PlayerChannel"}
    params = {}
    for prm in root.iter():
        if _tag(prm) == "ProviderParameter":
            kids = {_tag(k): k.text for k in prm}
            params[kids.get("Name")] = kids.get("Value")

    specs = []  # (first_frame, last_frame, [team index of each group])
    for spec in root.iter():
        if _tag(spec) != "DataFormatSpecification":
            continue
        groups = []
        for ref in spec.iter():
            if _tag(ref) == "PlayerChannelRef" and ref.get("playerChannelId").endswith("_x"):
                player = player_of_channel[ref.get("playerChannelId")]
                groups.append(team_ids.index(team_of_player[player]))
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), groups))
    per_team = max(max(np.bincount(g, minlength=2)) for _, _, g in specs)

    n = specs[-1][1]
    px = np.full((n, 2 * per_team), np.nan)
    py = np.full((n, 2 * per_team), np.nan)
    ball = np.full((n, 2), np.nan)
    with open(os.path.join(folder, "Sample_Game_{}_tracking.txt".format(game))) as handle:
        text = handle.read().replace(":", ";").replace(",", ";")
    rows = [line.split(";") for line in text.splitlines()]
    del text
    for first, last, groups in specs:
        slot = [0, per_team]
        cols = []
        for team in groups:
            cols.append(slot[team])
            slot[team] += 1
        block = np.array([r for r in rows[first - 1:last]], dtype=object)
        frames = block[:, 0].astype(int)
        if not np.array_equal(frames, np.arange(first, last + 1)):
            raise ValueError("game {}: frames {}..{} not contiguous".format(game, first, last))
        values = np.where(block[:, 1:] == "NaN", np.nan, block[:, 1:]).astype(float)
        for g, col in enumerate(cols):
            px[first - 1:last, col] = values[:, 2 * g]
            py[first - 1:last, col] = values[:, 2 * g + 1]
        ball[first - 1:last] = values[:, 2 * len(groups):2 * len(groups) + 2]

    period = np.zeros(n, np.int8)
    period[int(params["first_half_start"]) - 1:int(params["first_half_end"])] = 1
    period[int(params["second_half_start"]) - 1:int(params["second_half_end"])] = 2
    team = np.array([HOME] * per_team + [AWAY] * per_team, np.int8)
    return period, px, py, ball, team


def load_tracking(game):
    """dict in solution/atd.py's format, plus 'period' per frame. Index = Frame - 1."""
    if game == 3:
        period, px, py, ball, team = _load_tracking_epts(game)
        length, width = PITCH
        px, py = px * length, py * width
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
    for p in np.unique(trk["period"][trk["period"] > 0]):
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


def _events(game):
    """Events as a DataFrame in the CSV layout of games 1 and 2, whatever the source format."""
    folder = _game_dir(game)
    if game != 3:
        return pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(game)))
    with open(os.path.join(folder, "Sample_Game_{}_events.json".format(game))) as handle:
        data = json.load(handle)["data"]
    root = ET.parse(os.path.join(folder, "Sample_Game_{}_metadata.xml".format(game))).getroot()
    home_id = [t.get("id") for t in root.iter() if _tag(t) == "Team"][0]
    rows = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = sub["name"]
        elif sub:
            sub = "-".join(s["name"] for s in sub)
        rows.append({"Team": "Home" if e["team"]["id"] == home_id else "Away",
                     "Type": e["type"]["name"], "Subtype": sub or np.nan,
                     "Period": e["period"], "Start Frame": e["start"]["frame"],
                     "End Frame": e["end"]["frame"] or e["start"]["frame"]})
    return pd.DataFrame(rows)


def derive_labels(game, n_frames, blocks):
    """H/A/D/X per frame from the event data.

    Alive frames take the team of the most recent possession event. Play is dead from a stoppage
    event (ball out, foul, offside, goal, end of half) until the next set piece. Frames outside
    the two periods, and before each period's first event, are X.
    """
    ev = _events(game)
    ev = ev.sort_values(["Start Frame"], kind="stable").reset_index(drop=True)
    labels = np.full(n_frames, "X", dtype="<U1")

    for period, first, last in blocks:
        rows = ev[ev["Period"] == period]
        state, team = "X", None
        cursor = dead_from = first
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
                dead_from = stop
                state = "D"
            elif row["Type"] in POSSESSION_TYPES:
                if row["Type"] in ("BALL LOST", "BALL OUT"):
                    if state == "D":
                        continue
                elif state == "D" and start <= dead_from:
                    # logged alongside the stoppage itself (game 3 records a carry at the frame
                    # the ball goes out), not a restart
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
    for g in GAMES + HOLDOUT_GAMES:
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
