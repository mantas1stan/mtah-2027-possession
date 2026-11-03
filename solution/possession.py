"""v1: per-frame evidence for {H, A, D}, decoded into possessions by a 3-state Viterbi.

Every constant below is set from football priors and from label-free statistics of the tracking,
never fitted to the public ground truth (RULES.md forbids tuning on it).

Evidence, as per-frame log-scores:
  alive vs dead   ball visible in the field, players moving        -> alive
                  ball outside the lines, ball absent, players slow, ball resting -> dead
  home vs away    which team has a player within control range of the ball, discounted while
                  the ball travels fast (a pass flying past an opponent is not control);
                  optionally a light prior from the Smart Tagging phase team (off, see
                  TAG_PRIOR).

The decoder pays a fixed cost for every change of state, so a lone touch or a deflection cannot
flip possession -- only sustained control by the other team can. That is what turns flickering
frame evidence into whole possessions, which is what spellF1 scores.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Params
FPS = 25
H, A, D = 0, 1, 2

# alive/dead evidence, nats per frame (positive = alive)
ALIVE_BALL_IN_FIELD = 0.30
DEAD_BALL_OUT = -1.00       # ball beyond a line by more than OUT_MARGIN_M
OUT_MARGIN_M = 0.5
DEAD_BALL_MISSING = -0.20
# v3: the tracker also drops the ball briefly in open play (aerial balls, crowded areas); only a
# gap at least MISSING_MIN_RUN frames long carries the full dead-ball weight
MISSING_MIN_RUN = 1
DEAD_BALL_MISSING_SHORT = -0.20
# v3: dead-ball stretches shorter than this are decoder blips, not stoppages, and are removed
MIN_DEAD_FRAMES = 0
SPEED_PIVOT = 1.5           # m/s mean player speed separating play from stoppage (histogram mode ~1.1)
SPEED_GAIN = 0.15           # nats per frame per m/s away from the pivot
SPEED_SMOOTH = FPS          # +/- frames for the mean player speed
RESTING_BALL_SPEED = 0.5    # m/s; a ball this still for RESTING_FRAMES is being set for a restart
RESTING_FRAMES = FPS
DEAD_BALL_RESTING = -0.60
# v2: a resting ball while the players are also slow is a set piece being taken, not a player
# shielding the ball in open play
DEAD_BALL_RESTING_SLOW = -1.50
RESTING_SLOW_SPEED = 1.6    # m/s, 3-s mean player speed
# v2: the whole pitch nearly standing still with the ball visible is a stoppage (foul, injury,
# free kick being set). Even a slow build-up keeps the mean player speed above ~1.2 m/s.
STOPPAGE_SPEED = 1.1        # m/s, 3-s mean player speed below which the penalty starts
STOPPAGE_RAMP = 0.4         # m/s over which it reaches full strength
DEAD_STOPPAGE = -0.60
SLOW_SMOOTH = 3 * FPS // 2  # +/- frames for the 3-s mean player speed

# team evidence
CONTROL_RADIUS_M = 2.5
TEAM_GAIN = 0.30
FAST_BALL = 7.0             # m/s; above this the ball is travelling, not controlled
FAST_BALL_DISCOUNT = 0.25
BALL_SMOOTH = 2             # +/- frames for ball velocity
# Nats per frame toward the Smart Tagging phase team. Off: the external games have no Smart
# Tagging, so this weight cannot be tuned on external data as the rules require.
TAG_PRIOR = 0.0

# decoder: cost of each change of state, in nats
COST_SWITCH_TEAM = 6.0
COST_TO_DEAD = 8.0
COST_FROM_DEAD = 8.0

# Smart Tagging: codes whose Team is the side in possession, and codes whose Team is the other one
TAG_ATTACKING = ("build up", "progression", "creating chances", "ball in final third",
                 "ball in the box", "players in the box", "long ball", "attacking transition")
TAG_DEFENDING = ("defending in attacking third", "defending in middle third",
                 "defending in defensive third", "defensive transition")
TAG_TEAM = {"barcelona": H, "atletic club": A, "athletic club": A}

# The FAQ publishes the second-half kickoff: graded frames before it are all dead ball.
SECOND_HALF_KICKOFF = 113742


def _centered_mean(x, half):
    """Mean over +/- half frames ignoring NaN."""
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate(([0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _run_lengths_at(mask):
    """For each frame, the length of the run of True it belongs to (0 where False)."""
    x = np.concatenate(([0], mask.astype(np.int8), [0]))
    e = np.diff(x)
    starts, stops = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    out = np.zeros(len(mask), np.int64)
    for s, t in zip(starts, stops):
        out[s:t] = t - s
    return out


def evidence(trk, tags, game=GAME):
    """(n, 3) per-frame log-scores for H, A, D."""
    length, width = GAMES[game]["pitch"]
    bx, by = trk["bx"], trk["by"]
    n = len(bx)
    has = ~np.isnan(bx)

    # alive vs dead
    out = has & ((bx < -OUT_MARGIN_M) | (bx > length + OUT_MARGIN_M)
                 | (by < -OUT_MARGIN_M) | (by > width + OUT_MARGIN_M))
    with np.errstate(invalid="ignore"):
        frame_speed = np.nanmean(trk["ps"], axis=1)
    mean_speed = _centered_mean(frame_speed, SPEED_SMOOTH)
    slow_speed = np.nan_to_num(_centered_mean(frame_speed, SLOW_SMOOTH), nan=0.0)
    speed_term = SPEED_GAIN * np.nan_to_num(mean_speed - SPEED_PIVOT, nan=-SPEED_PIVOT)
    stoppage = DEAD_STOPPAGE * np.clip((STOPPAGE_SPEED - slow_speed) / STOPPAGE_RAMP, 0.0, 1.0)

    sx, sy = _centered_mean(bx, BALL_SMOOTH), _centered_mean(by, BALL_SMOOTH)
    ball_speed = np.hypot(np.gradient(sx), np.gradient(sy)) * FPS
    resting = has & (np.nan_to_num(ball_speed, nan=99.0) < RESTING_BALL_SPEED)
    resting &= _run_lengths_at(resting) >= RESTING_FRAMES

    long_gap = _run_lengths_at(~has) >= MISSING_MIN_RUN
    missing = np.where(long_gap, DEAD_BALL_MISSING, DEAD_BALL_MISSING_SHORT)
    alive = np.where(~has, missing, np.where(out, DEAD_BALL_OUT, ALIVE_BALL_IN_FIELD))
    resting_term = np.where(resting, np.where(slow_speed < RESTING_SLOW_SPEED,
                                              DEAD_BALL_RESTING_SLOW, DEAD_BALL_RESTING), 0.0)
    alive = alive + speed_term + resting_term + np.where(has, stoppage, 0.0)

    # home vs away: closeness of each team's nearest player to the ball
    dist = np.hypot(trk["px"] - bx[:, None], trk["py"] - by[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    near = np.empty((n, 2))
    for team in (H, A):
        near[:, team] = dist[:, trk["team"] == team].min(axis=1)
    control = np.clip(1.0 - near / CONTROL_RADIUS_M, 0.0, 1.0)
    weight = np.where(np.nan_to_num(ball_speed, nan=0.0) > FAST_BALL, FAST_BALL_DISCOUNT, 1.0)
    team = np.where(has & ~out, TEAM_GAIN * weight * (control[:, H] - control[:, A]), 0.0)
    team = team + TAG_PRIOR * tag_team_prior(tags, n)

    score = np.zeros((n, 3))
    score[:, H] = 0.5 * alive + 0.5 * team
    score[:, A] = 0.5 * alive - 0.5 * team
    score[:, D] = -0.5 * alive
    return score


def tag_team_prior(tags, n):
    """+1 where a Smart Tagging phase says home has the ball, -1 for away, 0 otherwise."""
    prior = np.zeros(n)
    for ev in tags:
        code = ev["code"].strip().lower()
        side = TAG_TEAM.get((ev["tags"].get("Team") or "").strip().lower())
        if side is None:
            continue
        if code in TAG_ATTACKING:
            sign = 1.0 if side == H else -1.0
        elif code in TAG_DEFENDING:
            sign = -1.0 if side == H else 1.0
        else:
            continue
        prior[max(0, ev["start_frame"]):min(n, ev["end_frame"])] += sign
    return np.clip(prior, -1.0, 1.0)


def viterbi(score):
    """Most likely state path under per-frame scores and fixed change costs."""
    cost = np.array([[0.0, COST_SWITCH_TEAM, COST_TO_DEAD],
                     [COST_SWITCH_TEAM, 0.0, COST_TO_DEAD],
                     [COST_FROM_DEAD, COST_FROM_DEAD, 0.0]])
    t_len = len(score)
    back = np.zeros((t_len, 3), np.int8)
    value = score[0].copy()
    for t in range(1, t_len):
        cand = value[:, None] - cost          # from i to j
        back[t] = np.argmax(cand, axis=0)
        value = cand[back[t], np.arange(3)] + score[t]
    path = np.empty(t_len, np.int8)
    path[-1] = int(np.argmax(value))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def load_params(path=os.path.join(HERE, "params.json")):
    """Override the defaults above with the values tuned on external games (solution/tune.py)."""
    import json
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        params = json.load(handle)
    for name, value in params.items():
        if name not in globals():
            raise KeyError("unknown parameter {} in {}".format(name, path))
        globals()[name] = value
    return params


def drop_short_dead(path, min_frames):
    """Relabel dead stretches shorter than min_frames that sit between two alive stretches.

    Same team either side: the possession simply continues. Different teams: the hand-over is
    placed in the middle of the blip.
    """
    if min_frames <= 0:
        return path
    path = path.copy()
    breaks = np.concatenate(([0], np.flatnonzero(path[1:] != path[:-1]) + 1, [len(path)]))
    for i in range(1, len(breaks) - 2):
        start, stop = breaks[i], breaks[i + 1]
        if path[start] != D or stop - start >= min_frames:
            continue
        before, after = path[start - 1], path[stop]
        if before == D or after == D:
            continue
        middle = (start + stop) // 2
        path[start:middle] = before
        path[middle:stop] = after
    return path


def predict(trk, tags, game=GAME, blocks=None):
    """Label array over all video frames: H/A/D on graded frames, X elsewhere.

    blocks defaults to the competition game's halves; external games pass their own.
    """
    score = evidence(trk, tags, game)
    labels = np.full(len(score), "X", dtype="<U1")
    for _, first, last in blocks or half_video_blocks(game):
        path = drop_short_dead(viterbi(score[first:last + 1]), MIN_DEAD_FRAMES)
        labels[first:last + 1] = np.array(["H", "A", "D"])[path]
    if blocks is None:
        graded_pre_kickoff = slice(half_video_blocks(game)[1][1], SECOND_HALF_KICKOFF)
        labels[graded_pre_kickoff] = "D"
    return labels
