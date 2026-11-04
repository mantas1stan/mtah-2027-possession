"""Learned per-frame evidence: two logistic regressions trained on the external games.

    python solution/learned.py          # fit on the tuning games, write solution/model.json

  alive model   P(ball in play | features)        -- intercept, trained on every labelled frame
  team model    P(home has the ball | features)   -- no intercept and only antisymmetric features
                                                     (home minus away), so it cannot learn a
                                                     preference for either side from the
                                                     external games, only how to read the geometry

Training data: the event-derived labels of Metrica's sample games (solution/external.py). Model
choices are made by leave-one-game-out cross-validation (--cv); the submitted model is then
fitted on all three games (--final). possession.py turns the two logits into Viterbi scores.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAMES, half_video_blocks  # noqa: E402

# Params
MODEL_PATH = os.path.join(HERE, "model.json")
FPS = 25
FAR_M = 30.0          # stand-in distance where a team has nobody tracked
L2 = 1.0              # ridge penalty on standardised coefficients
NEWTON_STEPS = 25
CONTEXT_WINDOWS = (12, 50, 125)  # frames looked back / ahead for the control context
# Team shape and movement features (attacking direction, forward movement, spread, pressing,
# ball half). Kept switchable so cross-validation can compare with and without them.
SHAPE_FEATURES = True
# Set-piece shape features for the alive model (crowding around the ball, overall spread, how
# still the players are relative to each other).
ALIVE_SHAPE_FEATURES = False  # tried: no measurable gain in cross-validation
# Training weight of frames within HANDOVER_WINDOW frames of a change of possession (both
# models): those frames are few but decide where possessions start and end.
HANDOVER_WEIGHT = 40.0
HANDOVER_WINDOW = 50
MAX_PLAYER_SPEED = 12.0  # m/s; faster "movement" is a tracking identity switch


def _centered(x, lo, hi):
    """Mean of x over [t+lo, t+hi] ignoring NaN; NaN where the window has no value."""
    ok = ~np.isnan(x)
    cs = np.concatenate(([0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    a = np.clip(i + lo, 0, len(x))
    b = np.clip(i + hi + 1, 0, len(x))
    n = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[b] - cs[a]) / np.maximum(n, 1), np.nan)


def _run_lengths(mask):
    x = np.concatenate(([0], mask.astype(np.int8), [0]))
    e = np.diff(x)
    out = np.zeros(len(mask))
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        out[s:t] = t - s
    return out


def _since_until(has):
    """Frames since the ball was last seen and until it is next seen (0 where it is seen)."""
    n = len(has)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(has, idx, -10 ** 6))
    nxt = np.minimum.accumulate(np.where(has, idx, 10 ** 7)[::-1])[::-1]
    return np.clip(idx - last, 0, 10 ** 4), np.clip(nxt - idx, 0, 10 ** 4)


def _fill(x, has, forward=True):
    """Carry the last (or next) value seen while the ball was visible across gaps."""
    idx = np.arange(len(x))
    if forward:
        src = np.maximum.accumulate(np.where(has, idx, 0))
    else:
        src = np.minimum.accumulate(np.where(has, idx, len(x) - 1)[::-1])[::-1]
    return x[src]


_FEATURE_CACHE = {}


def features(trk, game):
    """(alive_X, alive_names, team_X, team_names), one row per frame. Cached per tracking dict."""
    key = (id(trk), game)
    hit = _FEATURE_CACHE.get(key)
    if hit is None or hit[0] is not trk:  # the cached object itself, not just a reused id
        hit = _FEATURE_CACHE[key] = (trk, _features(trk, game))
    return hit[1]


def _half_of_frame(trk, game):
    """Half number per frame (0 outside play): from trk['period'] if present, else the game's
    published half boundaries."""
    if "period" in trk:
        return np.asarray(trk["period"], int)
    out = np.zeros(len(trk["bx"]), int)
    for half, first, last in half_video_blocks(game):
        out[first:last + 1] = half
    return out


def _shape_features(trk, game, has, bx, length, dist):
    """Home-minus-away features of team shape and movement; they work without the ball too."""
    px, py = trk["px"].astype(np.float64), trk["py"].astype(np.float64)
    with np.errstate(invalid="ignore"):
        cx = {t: np.nanmean(px[:, trk["team"] == t], axis=1) for t in (0, 1)}
        spread = {t: np.sqrt(np.nanvar(px[:, trk["team"] == t], axis=1)
                             + np.nanvar(py[:, trk["team"] == t], axis=1)) for t in (0, 1)}
    half = _half_of_frame(trk, game)
    direction = np.zeros(len(bx))  # +1: home attacks towards +x
    for h in np.unique(half[half > 0]):
        sel = half == h
        gap = np.nanmean(cx[1][sel] - cx[0][sel])
        direction[sel] = 1.0 if gap > 0 else -1.0

    forward = {}
    for t, sign in ((0, 1.0), (1, -1.0)):
        smooth = _centered(cx[t], -5, 5)
        forward[t] = np.clip(np.gradient(smooth) * FPS, -6, 6) * sign * direction
    fwd = forward[0] - forward[1]

    # pressing: mean speed towards the ball of each team's three nearest players
    pvx = np.clip(np.gradient(px, axis=0) * FPS, -MAX_PLAYER_SPEED, MAX_PLAYER_SPEED)
    pvy = np.clip(np.gradient(py, axis=0) * FPS, -MAX_PLAYER_SPEED, MAX_PLAYER_SPEED)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = (bx[:, None] - px) / dist
        uy = (trk["by"][:, None] - py) / dist
        toward = np.nan_to_num(pvx * ux + pvy * uy)
    press = {}
    for t in (0, 1):
        cols = np.flatnonzero(trk["team"] == t)
        d = dist[:, cols]
        k = np.argsort(d, axis=1)[:, :3]
        press[t] = np.take_along_axis(toward[:, cols], k, axis=1).mean(axis=1)
    press_diff = np.where(has, press[1] - press[0], np.nan)  # away pressing -> home has it

    ball_half = np.where(has, (bx - length / 2) / (length / 2) * direction, np.nan)
    cols = {
        "fwd_12": np.nan_to_num(_centered(fwd, -12, 12)),
        "fwd_50": np.nan_to_num(_centered(fwd, -50, 50)),
        "spread_25": np.nan_to_num(_centered(spread[0] - spread[1], -25, 25)),
        "press": np.nan_to_num(press_diff),
        "press_12": np.nan_to_num(_centered(press_diff, -12, 12)),
        "ball_half": np.nan_to_num(ball_half),
        "ball_half_seen": np.where(has, 0.0, _fill(np.nan_to_num(ball_half), has, True)),
    }
    return cols


def _features(trk, game):
    length, width = GAMES[game]["pitch"]
    bx, by = trk["bx"], trk["by"]
    has = ~np.isnan(bx)
    n = len(bx)

    sx, sy = _centered(bx, -2, 2), _centered(by, -2, 2)
    vx, vy = np.gradient(sx) * FPS, np.gradient(sy) * FPS
    ball_speed = np.nan_to_num(np.clip(np.hypot(vx, vy), 0, 40))
    ax = np.nan_to_num(np.clip(np.hypot(np.gradient(vx), np.gradient(vy)) * FPS, 0, 200))

    px, py = trk["px"].astype(np.float64), trk["py"].astype(np.float64)
    pvx, pvy = np.gradient(px, axis=0) * FPS, np.gradient(py, axis=0) * FPS
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)

    per_team = {}
    for team in (0, 1):
        d = dist[:, trk["team"] == team]
        order = np.argsort(d, axis=1)
        d_sorted = np.take_along_axis(d, order, axis=1)
        d1 = np.minimum(d_sorted[:, 0], FAR_M)
        d2 = np.minimum(d_sorted[:, 1], FAR_M)
        cols = np.flatnonzero(trk["team"] == team)[order[:, 0]]
        rows = np.arange(n)
        rel = np.hypot(pvx[rows, cols] - vx, pvy[rows, cols] - vy)
        rel = np.nan_to_num(np.clip(rel, 0, 30), nan=30.0)
        near = np.exp(-d1 / 1.0)
        per_team[team] = {
            "d1": d1, "d2": d2, "near": near,
            "with_ball": near * np.exp(-rel / 3.0),
            "n2": np.sum(d < 2.0, axis=1), "n5": np.sum(d < 5.0, axis=1),
        }
    h, a = per_team[0], per_team[1]
    present = has.astype(float)

    # team features: home minus away, zero where the ball is not seen
    control = np.where(has, h["near"] - a["near"], np.nan)
    team_cols = {
        "d1_diff": a["d1"] - h["d1"],
        "logd1_diff": np.log(a["d1"] + 0.3) - np.log(h["d1"] + 0.3),
        "d2_diff": a["d2"] - h["d2"],
        "near_diff": h["near"] - a["near"],
        "with_ball_diff": h["with_ball"] - a["with_ball"],
        "n2_diff": h["n2"] - a["n2"],
        "n5_diff": h["n5"] - a["n5"],
        "near_diff_slow": (h["near"] - a["near"]) * (ball_speed < 5),
        "near_diff_turn": (h["near"] - a["near"]) * np.clip(ax / 50.0, 0, 1),
    }
    for name in list(team_cols):
        team_cols[name] = np.where(has, np.nan_to_num(team_cols[name]), 0.0)
    for w in CONTEXT_WINDOWS:
        team_cols["ctx_past_{}".format(w)] = np.nan_to_num(_centered(control, -w, -1))
        team_cols["ctx_next_{}".format(w)] = np.nan_to_num(_centered(control, 1, w))
    seen = np.nan_to_num(control)
    team_cols["last_seen"] = np.where(has, 0.0, _fill(seen, has, True))
    team_cols["next_seen"] = np.where(has, 0.0, _fill(seen, has, False))
    if SHAPE_FEATURES:
        team_cols.update(_shape_features(trk, game, has, bx, length, dist))

    # alive features
    since, until = _since_until(has)
    with np.errstate(invalid="ignore"):
        frame_speed = np.nanmean(trk["ps"], axis=1)
    sp1 = np.nan_to_num(_centered(frame_speed, -25, 25), nan=0.0)
    sp3 = np.nan_to_num(_centered(frame_speed, -75, 75), nan=0.0)
    trend = np.nan_to_num(_centered(frame_speed, 0, 50) - _centered(frame_speed, -50, -1))
    outside = np.where(has, np.maximum.reduce([-bx, bx - length, -by, by - width]), 0.0)
    edge = np.where(has, np.minimum.reduce([bx, length - bx, by, width - by]), 0.0)
    resting = has & (ball_speed < 0.5)
    dmin = np.minimum(h["d1"], a["d1"])
    alive_cols = {
        "has": present,
        "log_gap_len": np.log1p(_run_lengths(~has)),
        "log_since": np.log1p(np.where(has, 0, since)),
        "log_until": np.log1p(np.where(has, 0, until)),
        "log_seen_len": np.log1p(_run_lengths(has)),
        "outside": np.clip(np.nan_to_num(outside), 0, 5) * (outside > 0),
        "edge": np.clip(np.nan_to_num(edge), 0, 20),
        "ball_speed": ball_speed * present,
        "log_resting_len": np.log1p(_run_lengths(resting)),
        "speed_1s": sp1, "speed_3s": sp3, "speed_trend": trend,
        # no "number of tracked players": the external tracking is complete on almost every
        # frame, so the model learned that a missing player means a stoppage -- which does not
        # transfer to the competition's automatic tracking, where players drop out in open play
        "dmin": np.where(has, np.minimum(dmin, 15), 0.0),
    }
    if ALIVE_SHAPE_FEATURES:
        px, py = trk["px"].astype(np.float64), trk["py"].astype(np.float64)
        with np.errstate(invalid="ignore"):
            spread_all = np.sqrt(np.nanvar(px, axis=1) + np.nanvar(py, axis=1))
            speed_sd = np.nanstd(trk["ps"], axis=1)
        crowd = np.sum(dist < 10.0, axis=1)
        alive_cols["spread_all"] = np.nan_to_num(_centered(spread_all, -12, 12))
        alive_cols["speed_sd_1s"] = np.nan_to_num(_centered(speed_sd, -25, 25))
        alive_cols["crowd_10m"] = np.where(has, crowd, 0) / 22.0
        alive_cols["corner_zone"] = np.where(
            has, (np.minimum(bx, length - bx) < 3) & (np.minimum(by, width - by) < 3), 0.0)
    to_matrix = lambda cols: np.column_stack([np.asarray(v, float) for v in cols.values()])  # noqa: E731
    return to_matrix(alive_cols), list(alive_cols), to_matrix(team_cols), list(team_cols)


def fit_logistic(X, y, intercept, weight=None):
    """Ridge logistic regression by Newton's method on standardised columns."""
    weight = np.ones(len(y)) if weight is None else weight
    mu = X.mean(axis=0) if intercept else np.zeros(X.shape[1])
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    if intercept:
        Z = np.column_stack([np.ones(len(Z)), Z])
    w = np.zeros(Z.shape[1])
    penalty = np.full(Z.shape[1], L2)
    if intercept:
        penalty[0] = 0.0
    for _ in range(NEWTON_STEPS):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        grad = Z.T @ (weight * (p - y)) + penalty * w
        hess = (Z * (weight * p * (1 - p))[:, None]).T @ Z + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "intercept": bool(intercept)}


def logit(model, X):
    Z = (X - np.array(model["mu"])) / np.array(model["sd"])
    w = np.array(model["w"])
    return Z @ w[1:] + w[0] if model["intercept"] else Z @ w


def load_model(path=MODEL_PATH):
    with open(path) as handle:
        return json.load(handle)


def train(games):
    import eval_external
    import external
    rows_a, ya, rows_t, yt, wa, wt = [], [], [], [], [], []
    for g in games:
        trk, blocks, labels = eval_external.get_game(g)
        Xa, names_a, Xt, names_t = features(trk, eval_external.EXTERNAL_GAME)
        graded = labels != "X"
        change = np.zeros(len(labels), bool)
        change[1:] = (labels[1:] != labels[:-1]) & graded[1:] & graded[:-1]
        near_change = np.convolve(change, np.ones(2 * HANDOVER_WINDOW + 1), "same") > 0
        weight = np.where(near_change, HANDOVER_WEIGHT, 1.0)
        rows_a.append(Xa[graded])
        ya.append((labels[graded] != "D").astype(float))
        wa.append(weight[graded])
        alive = (labels == "H") | (labels == "A")
        rows_t.append(Xt[alive])
        yt.append((labels[alive] == "H").astype(float))
        wt.append(weight[alive])
    alive_model = fit_logistic(np.vstack(rows_a), np.concatenate(ya), intercept=True,
                               weight=np.concatenate(wa))
    team_model = fit_logistic(np.vstack(rows_t), np.concatenate(yt), intercept=False,
                              weight=np.concatenate(wt))
    alive_model["names"], team_model["names"] = names_a, names_t
    prior = float(np.mean(np.concatenate(ya)))
    alive_model["prior_logit"] = float(np.log(prior / (1 - prior)))
    return {"alive": alive_model, "team": team_model, "trained_on": list(games)}


def cross_validate(games):
    """Leave one game out: fit on the others, score the left-out game with the current params.

    Returns {game: spellF1}. This is how feature and model choices are compared.
    """
    import eval_external
    import possession
    out = {}
    for test in games:
        possession.MODEL_OVERRIDE = train([g for g in games if g != test])
        out[test] = eval_external.score_game(test)["spell_f1_mean"]
    possession.MODEL_OVERRIDE = None
    return out


if __name__ == "__main__":
    import argparse
    import external
    import possession
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cv", action="store_true", help="leave-one-game-out scores only")
    parser.add_argument("--final", action="store_true",
                        help="fit on all three external games (the submitted model)")
    args = parser.parse_args()
    if args.cv:
        possession.load_params()
        scores = cross_validate(external.GAMES + external.HOLDOUT_GAMES)
        print("leave-one-game-out spellF1", {g: round(v, 4) for g, v in scores.items()},
              "mean {:.4f}".format(sum(scores.values()) / len(scores)))
        raise SystemExit
    model = train(external.GAMES + external.HOLDOUT_GAMES if args.final else external.GAMES)
    with open(MODEL_PATH, "w", newline="\n") as handle:
        json.dump(model, handle, indent=1)
        handle.write("\n")
    for part in ("alive", "team"):
        m = model[part]
        w = m["w"][1:] if m["intercept"] else m["w"]
        print(part, ", ".join("{} {:+.2f}".format(k, v) for k, v in zip(m["names"], w)))
    print("wrote", MODEL_PATH)
