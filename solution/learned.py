"""Learned per-frame evidence: two logistic regressions trained on the external games.

    python solution/learned.py          # fit on the tuning games, write solution/model.json

  alive model   P(ball in play | features)        -- intercept, trained on every labelled frame
  team model    P(home has the ball | features)   -- no intercept and only antisymmetric features
                                                     (home minus away), so it cannot learn a
                                                     preference for either side from the
                                                     external games, only how to read the geometry

Training data: the event-derived labels of Metrica's sample games 1 and 2 (solution/external.py).
Game 3 is never used here. possession.py turns the two logits into Viterbi scores.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAMES  # noqa: E402

# Params
MODEL_PATH = os.path.join(HERE, "model.json")
FPS = 25
FAR_M = 30.0          # stand-in distance where a team has nobody tracked
L2 = 1.0              # ridge penalty on standardised coefficients
NEWTON_STEPS = 25
CONTEXT_WINDOWS = (12, 50, 125)  # frames looked back / ahead for the control context


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


def features(trk, game):
    """(alive_X, alive_names, team_X, team_names), one row per frame."""
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
    tracked = np.sum(~np.isnan(trk["px"]), axis=1)
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
        "tracked": tracked / 22.0,
        "dmin": np.where(has, np.minimum(dmin, 15), 0.0),
    }
    to_matrix = lambda cols: np.column_stack([np.asarray(v, float) for v in cols.values()])  # noqa: E731
    return to_matrix(alive_cols), list(alive_cols), to_matrix(team_cols), list(team_cols)


def fit_logistic(X, y, intercept):
    """Ridge logistic regression by Newton's method on standardised columns."""
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
        grad = Z.T @ (p - y) + penalty * w
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty)
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
    rows_a, ya, rows_t, yt = [], [], [], []
    for g in games:
        trk, blocks, labels = external.load_game(g)
        Xa, names_a, Xt, names_t = features(trk, eval_external.EXTERNAL_GAME)
        graded = labels != "X"
        rows_a.append(Xa[graded])
        ya.append((labels[graded] != "D").astype(float))
        alive = (labels == "H") | (labels == "A")
        rows_t.append(Xt[alive])
        yt.append((labels[alive] == "H").astype(float))
    alive_model = fit_logistic(np.vstack(rows_a), np.concatenate(ya), intercept=True)
    team_model = fit_logistic(np.vstack(rows_t), np.concatenate(yt), intercept=False)
    alive_model["names"], team_model["names"] = names_a, names_t
    prior = float(np.mean(np.concatenate(ya)))
    alive_model["prior_logit"] = float(np.log(prior / (1 - prior)))
    return {"alive": alive_model, "team": team_model, "trained_on": list(games)}


if __name__ == "__main__":
    import external
    model = train(external.GAMES)
    with open(MODEL_PATH, "w", newline="\n") as handle:
        json.dump(model, handle, indent=1)
        handle.write("\n")
    for part in ("alive", "team"):
        m = model[part]
        w = m["w"][1:] if m["intercept"] else m["w"]
        print(part, ", ".join("{} {:+.2f}".format(k, v) for k, v in zip(m["names"], w)))
    print("wrote", MODEL_PATH)
