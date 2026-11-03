"""Tune possession.py's parameters on the external games only.

    python solution/tune.py                 # tune on games 1+2, write solution/params.json
    python solution/tune.py --cross         # tune on one game, score the other (generalisation)

Coordinate search: each parameter in turn is tried over a grid while the others are held, the best
value kept, and the sweep repeated until nothing improves. The objective is the mean spellF1 over
the tuning games, computed with the official metric code. The competition game is never used.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_external  # noqa: E402
import possession  # noqa: E402

# Params
PARAMS_PATH = os.path.join(HERE, "params.json")
MAX_SWEEPS = 4
GRID = {
    "DEAD_BALL_MISSING": [-0.05, -0.1, -0.2, -0.3, -0.5, -0.8, -1.2, -1.8, -2.5, -4.0],
    "ALIVE_BALL_IN_FIELD": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2],
    "DEAD_BALL_OUT": [0.0, -0.15, -0.3, -0.6, -1.0, -2.0, -3.0],
    "SPEED_PIVOT": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    "SPEED_GAIN": [0.0, 0.05, 0.1, 0.15, 0.25, 0.4],
    "DEAD_BALL_RESTING": [0.0, -0.3, -0.6, -1.0],
    "DEAD_BALL_RESTING_SLOW": [0.0, -0.3, -0.6, -1.0, -1.5, -2.5],
    "DEAD_STOPPAGE": [0.0, -0.15, -0.3, -0.6, -1.0, -1.5],
    "STOPPAGE_SPEED": [0.8, 1.0, 1.1, 1.3],
    "CONTROL_RADIUS_M": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
    "TEAM_GAIN": [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0],
    "COST_SWITCH_TEAM": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0],
    "COST_TO_DEAD": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
    "COST_FROM_DEAD": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
}
# FAST_BALL_DISCOUNT tuned to 1.0 (no discount) in the first search, which makes FAST_BALL moot:
# both are dropped from the search and fixed in params.json.


def objective(games):
    return sum(eval_external.score_game(g)["spell_f1_mean"] for g in games) / len(games)


def apply(params):
    for k, v in params.items():
        setattr(possession, k, v)


def current():
    return {k: getattr(possession, k) for k in GRID}


def coordinate_search(games, log=print):
    best_params = current()
    best = objective(games)
    log("start  spellF1 {:.4f}".format(best))
    for sweep in range(MAX_SWEEPS):
        improved = False
        for name, grid in GRID.items():
            for value in grid:
                if value == best_params[name]:
                    continue
                apply(dict(best_params, **{name: value}))
                score = objective(games)
                if score > best + 1e-4:
                    best, best_params, improved = score, dict(best_params, **{name: value}), True
                    log("  sweep {} {:<24} = {:<6} spellF1 {:.4f}".format(sweep, name, value, best))
            apply(best_params)
        if not improved:
            break
    return best_params, best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cross", action="store_true")
    args = parser.parse_args()
    fixed = possession.load_params(PARAMS_PATH)  # resume from the last search, if any
    defaults = current()

    if args.cross:
        for train, test in ((1, 2), (2, 1)):
            apply(defaults)
            params, score = coordinate_search([train], log=lambda *_: None)
            print("tuned on game {}: {:.4f}   held-out game {}: {:.4f}".format(
                train, score, test, objective([test])))
    else:
        params, score = coordinate_search(list(eval_external.external.GAMES))
        params = dict(fixed, **params)
        with open(PARAMS_PATH, "w", newline="\n") as handle:
            json.dump(params, handle, indent=2, sort_keys=True)
            handle.write("\n")
        for g in eval_external.external.GAMES:
            print("game {}: spellF1 {:.4f}".format(g, objective([g])))
        print("wrote", PARAMS_PATH)
