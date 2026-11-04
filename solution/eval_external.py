"""Score the possession model on the external games, with the official metric code.

    python solution/eval_external.py

This is where models are compared: the competition game is never scored for tuning.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import external  # noqa: E402
import possession  # noqa: E402
import possession_metrics as pm  # noqa: E402

# The pitch size enters possession.evidence through games.GAMES; register the external one.
from games import GAMES  # noqa: E402

EXTERNAL_GAME = "METRICA-SAMPLE"
GAMES.setdefault(EXTERNAL_GAME, {"pitch": external.PITCH, "fps": external.FPS})


_GAMES = {}


def get_game(game):
    """(tracking, blocks, labels) for an external game, loaded once per process."""
    if game not in _GAMES:
        _GAMES[game] = external.load_game(game)
    return _GAMES[game]


def score_game(game):
    trk, blocks, gt = get_game(game)
    pred = possession.predict(trk, [], game=EXTERNAL_GAME, blocks=blocks)
    result = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    return result


def report(games=None):
    rows = []
    for g in games or external.GAMES:
        r = score_game(g)
        rows.append((g, r["spell_f1_mean"], r["macro_f1"]))
        print("game {}  spellF1 {:.4f}  macroF1 {:.4f}".format(g, r["spell_f1_mean"], r["macro_f1"]))
    print("mean    spellF1 {:.4f}  macroF1 {:.4f}".format(
        sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows) / len(rows)))


if __name__ == "__main__":
    possession.load_params()
    # The submitted model and parameters are fitted on all three games, so these scores are
    # in-sample. The out-of-sample estimate is `python solution/learned.py --cv`.
    print("games 1-2")
    report()
    print("game 3")
    report(external.HOLDOUT_GAMES)
    print("(in-sample for the submitted model; out-of-sample estimate: learned.py --cv)")
