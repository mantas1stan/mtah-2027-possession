"""Score the possession model on the external games, with the official metric code.

    python solution/eval_external.py

This is where parameters are chosen: the competition game is never scored for tuning.
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


def score_game(game, cache={}):
    if game not in cache:
        cache[game] = external.load_game(game)
    trk, blocks, gt = cache[game]
    pred = possession.predict(trk, [], game=EXTERNAL_GAME, blocks=blocks)
    result = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    return result


def report():
    rows = []
    for g in external.GAMES:
        r = score_game(g)
        rows.append((g, r["spell_f1_mean"], r["macro_f1"]))
        print("game {}  spellF1 {:.4f}  macroF1 {:.4f}".format(g, r["spell_f1_mean"], r["macro_f1"]))
    print("mean    spellF1 {:.4f}  macroF1 {:.4f}".format(
        sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows) / len(rows)))


if __name__ == "__main__":
    report()
