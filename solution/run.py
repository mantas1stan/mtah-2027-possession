"""Regenerate the submission from the provided data.

    MTAH_DATA_DIR=/path/to/data python solution/run.py [--out evaluation/results/v1.csv]

Prints only aggregate statistics of the prediction (counts and durations), never labels.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import atd  # noqa: E402
import load_smart_tagging  # noqa: E402
import possession  # noqa: E402
from games import half_video_blocks  # noqa: E402


def write_submission(path, labels):
    """frame,label CSV with '\\n' line endings on every OS, so the file is byte-reproducible."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def summary(labels):
    spells, deads = [], []
    for _, first, last in half_video_blocks("BAR-ATH"):
        w = labels[first:last + 1]
        b = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        for s, t in zip(b[:-1], b[1:]):
            (deads if w[s] == "D" else spells).append(t - s)
    g = labels[labels != "X"]
    spells, deads = np.array(spells), np.array(deads)
    print("graded frames {}  H {:.1%}  A {:.1%}  D {:.1%}".format(
        len(g), np.mean(g == "H"), np.mean(g == "A"), np.mean(g == "D")))
    print("possessions {}  mean {:.1f} s  median {:.1f} s  under 2 s: {}".format(
        len(spells), spells.mean() / 25, np.median(spells) / 25, int(np.sum(spells < 50))))
    print("dead-ball stretches {}  mean {:.1f} s".format(len(deads), deads.mean() / 25))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "v1.csv"))
    args = parser.parse_args()

    possession.load_params()
    labels = possession.predict(atd.load(), load_smart_tagging.load())
    write_submission(args.out, labels)
    summary(labels)
    print("wrote", os.path.relpath(args.out, ROOT))
