"""Schema probe for the ATD: what fields exist, and their aggregate statistics.

Prints structure and summary numbers only, never a raw record.

    python tools/probe_atd.py
"""

import collections
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import half_video_blocks, require  # noqa: E402

# Params
MAX_CATEGORIES = 12  # a field with at most this many distinct values is reported as categorical


def describe_meta(meta_path):
    """Element tag counts and the DataSpec / channel layout -- the schema, not the values."""
    root = ET.parse(meta_path).getroot()
    tags = collections.Counter(el.tag.split("}")[-1] for el in root.iter())
    print("metadata element counts:")
    for tag, n in sorted(tags.items()):
        print("  {:<28} {}".format(tag, n))

    print("\nchannel ids (schema):")
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("Channel", "PlayerChannel", "SensorChannel", "Sensor"):
            print("  {:<14} {}".format(tag, {k: v for k, v in el.attrib.items()
                                             if k.lower() in ("id", "channelid", "unit")}))


def graded(frame_id, blocks):
    video = frame_id - 1
    return any(first <= video <= last for _, first, last in blocks)


def describe_raw(raw_path):
    """Per-section field counts, and per-field stats for the ball section."""
    blocks = half_video_blocks("BAR-ATH")
    parts_count = collections.Counter()
    player_groups = collections.Counter()
    player_fields = collections.Counter()
    ball_fields = collections.Counter()
    ball_stats = collections.defaultdict(lambda: {"nan": 0, "num": 0, "other": 0,
                                                  "values": collections.Counter()})
    n_lines = n_graded = 0

    with open(raw_path) as handle:
        for line in handle:
            n_lines += 1
            parts = line.rstrip("\n").split(":")
            parts_count[len(parts)] += 1
            if len(parts) < 3:
                continue
            groups = parts[1].split(";")
            player_groups[len(groups)] += 1
            player_fields[len(groups[0].split(","))] += 1

            if not graded(int(parts[0]), blocks):
                continue
            n_graded += 1
            fields = parts[2].split(",")
            ball_fields[len(fields)] += 1
            for i, value in enumerate(fields):
                stat = ball_stats[i]
                if value.strip().lower() == "nan":
                    stat["nan"] += 1
                    continue
                try:
                    float(value)
                    stat["num"] += 1
                except ValueError:
                    stat["other"] += 1
                if len(stat["values"]) <= MAX_CATEGORIES:
                    stat["values"][value.strip()] += 1

    print("\nraw lines: {}   graded: {}".format(n_lines, n_graded))
    print("':'-sections per line: {}".format(dict(parts_count)))
    print("player groups per line: {}".format(dict(player_groups)))
    print("fields per player group: {}".format(dict(player_fields)))
    print("fields in ball section (graded frames): {}".format(dict(ball_fields)))
    print("\nball field stats over graded frames:")
    for i in sorted(ball_stats):
        stat = ball_stats[i]
        line = "  field {}: numeric {:>7}  NaN {:>7}  non-numeric {:>7}".format(
            i, stat["num"], stat["nan"], stat["other"])
        if len(stat["values"]) <= MAX_CATEGORIES:
            line += "   categorical: {}".format(dict(stat["values"].most_common()))
        else:
            line += "   continuous"
        print(line)


if __name__ == "__main__":
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    describe_meta(meta_path)
    describe_raw(raw_path)
