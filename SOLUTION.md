# Our solution

Frame-level possession for BAR-ATH from the provided tactical tracking and Smart Tagging, with
every parameter tuned on **external** matches only.

## How it works

1. **Evidence per frame** (`solution/possession.py`)
   - *alive vs dead*: ball visible in the field, ball missing, ball resting, players' mean speed;
   - *home vs away*: which team has a player within control range of the ball. (A Smart Tagging
     prior exists in the code but is off: it cannot be tuned on the external games.)
2. **Decoding**: a 3-state (H / A / D) Viterbi with a fixed cost for every change of state, so a
   lone touch or deflection cannot flip possession and each possession comes out as one interval.
3. **Tuning** (`solution/tune.py`): coordinate search on the official `spellF1`, computed against
   frame labels derived from the event data of Metrica Sports' open sample games 1 and 2
   (`solution/external.py`). Game 3 is held out and only scored afterwards. The competition game
   and its ground truth are never used for tuning.

## External data

Metrica Sports open sample data — https://github.com/metrica-sports/sample-data (anonymised
tracking and events, courtesy of Metrica Sports). Download it so that the folder layout is

```
<parent of this repo>/external-data/metrica-sample/Sample_Game_1/...
<parent of this repo>/external-data/metrica-sample/Sample_Game_2/...
```

or set `MTAH_EXTERNAL_DIR` to the folder holding `Sample_Game_1/` and `Sample_Game_2/`.

## Reproduce

```bash
pip install -r requirements-solution.txt
export MTAH_DATA_DIR=/path/to/competition/data        # holds tactical/ and broadcast/
python solution/tune.py                               # optional: re-tunes solution/params.json
python solution/run.py --out evaluation/results/<team_name>.csv
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

`run.py` is deterministic and writes `\n` line endings on every OS, so the file is byte-identical
across runs and machines.

## Layout

| path | what |
|---|---|
| `solution/atd.py` | reads the ATD into numpy arrays in metres (cached next to the data) |
| `solution/possession.py` | evidence, Viterbi decoder, parameter loading |
| `solution/params.json` | parameters tuned on the external games |
| `solution/external.py` | external games: tracking in the same format, H/A/D labels from events |
| `solution/eval_external.py` | scores the model on the external games with the official metric |
| `solution/tune.py` | the parameter search |
| `solution/run.py` | produces the submission |
| `tools/probe_atd.py` | prints the ATD schema and summary statistics |
| `docs/technical_document.md` | the technical document |
