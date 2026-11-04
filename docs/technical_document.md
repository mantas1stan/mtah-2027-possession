# Frame-level ball possession from automatic tracking

**More than a Hack 2027 — Phase 1** · Team: `<team_name>`

## 1. Summary

We label every frame of BAR-ATH as `H`, `A` or `D` with a small, interpretable pipeline: two
logistic regressions turn geometric features of the tactical tracking into per-frame evidence
(ball in play? which team controls it?), and a 3-state Viterbi with a cost for every change of
state decodes that evidence into whole possessions. The models are trained, and every parameter
tuned, on **external** matches — Metrica Sports' open sample games, labelled automatically from
their event data. The competition game, its video and its ground truth are never used for
training or tuning.

| model | trained/tuned on games 1–2 | held-out game 3 | **leave-one-game-out mean** |
|---|---|---|---|
| hand-built evidence | 0.752 | 0.686 | — |
| learned evidence | 0.732 | 0.720 | 0.724 |
| + team shape and movement features | — | 0.720 | 0.735 |
| **+ hand-over weighting (submitted)** | 0.777 | 0.747 | **0.761** |

All numbers are `spellF1` from the official metric code, against labels derived from Metrica's
event data. Game 3 was first held out and used once, to choose between the hand-built and learned
evidence. Every later choice was made by leave-one-game-out cross-validation (fit on two games,
score the third, rotate), and the submitted model is then fitted on all three games. **Our
estimate for an unseen match is the cross-validated 0.76.**

## 2. Approach and why

The ranking metric scores possessions as intervals, so the first requirement is that a possession
comes out as one interval, not as flickering frames. A hidden-state decoder gives that directly:
the label can only change when the evidence for a new state outweighs a fixed change cost, so a
lone touch, a deflection or a tracking glitch cannot split a possession, while a real hand-over
moves the label at the frame where the evidence flips — which is what makes the boundaries sharp.

We first built the evidence by hand (distance-to-ball control scores and dead-ball rules, each
weight tuned on the external games). Error analysis on the external games showed the main loss was
missed hand-overs between teams, which need several cues weighed together; we therefore replaced
the hand-built scores with two small linear models trained on the external games. They have few
coefficients (13 and 24), are regularised, and every feature has a football meaning, so they
cannot memorise three matches the way a large classifier could.

### 2.1 Per-frame evidence (`solution/learned.py`)

All quantities are computed in metres from the normalised tracking. No player identity is used,
only team membership, which suits tracking whose slots switch identity.

**Alive model** — logistic regression, P(ball in play):
- ball visible or not; length of the current gap, frames since the ball was last seen and until it
  is next seen; length of the current stretch of visibility;
- how far outside the lines the ball is; distance to the nearest line; ball speed; how long the
  ball has been resting;
- mean player speed over ±1 s and ±3 s, and its trend (players slow down after a whistle);
- distance from the ball to the nearest player.

**Team model** — logistic regression without intercept, P(home has the ball), using only
home-minus-away features so it cannot learn a preference for either side:
- nearest and second-nearest distance to the ball per team (linear and log);
- closeness of the nearest player, and closeness weighted by how well the player moves *with*
  the ball (dribbling);
- closeness while the ball is slow, and at sudden changes of ball direction (touches);
- players of each team within 2 m and 5 m;
- the control balance over the previous and next 0.5 s, 2 s and 5 s, and the last/next value
  seen around a gap in ball tracking;
- team shape and movement, which also work while the ball is lost: each team's forward movement
  towards the goal it attacks (attacking direction inferred per half from where each team stands),
  the difference in team spread, pressing (speed of each team's three nearest players towards the
  ball), and which team's attacking half the ball is in.

The logits are scaled (`ALIVE_SCALE`, `TEAM_SCALE`, tuned) before decoding, because consecutive
frames are far from independent.

**Hand-over weighting.** Frames within 2 s of a change of state are a small share of all frames,
yet they decide where every possession starts and ends. Both models weight them ×40 in training;
in cross-validation this raised spellF1 from 0.735 to 0.761, on every left-out game. Weights from
40 to 100 and windows from 1 to 3 s all score within 0.002 of each other.

**Tried and dropped.** Set-piece shape features for the alive model (crowding around the ball,
overall spread, corner zone, variation of player speeds): +0.0004 in cross-validation, i.e.
nothing, so they are off (`ALIVE_SHAPE_FEATURES`).

**Removed for not transferring.** The number of tracked players was first an alive feature. The
external tracking has all 22 players on almost every frame, so the model learned that a missing
player means a stoppage; the competition's automatic tracking regularly misses a player in open
play, and on BAR-ATH this feature alone marked 19% of ball-visible frames as dead (2–4% on the
external games). We found this by comparing average feature values on ball-visible frames between
the external games and BAR-ATH — summary statistics only, no labels — and removed the feature;
cross-validated spellF1 was unchanged (0.761). A check worth repeating for any feature: does it
mean the same thing in both tracking sources?

### 2.2 Decoding

A Viterbi over {H, A, D} per half, with costs `COST_SWITCH_TEAM` (H↔A), `COST_TO_DEAD` and
`COST_FROM_DEAD`. Dead-ball stretches shorter than `MIN_DEAD_FRAMES` that sit between two alive
stretches are then removed (the possession continues, or the hand-over is placed at the middle).

The graded frames before the second-half kickoff (published in the FAQ) are set to `D`.
Ungraded frames are `X`.

## 3. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (players + ball) | **used** — the only input to the labels |
| Smart Tagging | **dropped** — tried as a light team prior, but the external games carry no Smart Tagging, so its weight could not be tuned on external data as the rules require |
| homography | not needed: the ATD is already in pitch coordinates |
| broadcast ATD / homography | not used: not synchronised to the tactical timeline |
| video | not used |

## 4. Training / tuning data

| dataset | source | labelled by | amount |
|---|---|---|---|
| Metrica Sports open sample games 1–3 | https://github.com/metrica-sports/sample-data | Metrica Sports (event data); frame labels derived from it by our code | 3 matches (≈ 4.8 h of tracking at 25 fps): used for leave-one-game-out validation, then all three for the final fit |

No data about BAR-ATH other than the provided automatic assets is used, and nothing was labelled
by hand.

**Frame labels from events** (`solution/external.py`). Mirroring the definition in
CHALLENGE.md: alive frames take the team of the most recent possession event (pass, carry, set
piece, shot, recovery); a loose ball stays with the last team; play is dead from a stoppage
(ball out, foul, offside, goal, end of half) until the next set piece, or until the next
possession event when a quick restart was not logged as a set piece.

**Making the external games look like the competition data.** Player identities are discarded
(only team membership is kept) and the ball is used exactly as tracked. The external games have no
speed channel, so player speeds are computed from positions; the competition ATD's own speeds are
used for BAR-ATH. The ball's absence rate (32–41% against 41%) and the distribution of mean player
speed (median 1.52–1.56 m/s against 1.52 m/s) match closely, which is why the models transfer.
Every feature was also checked for the same meaning in both sources (see "Removed for not
transferring" above).

**Training** (`solution/learned.py`): both logistic regressions are fitted by Newton's method with
a ridge penalty on standardised features. Feature and weighting choices are compared by
leave-one-game-out cross-validation (`--cv`); the submitted model is fitted on all three games
(`--final`).

**Tuning** (`solution/tune.py --learned`): coordinate search over a grid per parameter (logit
scales and bias, the three change costs, the minimum dead-ball length), maximising mean `spellF1`
with the official metric code — on games 1 and 2 while estimating performance, and on all three
games (`--final`) for the submitted parameters.

## 5. Possession rules implemented

- **Control** = a team's player close to and moving with the ball, sustained long enough to
  outweigh the change cost; a single touch or deflection by the other team does not change
  possession.
- **Loose balls / contested balls**: no control evidence either way, so possession stays with the
  last team (the decoder has no reason to switch).
- **Restarts**: dead ball ends when the ball reappears and is played; the restarting team gets the
  ball from the control evidence of whoever plays it.
- **Dead ball**: ball out of play is seen mainly as the tracker losing the ball for a sustained
  period (on the external games 92–97% of missing-ball frames are dead), plus a resting ball and
  low collective speed.

## 6. Limitations and what we would do next

- The external labels come from event data, while the competition's come from corrected tracking:
  boundaries of dead-ball stretches in particular may be defined slightly differently.
- Dead-ball detection relies heavily on the tracker dropping the ball during stoppages; a stoppage
  where the ball stays tracked (a quick free kick) is harder.
- Aerial balls: without ball height, a high ball passing over a player looks like control.
- Three external matches is little training data; more labelled matches (any provider) would let
  the evidence models use more features without overfitting.
- Next steps: more external training matches; using the video for ball height and whistle
  detection; Smart Tagging as a prior if external tagged data exists.

## 7. Compliance with the rules

- **No hand labels, no human-produced data about BAR-ATH.** Every label comes out of the code; the
  only human-produced data used is Metrica's event data for three *other* matches.
- **No training, fitting or tuning on the ground truth or the video.** All models are trained and
  all parameters tuned on the external matches only. The video is not used at all. Early
  prototypes were scored against the public 5-minute window as a pipeline check before the
  organisers' clarification of the rules; none of their parameters survive — every current
  parameter comes from the external training and tuning described above.
- **Automatic checks on the provided data** are limited to label-free summary statistics (feature
  averages, possession and stoppage counts), used to detect features that do not transfer.
- **Reproducible**: the repository regenerates the submitted file byte-for-byte (verified from a
  fresh clone).

## 8. Reproduction

```bash
pip install -r requirements-solution.txt
export MTAH_DATA_DIR=/path/to/competition/data
export MTAH_EXTERNAL_DIR=/path/to/metrica-sample      # only for tuning / external scores
python solution/learned.py --cv      # optional: leave-one-game-out scores
python solution/learned.py --final   # optional: refits solution/model.json on games 1-3
python solution/tune.py --learned --final   # optional: regenerates solution/params.json
python solution/eval_external.py     # external scores in the table above
python solution/run.py --out <team_name>.csv
python evaluation/validate_submission.py <team_name>.csv
```

`run.py` is deterministic and writes `\n` line endings, so the file is byte-identical across runs
and machines. Run times on a laptop: `run.py` about 30 s (the first run also caches the parsed
tracking); `learned.py --final` about 1 min; `tune.py --learned --final` 10–15 min.
