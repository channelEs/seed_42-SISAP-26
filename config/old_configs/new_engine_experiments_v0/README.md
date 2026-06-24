# Additional recall experiments

These configs are meant to be run after the current baseline on the same dataset.

## What to change in the implementation

The search path already supports two knobs that matter most for the next runs:

- `heap_factor`: controls how aggressively the upper-bound pruning skips blocks.
- `md`: controls the hard cap on how many documents are examined during search.

For the next experiments, keep `k=2000` and `itr=25`, because those are already the best trade-off from the current runs.

## Recommended run order

1. config_exp_F_missing_link.json
   - `nb=500`, `nd=200`, `heap_factor=0.20`, `md=60000`
   - This is the highest-value single run because it combines the best breadth and the best pruning softness while removing the hard search cap.

2. config_exp_G_trim_300_120_0_20.json
   - `nb=300`, `nd=120`, `heap_factor=0.20`, `md=50000`
   - This checks whether the full `nb=500` breadth is really necessary.

3. config_exp_H_tighten_500_200_0_25.json
   - `nb=500`, `nd=200`, `heap_factor=0.25`, `md=50000`
   - This tests whether a slightly tighter pruning threshold can keep latency lower while preserving most of the recall gain.

4. config_exp_I_middle_300_120_0_25.json
   - `nb=300`, `nd=120`, `heap_factor=0.25`, `md=50000`
   - This is the most likely "knee" candidate if you want a faster search path.

## What to watch in the logs

After each run, compare:

- Recall@30
- Avg_Time_Per_Query_ms
- Avg_Docs_Examined

Interpretation:

- If Recall@30 jumps above 0.88 with `Avg_Docs_Examined` still comfortably below `md`, you are on the right track.
- If Recall@30 is still below 0.85, the bottleneck is likely the index breadth or the current summaries, not just the pruning threshold.
- If `Avg_Docs_Examined` is close to `md`, then the hard cap is still limiting the search and you should keep `md` high.

## Short guidance

If you want the minimum number of runs to try first, run only:

1. config_exp_F_missing_link.json
2. config_exp_H_tighten_500_200_0_25.json

Those two runs will tell you whether the recall gain from the wider index is worth the extra search time.
