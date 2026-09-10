# Evaluation logistics amendment

The first `global_gate`, seed-0 pilot attempt stopped before any training step
because the runner tried to mutate a frozen Pydantic configuration. The sole
authorized retry completed under the uniquely named `attempt2` run directory.

Before evaluation, the runner was amended to resolve each successful training
run from the campaign record instead of constructing a directory name. This
prevents the failed, empty first attempt from being mistaken for the completed
checkpoint. The evaluator now also verifies the frozen input, role-archive,
checkpoint, and unchanged control-model implementation hashes before model
calls. No scientific setting, model parameterization, checkpoint, context,
target, random seed, or evaluation statistic changed.

The campaign record retains the training source commit and records the
evaluation source commit and runner hash separately.

The first evaluation launch used the allowed maximum of four workers. Each
process required about 8.8 GiB of device memory during the large target/MC
operation, so four workers exceeded the 32.6-GiB device and produced CUDA OOM
failures. The parent was stopped to prevent queued work from continuing. Fully
written cells are recovered only after validating their summaries and output
hashes. The sole bounded retry uses three workers, which preserves every
scientific and numerical setting while remaining below measured device memory.

Post-evaluation mechanism-map validation found that the checkpoint loader first
loaded the learned global-gate scalar and then reset it while applying the mode
wrapper. Training checkpoints retained the learned values (cutoffs about
3.88--4.10), but 90 evaluations for modes with global gates used the initial
cutoff 3. The 30 `global_attention` evaluations and 30 reused full-model cells
are unaffected. The invalid evaluations are retained server-side and excluded;
a restoration regression test was added before a uniquely named rerun of only
the 90 affected cells. No score from the invalid evaluations is reported.
