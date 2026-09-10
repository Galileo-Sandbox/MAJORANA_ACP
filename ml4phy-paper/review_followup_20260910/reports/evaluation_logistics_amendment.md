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
