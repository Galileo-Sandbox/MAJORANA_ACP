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
