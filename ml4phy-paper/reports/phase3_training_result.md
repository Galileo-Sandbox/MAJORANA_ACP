# Phase 3 neural training result

Status: all 36 approved jobs completed without scientific failure. The campaign used 1762.40 seconds (29.37 minutes) of controlled wall time under the separate two-hour cap. Four independent CUDA jobs ran concurrently after a four-job measured pilot; the maximum recorded per-process VRAM was 5,876 MiB.

Every job used the frozen 3,000-step, batch-16 schedule. Batch size was not increased because that would change sampled exposure and the optimizer update definition. Parallelism was across independent approved jobs. Checkpoint histories end at step 2,999, all losses are finite, and no training log contains a warning.

The two new 5k subsets retain 4,981 and 4,980 sampling-eligible events. The 10k prefix retains 9,980. Each resolved configuration and reconstructed density buffer uses only its matching subset; none uses the 18,866-event full pool. Checkpoints and event-level training artifacts remain server-only.

These completed models are ready for the frozen 360-cell, context-size-500 evaluation. No performance claim is made from training loss.
