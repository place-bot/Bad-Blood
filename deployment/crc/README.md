# Notre Dame CRC execution

Use the [training guide](../../bp_training/README.md) for data preparation, experiment definitions and evaluation. This directory provides an executable single-GPU job entry point and intentionally incomplete scheduler templates. No CRC account access or GPU allocation has been verified from this checkout.

The general CRC cluster documents Grid Engine (`qsub`). CAML documents HTCondor and interactive GPU workflows. Confirm the applicable resource with CRC before filling in queue, GPU, parallel-environment, memory, walltime, storage and container/module directives. Do not use a guessed Slurm script or assume the same directives work on both resources. [CRC quick start](https://docs.crc.nd.edu/new_user/quick_start.html), [CAML](https://docs.crc.nd.edu/resources/caml.html).

1. Obtain an allocated compute session and verify the GPU/driver, project storage and dataset permissions. Do not train on a login node.
2. Provision a working Torch 2.8 CUDA environment. Resolve/pin its actual Linux packages or container digest after the smoke check; the local Mac environment is not a CRC lockfile.
3. Prepare one immutable dataset and audited manifest. Every job must see the same absolute paths. The current wrapper uses shared storage; per-node staging needs a consistent manifest/path design before use.
4. Start with one GPU, four CPU cores and 32 GiB RAM if permitted. These are profiling defaults, not a confirmed allocation. Check worker/thread counts against the scheduler CPU allocation.
5. Run one preflight and one V1 job. Measure a full epoch including validation/checkpoints and peak device memory before estimating GPU-hours or increasing concurrency.
6. Freeze the environment and code revision, then schedule the predeclared seed/config combinations. Save scheduler logs and IDs alongside the private run ledger.

Example environment, with all paths replaced by verified account-specific values:

```bash
export BP_REPO=/project/bp/Bad-Blood
export BP_PYTHON=/project/bp/environment/bin/python
export BP_MANIFEST=/project/bp/manifests/vitaldb-development-v1.json
export BP_RUN_ROOT=/project/bp/runs
export BP_CONFIG=/project/bp/Bad-Blood/configs/model_v2/A_v1.json
export BP_RUN_ID=A-20260923-r1
export BP_SEED=20260923
bash "$BP_REPO/deployment/crc/run_job.sh"
```

Run that wrapper only inside the verified allocation or from the configured scheduler job. It performs preflight then replaces itself with the trainer, allowing termination signals to reach the training loop. It forces CUDA and never automatically submits more jobs. To resume a stopped run, retain the same environment/config/seed/run ID and set `BP_RESUME=1`; see lock recovery and epoch-boundary limitations in the training guide.

`grid_engine.template` exits with an error until copied/configured. `condor.template` has no submission stanza until configured. These guards are deliberate: current account-specific resource names and dataset-transfer rules are unavailable here. Keep filled-in scripts and account paths in your private project storage.

`experiments.json` describes the initial 18 runs and optional nine attention runs. The three-run combined candidate depends on initial findings. The wrapper does not infer results, auto-select a model, tune on held-out data, or reserve unverified resources.
