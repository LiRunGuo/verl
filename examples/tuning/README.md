# Tuning Examples

Examples that focus on *tuning-related* features rather than on a specific RL algorithm. Everything in here trains with `verl.trainer.main_ppo` and the current Hydra API.

## Subdirectories

### `lora/` — LoRA fine-tuning

Canonical LoRA GRPO scripts (training only adapters, rollout still serves the adapter via `load_format=safetensors + layered_summon`).

| Script                                         | Model             | Infer | Train    | Notes                       |
|------------------------------------------------|-------------------|-------|----------|-----------------------------|
| `run_qwen3_8b_fsdp.sh`               | Qwen3-8B          | vLLM  | FSDP     | text, GSM8K                 |
| `run_qwen3_8b_from_adapter_fsdp.sh`  | Qwen3-8B          | vLLM  | FSDP     | start from existing adapter |
| `run_qwen3_8b_merge_fsdp.sh`         | Qwen3-8B          | vLLM  | FSDP     | merge adapter into base     |
| `run_qwen2_5_vl_7b_fsdp.sh`          | Qwen2.5-VL-7B     | vLLM  | FSDP     | vision, Geo3K               |
| `run_qwen3_30b_a3b_megatron.sh`      | Qwen3-30B-A3B     | vLLM  | Megatron | MoE                         |

Key flags:
- `actor_rollout_ref.model.lora_rank`, `actor_rollout_ref.model.lora_alpha`
- `actor_rollout_ref.rollout.load_format=safetensors`
- `actor_rollout_ref.rollout.layered_summon=True`

### `scaling/` — Large-model scale demos

Single/multi-node tuning recipes for large dense models; geared to practitioners trying to fit and run these models out of the box with GRPO + GSM8K/MATH.

| Script                                    | Model           | Infer | Train    | Hardware                 |
|-------------------------------------------|-----------------|-------|----------|--------------------------|
| `run_qwen2_5_32b_megatron.sh`        | Qwen2.5-32B     | vLLM  | Megatron | 1×8 GPUs (TP=8)          |
| `run_qwen2_5_72b_fsdp.sh`            | Qwen2.5-72B     | vLLM  | FSDP     | 4×8 GPUs (TP=16, offload)|
| `run_qwen3_8b_fsdp.sh`               | Qwen3-8B        | vLLM  | FSDP     | 1 node, 4-8×A800-80GB    |

#### `run_qwen3_8b_fsdp.sh` — min / recommended profiles

This is the script behind the A800 rows of
[`docs/perf/device_tuning.rst`](../../docs/perf/device_tuning.rst). It carries
both reference points that page asks for behind one `TUNING_PROFILE` toggle, so
the same GRPO + GSM8K/MATH workload can be run memory-lean or throughput-lean
without a second script:

| `TUNING_PROFILE` | GPUs       | gradient ckpt | activation offload | FSDP offload       | defer grad sync | vLLM mem | rollout TP | train batch | `ppo_max_token_len_per_gpu` |
|------------------|------------|---------------|--------------------|--------------------|-----------------|----------|------------|-------------|-----------------------------|
| `min`            | 4×A800-80GB | on           | on                 | param + optimizer  | off             | 0.25     | 1          | 32          | 4096                        |
| `min`            | 8×A800-80GB | on           | on                 | param + optimizer  | off             | 0.25     | 1          | 64          | 4096                        |
| `recommended`    | 4×A800-80GB | on           | off                | none               | on              | 0.40     | 2          | 64          | 8192                        |
| `recommended`    | 8×A800-80GB | on           | off                | none               | on              | 0.40     | 2          | 128         | 8192                        |

Measured on A800-80GB (PCIe), Qwen3-8B, 1K prompt / 2K response,
`rollout.n=5`, three steps per configuration, with the startup validation pass
skipped so the numbers are training-only. Per-step times are `step 2 / step 3`;
the phase columns are step 3, and `refit` is the FSDP → vLLM weight sync. The
four-card rows were re-measured later on an otherwise idle node and reproduce
within ~2% on step time and per-GPU throughput with identical peak memory, so a
co-tenant on the other socket of the node did not distort them.

| Config                | step (s)      | gen (s) | old_log_prob (s) | ref (s) | update_actor (s) | refit (s) | tokens/step | tok/s/GPU | peak alloc (GB) | peak reserved (GB) | MFU  |
|-----------------------|---------------|---------|------------------|---------|------------------|-----------|-------------|-----------|-----------------|--------------------|------|
| `min`, 4×A800         | 156.8 / 147.0 | 48.5    | 11.5             | 29.0    | 50.1             | 7.4       | 236-244K    | 389-402   | 41.0            | 56.4               | 0.20 |
| `min`, 8×A800         | 218.2 / 252.4 | 53.0    | 23.2             | 40.1    | 127.9            | 7.5       | 494-572K    | 232-283   | 25.1            | 34.5               | 0.10 |
| `recommended`, 4×A800 | 144.7 / 165.5 | 42.5    | 18.0             | 35.4    | 65.0             | 4.0       | 490-577K    | 846-872   | 49.5            | 69.4               | 0.37 |
| `recommended`, 8×A800 | 172.7 / 165.5 | 41.4    | 17.0             | 30.2    | 66.2             | 9.8       | 1015-1127K  | 796-816   | 38.2            | 50.8               | 0.33 |

Three things fall out of the table:

- **In the `min` profile, more cards buy memory rather than throughput.**
  Doubling from 4 to 8 cards cuts the peak allocation from 41.0 GB to 25.1 GB,
  but per-GPU throughput falls from 389-402 tok/s to 232-283 tok/s and MFU
  halves (0.20 → 0.10). Host offload plus per-micro-batch gradient
  synchronization stop amortizing once each shard is small: `update_actor`
  takes 128 s for 572K tokens on 8 cards against 50 s for 236K tokens on 4.
- **The `recommended` profile scales.** It holds 796-816 tok/s/GPU on 8 cards
  versus 846-872 on 4 (-6%) while doubling the batch, so the extra cards turn
  into aggregate throughput — ~6.5K tok/s per node against ~3.5K on 4 cards.
- **Weight refit grows with the world size.** The same reshard costs 4.0 s on 4
  cards and 9.8 s on 8, i.e. 6% of an 8-card step versus 2.5% of a 4-card one.
  It is the one phase that gets *worse* as ranks are added, so it is worth
  watching when scaling out.

Between the profiles, neither dominates: on 4 cards `min` trades ~2.2× of
per-GPU token throughput for an 8.5 GB lower peak, and `recommended` keeps
parameter and optimizer state resident, which is why its vLLM fraction is 0.40
rather than the canonical script's 0.60.

#### A800-80GB notes

These are the hardware facts that shaped the profile defaults above.

- **Eight cards, four NVLink pairs over two sockets — not one fabric.**
  `nvidia-smi topo -m` reports `NV8` only *inside* the pairs (0-1, 2-3, 4-5,
  6-7); the remaining hops inside a socket are `PIX`, and 0-3 to 4-7 crosses to
  the second socket over `SYS`. The gap is large in practice: an NCCL
  all-reduce of 64 MiB fp16 runs at **127 GB/s** across GPUs 0-1 but only
  **37 GB/s** algorithm bandwidth (55.7 GB/s bus) across four, a ~3.4× drop.
  Keep tensor-parallel groups inside a pair — hence `ROLLOUT_TP=2`, not 4 or 8,
  in the profiles.
- **The validation pass is not free.** `data.val_files` here is GSM8K test
  (1319 prompts) plus MATH test (5000), and `trainer.val_before_train=True`
  generates for all 6319 of them before the first step. On 4×A800 that is
  ~14 min — more than a training step. Use `trainer.val_before_train=False`
  while tuning the training knobs; it does not affect the recipe.
- **vLLM reserves KV for the model's full context.** With
  `rollout.max_model_len` unset, vLLM defaults it to Qwen3-8B's 40960 positions
  and refuses to start if the remaining KV budget cannot hold one such
  sequence. The scripts pin `max_model_len` to
  `max_prompt_length + max_response_length`.
- **Four cards is the floor for full-parameter GRPO at 8B.** Two-card runs OOM
  inside `update_actor` at ~78 GB even with gradient checkpointing, activation
  offload, FSDP param + optimizer offload and `use_no_sync_for_gradient_accumulation=False`
  all enabled: sharded fp32 Adam state plus the frozen reference model does not
  fit twice over. Eight-GPU values in `examples/grpo_trainer/` are not a
  statement that four are enough.
- **Qwen3-8B thinking truncates.** At `max_response_length=2048`, 38-59% of
  responses hit the cap in these runs. Raise the response budget (and the token
  budgets with it) if the task needs the full reasoning trace.

#### Rollout TP on a full node

The topology note above shows up end to end. Switching only the rollout TP of
the eight-card `recommended` configuration (batch 128) gives:

| rollout TP        | step 2 / step 3 (s) | gen (s) | update_actor (s) | tok/s/GPU | peak alloc (GB) |
|-------------------|---------------------|---------|------------------|-----------|-----------------|
| 2 (inside a pair) | 172.7 / 165.5       | 41.4    | 66.2             | 816 / 796 | 38.2            |
| 4 (crosses pairs) | 197.9 / 184.8       | 61.2    | 66.2             | 709 / 704 | 38.0            |

The training phases are identical to the tenth of a second (`update_actor`
66.2 s, `ref` 30.0 vs 30.2 s) because FSDP spans all eight cards either way; the
whole ~14% end-to-end gap is the 48% slower rollout generation.

## Conventions

- All scripts expose `MODEL_PATH`, `NNODES`, `NGPUS_PER_NODE`, batch sizes, learning rates, `ROLLOUT_TP`, `ROLLOUT_N`, etc. via `VAR=${VAR:-default}`.
- Dynamic batch size and `trainer.balance_batch=True` are enabled by default.
- No deprecated knobs (`ppo_micro_batch_size`, `data.val_batch_size`, top-level `reward_model.*`, `actor.ulysses_sequence_parallel_size`, `ppo_megatron_trainer.yaml`).
