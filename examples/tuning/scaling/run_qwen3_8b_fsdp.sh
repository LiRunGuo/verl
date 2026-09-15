#!/usr/bin/env bash
# GRPO | text | vLLM rollout | FSDP training | NVIDIA GPUs (80GB class)
# Resource reference for Qwen3-8B, validated on NVIDIA A800-80GB (PCIe).
# Same task as examples/grpo_trainer/run_qwen3_8b_fsdp.sh (GRPO + GSM8K/MATH),
# but the memory/throughput knobs are grouped into two named profiles:
#
#   TUNING_PROFILE=min          smallest footprint: every memory optimization
#                               on (gradient checkpointing, activation offload,
#                               FSDP param + optimizer offload), no deferred
#                               gradient sync, a tight dynamic-batch token
#                               budget and a small vLLM memory fraction.
#   TUNING_PROFILE=recommended  throughput: offloads off, deferred gradient
#                               sync on, twice the token budget and a vLLM
#                               memory fraction sized to leave the trainer room.
#
# Measured numbers for both profiles and the A800 hardware notes live in
# examples/tuning/README.md.
#
# Override any knob below with the matching environment variable, e.g.
#   TUNING_PROFILE=min NGPUS_PER_NODE=8 bash examples/tuning/scaling/run_qwen3_8b_fsdp.sh

set -xeuo pipefail

TUNING_PROFILE=${TUNING_PROFILE:-recommended}

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}

max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
# vLLM sizes its KV pool so that at least one sequence of ``max_model_len``
# fits, and defaults that to the model's full context (40960 for Qwen3-8B).
# Pinning it to what the trainer actually samples is a memory optimization with
# no throughput cost here; at ``min`` rates the default does not fit at all.
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}
rollout_n=${ROLLOUT_N:-5}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_grpo_tuning_qwen3_8b}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_${TUNING_PROFILE}_vllm_fsdp}
# ---- end user-adjustable ----

# ---- profile defaults (every value is env-overridable) ----
case "${TUNING_PROFILE}" in
    min)
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
        train_batch_size=${TRAIN_BATCH_SIZE:-32}
        ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
        ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
        gradient_checkpointing=${GRADIENT_CHECKPOINTING:-True}
        activation_offload=${ACTIVATION_OFFLOAD:-True}
        actor_param_offload=${ACTOR_PARAM_OFFLOAD:-True}
        actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-True}
        # Deferring the gradient reduce-scatter keeps unsharded gradients alive
        # across the whole mini-batch. That is the throughput default, but the
        # min profile cannot afford it.
        no_sync_for_grad_accum=${NO_SYNC_FOR_GRAD_ACCUM:-False}
        rollout_tp=${ROLLOUT_TP:-1}
        # vLLM shares the cards with the trainer, so it only gets a small slice.
        # Anything above ~0.3 asks for more free HBM than is left after the
        # actor engine is built, and the vLLM core refuses to start.
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.25}
        ;;
    recommended)
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
        train_batch_size=${TRAIN_BATCH_SIZE:-64}
        ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
        # The stored activations scale with this budget, so it is the first knob
        # to move when the actor update starts OOMing.
        ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
        gradient_checkpointing=${GRADIENT_CHECKPOINTING:-True}
        activation_offload=${ACTIVATION_OFFLOAD:-False}
        actor_param_offload=${ACTOR_PARAM_OFFLOAD:-False}
        actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-False}
        no_sync_for_grad_accum=${NO_SYNC_FOR_GRAD_ACCUM:-True}
        # TP=2 keeps each vLLM replica inside one NVLink pair; the two pairs of
        # an A800 PCIe node are only joined by PIX/SYS links. See the hardware
        # notes in examples/tuning/README.md.
        rollout_tp=${ROLLOUT_TP:-2}
        # 0.6 (what the 8-GPU canonical script uses) OOMs here: the model is
        # sharded over 4 cards instead of 8, so there is twice as much training
        # state per card competing with vLLM's reservation.
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
        ;;
    *)
        echo "TUNING_PROFILE must be one of: min, recommended (got: ${TUNING_PROFILE})" >&2
        exit 1
        ;;
esac

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet

train_files="['$gsm8k_train', '$math_train']"
val_files="['$gsm8k_test', '$math_test']"
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=${gradient_checkpointing}
    actor_rollout_ref.model.enable_activation_offload=${activation_offload}
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=${no_sync_for_grad_accum}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
)

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the vllm × fsdp extras of the committed uv.lock;
# NPU falls back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
