#!/bin/bash

# Launch vLLM Docker for profiler / bench / validate.
#
# Mounts the LLMServingSim repo root as /workspace so the profiler,
# bench, datasets generators, and shared model configs are all visible:
#
#     /workspace/profiler/            profiler package + scripts
#     /workspace/bench/               bench + validate
#     /workspace/workloads/            workload JSONLs and generators
#     /workspace/configs/model/       HF model configs
#
# The working directory defaults to /workspace so any of the modules
# can be run via ``python -m profiler``, ``python -m bench``, etc.
#
# The official vllm/vllm-openai image already provides vllm, pydantic,
# pyyaml, rich, and huggingface_hub — no extra pip installs required.

set -euo pipefail

# Resolve the repo root regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim

# GPU selection:
#   ./scripts/docker-vllm.sh                 -> expose all GPUs
#   VLLM_GPUS=2 ./scripts/docker-vllm.sh     -> expose only host GPU 2
#   VLLM_GPUS=0,1 ./scripts/docker-vllm.sh   -> expose host GPUs 0 and 1
VLLM_GPUS="${VLLM_GPUS:-all}"
if [[ "$VLLM_GPUS" == "all" ]]; then
  GPU_ARG="all"
else
  GPU_ARG="device=${VLLM_GPUS}"
fi

# datasets/matplotlib are useful for workload generation and plots, but not
# required for profiling. Keep them opt-in to avoid PyPI/SSL failures on
# restricted servers.
START_CMD="exec bash"
if [[ "${VLLM_INSTALL_EXTRAS:-0}" == "1" ]]; then
  START_CMD="pip install datasets matplotlib && exec bash"
fi

docker run --name vllm_docker \
  --gpus "$GPU_ARG" \
  -it \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  -v "$REPO_ROOT":/workspace \
  --volume "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --shm-size=16g \
  -w /workspace \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.19.0 \
  -c "$START_CMD"
