# scripts

Shared environment / build entry points. Module-specific run scripts
(e.g. `profiler/profile.sh`, `bench/bench.sh`, `workloads/examples/*.sh`)
live with their module — only setup and build helpers are here.

## Files

| File | Purpose |
| --- | --- |
| `docker-vllm.sh`  | Launch the vLLM Docker container (profiler + bench + workloads.generators). Mounts repo root as `/workspace`, uses official `vllm/vllm-openai:v0.19.0` image, and pre-installs `datasets` + `matplotlib` on first start. |
| `docker-sim.sh`   | Launch the simulator Docker container (ASTRA-Sim + sim Python deps). |
| `install-vllm.sh` | Bare-metal vLLM install via `uv venv` for environments without Docker. Brings in vLLM 0.19.0 plus `datasets` and `matplotlib`. |
| `compile.sh`      | Build ASTRA-Sim's analytical backend and install the Chakra trace converter. |
| `run-qwen3-lpddr-sweep.py` | Generate and run Qwen3-32B dense/sparse/LPDDR simulator sweep configs, then write `summary.csv`. |

## Typical first-time setup

Inside Docker (recommended):

```bash
./scripts/docker-vllm.sh   # for profiling, benchmarking, dataset generation
./scripts/docker-sim.sh    # for simulation
./scripts/compile.sh       # one-time ASTRA-Sim + Chakra build (inside docker-sim)
```

Bare metal (vLLM side only):

```bash
./scripts/install-vllm.sh
```

## Qwen3 LPDDR sparse sweep

Run from the simulator environment after profiling Qwen3-32B for the target
hardware and TP degree:

```bash
python scripts/run-qwen3-lpddr-sweep.py \
  --hardware RTXPRO6000 \
  --num-npus 2 --tp-size 2 \
  --dataset workloads/sharegpt-qwen3-32b-300-sps10.jsonl
```

Useful options:

```bash
python scripts/run-qwen3-lpddr-sweep.py --dry-run
python scripts/run-qwen3-lpddr-sweep.py --summary-only
python scripts/run-qwen3-lpddr-sweep.py --hbm-sizes 44,48,56 --lpddr-sizes 0,64,128 --sparse-ks 128,256
python scripts/run-qwen3-lpddr-sweep.py --jobs 8
```

Outputs go under `outputs/sweep/qwen3_lpddr/` by default. Generated cluster
configs go under `configs/cluster/generated/qwen3_lpddr/`.
When `--jobs` is greater than 1, each simulator run writes its stdout/stderr to
a `.log` file next to that run's request CSV to avoid interleaved terminal logs.

## Editing notes

* `docker-vllm.sh` ships with a placeholder `HF_TOKEN="<your_token>"`.
  Set it to a real HuggingFace token before running so gated configs
  (Llama, etc.) auto-download on first use.
* `VLLM_GPUS=all` is the default. Use `VLLM_GPUS=2 ./scripts/docker-vllm.sh`
  or `VLLM_GPUS=0,1 ./scripts/docker-vllm.sh` to expose specific host GPUs.
* `datasets` and `matplotlib` are no longer installed by default. Use
  `VLLM_INSTALL_EXTRAS=1 ./scripts/docker-vllm.sh` if you need workload
  generation or plotting packages inside the vLLM container.
