---
title: LPDDR sparse KV testing manual
sidebar_position: 4
---

# LPDDR sparse KV testing manual

이 문서는 LPDDR sparse KV tiering 구현을 GPU가 있는 머신에서 검증하는 절차다.
현재 개발 머신에 충분한 GPU가 없어도 코드 구현과 문서 작업은 가능하지만,
profiler와 vLLM validation은 GPU 머신에서 수행해야 한다.

## What can be tested without a strong GPU

로컬 CPU-only 또는 약한 GPU 머신에서 가능한 검증:

1. Python syntax/import check.
2. Cluster config parsing.
3. Workload JSONL parsing.
4. Sparse selection metadata가 `Request -> Batch -> trace_generator`로 전달되는지 확인.
5. 기존 committed profiler CSV를 사용하는 작은 simulator smoke test.

제한:

- 새 hardware/model profile을 만들 수 없다.
- vLLM bench validation을 돌릴 수 없다.
- TP=4, 대형 MoE, 긴 context workload는 현실적으로 어렵다.

Host Python에서 직접 simulator를 실행하려면 최소한 다음 패키지가 필요하다.
이 repo에는 현재 `requirements.txt`가 없으므로, 가능하면 아래 Docker simulator
container 안에서 검증하는 것을 우선한다.

```bash
python -m pip install pandas numpy rich
```

로컬에서 `ModuleNotFoundError: No module named 'pandas'` 또는
`ModuleNotFoundError: No module named 'rich'`가 나오면 simulator logic 문제가
아니라 host Python dependency 문제다.

## GPU machine prerequisites

권장 환경:

- Linux host
- NVIDIA GPU with enough HBM for the target model/profile run
- Docker with NVIDIA Container Toolkit
- Git
- Python available on host for small scripts
- HuggingFace token for gated models, if needed

Repository setup:

```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
cd LLMServingSim
```

If testing a local branch:

```bash
git fetch <your-remote>
git checkout <your-branch>
git submodule update --init --recursive
```

## Build simulator container

Start the simulator container:

```bash
./scripts/docker-sim.sh
```

Inside the container:

```bash
./scripts/compile.sh
```

Expected result:

- Chakra converter installs successfully.
- ASTRA-Sim analytical backend binary exists at
  `astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra`.

## Profile the model on the GPU machine

Start the vLLM container:

```bash
./scripts/docker-vllm.sh
```

Set your model and hardware label in `profiler/profile.sh`.

For an initial single-GPU MVP:

```bash
MODEL="meta-llama/Llama-3.1-8B"
HARDWARE="RTXPRO6000"
TP_DEGREES="1"
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=128
ATTENTION_MAX_KV=16384
```

Run:

```bash
./profiler/profile.sh
```

Expected output:

```text
profiler/perf/<hardware>/<model>/<variant>/tp1/dense.csv
profiler/perf/<hardware>/<model>/<variant>/tp1/per_sequence.csv
profiler/perf/<hardware>/<model>/<variant>/tp1/attention.csv
profiler/perf/<hardware>/<model>/<variant>/meta.yaml
```

For MoE models, `moe.csv` should also exist.

For later TP experiments, profile the TP degree you want to simulate:

```bash
HARDWARE=H100 TP_DEGREES=1,2,4 ./profiler/profile-all.sh
```

## Run a sparse LPDDR simulator smoke test

Use the example config:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_lpddr_sparse_instance.json \
  --dtype bfloat16 \
  --block-size 16 \
  --dataset workloads/example_trace.jsonl \
  --output outputs/lpddr_sparse_smoke.csv \
  --num-req 10 \
  --log-level INFO
```

Expected output files:

```text
outputs/lpddr_sparse_smoke.csv
outputs/lpddr_sparse_smoke.sparse.csv
```

The request CSV keeps the existing schema for compatibility with
`bench validate`. The `.sparse.csv` sidecar contains batch-level tiering
metrics:

```text
instance_id
batch_id
effective_attention_k
hbm_hit_blocks
lpddr_hit_blocks
cpu_hit_blocks
hbm_hit_rate
lpddr_to_hbm_promotion_bytes
hbm_to_lpddr_eviction_bytes
promotion_count
eviction_count
copy_time_ns
```

## Verify that sparse K changes attention lookup

Run two simulations with different `sparse_k` values in the cluster config:

```json
"sparse_k": 64
```

and:

```json
"sparse_k": 512
```

Compare:

```bash
python - <<'PY'
import pandas as pd
for path in [
    "outputs/lpddr_sparse_k64.sparse.csv",
    "outputs/lpddr_sparse_k512.sparse.csv",
]:
    df = pd.read_csv(path)
    print(path)
    print(df[["effective_attention_k", "copy_time_ns", "hbm_hit_rate"]].describe())
PY
```

Expected:

- `effective_attention_k` follows the configured `sparse_k`, capped by current context length.
- Smaller `sparse_k` should reduce attention lookup latency relative to full-context decode.
- Copy metrics depend on HBM pressure and placement policy.

## Verify LPDDR promotion/eviction

Sparse tiering does not demote every non-selected HBM block. LPDDR eviction is
triggered only when HBM capacity is insufficient for scheduling or selected-block
promotion. A small workload with enough HBM can legitimately show zero
`hbm_to_lpddr_eviction_bytes`.

To increase LPDDR traffic, reduce `npu_mem.mem_size` in the example config.

Example:

```json
"npu_mem": {
  "mem_size": 16,
  "mem_bw": 1597,
  "mem_latency": 0
}
```

Then rerun:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_lpddr_sparse_instance.json \
  --dtype bfloat16 \
  --block-size 16 \
  --dataset workloads/example_trace.jsonl \
  --output outputs/lpddr_sparse_pressure.csv \
  --num-req 10
```

Inspect:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/lpddr_sparse_pressure.sparse.csv")
print(df[[
    "hbm_hit_rate",
    "lpddr_to_hbm_promotion_bytes",
    "hbm_to_lpddr_eviction_bytes",
    "promotion_count",
    "eviction_count",
    "copy_time_ns",
]].sum(numeric_only=True))
PY
```

Expected:

- `hbm_to_lpddr_eviction_bytes` becomes non-zero only when HBM pressure causes LPDDR spill/eviction.
- Later selected LPDDR blocks cause `lpddr_to_hbm_promotion_bytes` to become non-zero.
- `copy_time_ns` tracks promotion plus eviction copy cost.

## Run CPU offload baseline

Use the same workload and model with an existing CPU/remote-memory config, or disable sparse
attention and LPDDR fields in the same config.

Baseline command:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_single_instance.json \
  --dtype bfloat16 \
  --block-size 16 \
  --dataset workloads/example_trace.jsonl \
  --output outputs/cpu_offload_baseline.csv \
  --num-req 10
```

Compare request-level latency:

```bash
python - <<'PY'
import pandas as pd
for path in [
    "outputs/cpu_offload_baseline.csv",
    "outputs/lpddr_sparse_smoke.csv",
]:
    df = pd.read_csv(path)
    print(path)
    print("TTFT mean ms", df["TTFT"].mean() / 1e6)
    print("TPOT mean ms", df["TPOT"].mean() / 1e6)
    print("Latency mean ms", df["latency"].mean() / 1e6)
PY
```

## Run vLLM validation

Generate or reuse a vLLM bench run:

```bash
python -m bench run \
  --model meta-llama/Llama-3.1-8B \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --output-dir bench/results/lpddr_sparse_baseline \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 2048 \
  --dtype bfloat16 \
  --kv-cache-dtype auto
```

Run simulator against the same workload and engine settings:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_lpddr_sparse_instance.json \
  --dtype bfloat16 \
  --block-size 16 \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --output outputs/lpddr_sparse_sharegpt.csv \
  --num-req 300
```

Validate:

```bash
python -m bench validate \
  bench/results/lpddr_sparse_baseline \
  outputs/lpddr_sparse_sharegpt.csv \
  outputs/lpddr_sparse_sharegpt.log \
  --prefix lpddr_sparse
```

Note: vLLM does not implement this simulated LPDDR sparse KV policy unless you add it to
the runtime. This validation only checks that the simulator remains calibrated for the
non-LPDDR parts. LPDDR speedup claims should be compared against simulator baselines and,
later, a real runtime implementation if available.

## Minimum acceptance checklist

1. `python -m serving` completes with the LPDDR sparse config.
2. The normal request CSV is produced and remains readable by `bench validate`.
3. The `.sparse.csv` sidecar is produced.
4. Lowering HBM capacity increases eviction/promotion metrics.
5. Changing `sparse_k` changes `effective_attention_k`.
6. `copy_time_ns` changes when `hbm_link_bw` or `mem_latency` changes.
7. CPU/offload baseline and LPDDR sparse runs use the same workload and profiler bundle.

## Known MVP limitations

- LPDDR is represented in Python-side memory accounting first; ASTRA-Sim does not yet have
  a distinct LPDDR memory enum.
- Copy cost is represented as one synthetic `lpddr_kv_copy` compute row per batch, not a
  detailed memory-transfer graph.
- Scheduler admission still allocates newly computed KV with the existing full-KV block
  accounting. LPDDR spill is triggered by HBM pressure; it is not yet a full sparse-aware
  admission policy that can accept requests solely because non-selected KV may live in LPDDR.
- Prefix caching should remain disabled for LPDDR sparse KV MVP runs. The example config
  sets `enable_prefix_caching: false` because prefix RadixCache accounting and sparse
  residency moves are not fully unified yet.
- Sparse trace fields are treated as token positions in the request context.
- TP/DP/EP paths need additional validation after the single-instance TP=1 MVP passes.
- Direct LPDDR read mode is not modeled yet; the MVP assumes promote-before-attention.
