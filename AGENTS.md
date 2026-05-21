# AGENTS.md

Agent quick-reference for the IR Project codebase.

---

## Assignment context

Academic IR project (Radboud / I00041).

- **Step 1**: build and evaluate a hybrid neural IR system on Robust04 (TREC Robust 2004)
- **Step 2**: apply and improve the system on the Open Web Index (OWI) collection

The system **must go beyond lexical retrieval** — neural or learning-to-rank is expected (SPLADE, dense retrieval, re-ranking, etc.).

## Outstanding work / known gaps

- **OWI evaluation is not done** — the system has only been evaluated on Robust04. Applying and evaluating the hybrid pipeline on OWI is the primary remaining task (and the main reason for the low method score in grading).
- **No validation split** — all hyperparameter tuning (BM25 k1/b, RM3 params, hybrid weight w) was done directly against the Robust04 test qrels. There is no separate train/val split. Keep this in mind when extending or reporting experiments.

---

## Setup

### Local development

```bash
uv sync          # use this — NOT pip install -r requirements.txt
```

`requirements.txt` pins stale HuggingFace versions (`transformers==4.18.0`, `huggingface_hub==0.5.1`) that are incompatible with the rest of the stack. Trust `pyproject.toml` instead.

Python version: `~=3.10` (per `pyproject.toml`; the README says >=3.12 — that's wrong).

Two packages are installed from git HEAD:
- `splade` → `https://github.com/naver/splade`
- `pyt_splade` → `https://github.com/cmacdonald/pyt_splade`

### Cluster (csedu SLURM)

```bash
# 1. Copy env file and fill in values
cp .env.example .env         # set NFS_USER_DIR and PROJECT_NAME

# 2. Run from cn84 (login node), sets up NFS dirs and ./data symlink
bash scripts/prepare_cluster.sh

# 3. Create venv on the node (venv lives in /scratch/$USER/… for performance)
bash scripts/setup_virtual_environment.sh

# 4. Sync venv to other nodes (run from cn84)
bash scripts/sync_csedu.sh
```

Cluster topology:
- Login nodes: `cn84`, `cn99`
- GPU nodes: `cn47`, `cn48`
- CPU nodes: `cn77`, `cn78`
- SLURM account: `csedui00041`, partition: `csedu`

---

## Data directory

`./data/` is a **symlink** created by `prepare_cluster.sh` pointing to NFS storage. It does not exist locally until you create it or run the prepare script.

All experiment and index-builder scripts use `--data-dir ./data` by default.

Exception: `ir_system.py` defaults to `./robust04_bm25_index` and `./robust04_splade_index` at the repo root (the locally-built indices committed alongside the code).

---

## SPLADE model

Scripts look for a local copy first, then fall back to HuggingFace:

```
./models/splade-cocondenser-ensembledistil/   ← checked first
naver/splade-cocondenser-ensembledistil       ← HuggingFace fallback
```

On cluster nodes without internet (`HF_HUB_OFFLINE=1` is set in all sbatch scripts), the model **must** be present at the local path. Download it once on a node with internet access and place it there.

`main.py` has a hardcoded Windows path (`C:\splade_model`) for non-MPS devices and is not a usable entrypoint — use `ir_system.py` or the experiment runners instead.

---

## OWI dataset

OWI is **not public** and only accessible on the csedu cluster:

```
/vol/csedu-nobackup/course/I00041_informationretrieval/shared/owi
```

You must call `ir_datasets_owi.register()` before loading any `owi/*` dataset. Both experiment runners and index builders do this automatically when `--dataset owi` or `--dataset owi/subsampled` is passed.

OWI document fields differ from Robust04:

| Dataset  | Text fields             | Query field |
|----------|-------------------------|-------------|
| robust04 | `title`, `body`         | `title`     |
| owi      | `title`, `main_content` | `text`      |

---

## Building indices

Must be done once before running experiments. Indices go under `./data/`.

### Local / single machine

```bash
# Both BM25 + SPLADE
python build_indices.py --dataset robust04

# OWI (subsampled, faster)
python build_indices.py --dataset owi/subsampled

# Flags
--bm25-only | --splade-only
--force          # rebuild even if index exists
--batch-size N   # SPLADE batch size (default 64; increase for more VRAM)
--device cuda|mps|cpu   # auto-detected if omitted
```

### Cluster (SLURM sbatch)

Robust04 (single job, GPU for SPLADE):
```bash
sbatch build_indices.sh                       # both indices
sbatch build_indices.sh --splade-only
sbatch build_indices_bm25.sh                  # CPU-only nodes
```

OWI full dataset (sharded, 30 SLURM array jobs):
```bash
sbatch build_indices_bm25_sharding.sh         # --array=0-29, dataset=owi hardcoded
sbatch build_indices_splade_sharding.sh       # --array=0-29
```

Sharded indices land at `./data/<prefix>_{bm25,splade}_index/part_0/`, `part_1/`, …

---

## Running experiments

### Local

```bash
python run_experiments_optimised.py --exp 20        # single experiment
python run_experiments_optimised.py --exp 1-5       # range
python run_experiments_optimised.py --list          # list all 20 experiments
python run_experiments_optimised.py                 # all experiments

# OWI dataset
python run_experiments_optimised.py --dataset owi --exp 1 20

# Cache management
python run_experiments_optimised.py --clear-cache   # wipe disk cache before run
```

### Cluster

```bash
sbatch run_experiments_optimised.sh --exp 20
sbatch run_experiments_optimised.sh --dataset owi --exp 1-5
```

> **Note:** `run_experiments_optimised.sh` line 32 has leftover vim undo markers at the end — do not copy the last line literally when modifying that file.

### Caching behaviour

`run_experiments_optimised.py` caches retrieval results to `./data/retrieval_cache/<prefix>/*.pkl` (memory + disk). Re-running experiments reuses cached results. `run_experiments.py` has no cache and reruns everything from scratch each time.

---

## Index structure: sharded vs single

Both runners auto-detect sharding in `load_sharded_index()`:
- Sharded: looks for `part_0/data.properties`, `part_1/data.properties`, … up to 49
- Single: flat `data.properties` directly under the index directory

**Limitation**: RM3 and Bo1 query expansion use only `_bm25_index_ref` (the first shard). Multi-shard expansion is not supported.

---

## Best model configuration

Determined by Experiment 20 (evaluated on Robust04):

- **SPLADE**: `naver/splade-cocondenser-ensembledistil`, `max_length=256`
- **BM25**: k1=0.9, b=0.4
- **RM3**: fb_docs=10, fb_terms=15, fb_lambda=0.5
- **Fusion**: `SPLADE_score + 20 × BM25_RM3_score`

`ir_system.py` implements this as the default `IRSystem` configuration.

---

## Key files

| File | Purpose |
|------|---------|
| `ir_system.py` | Clean `IRSystem` class for the best hybrid pipeline; defaults to local repo-root indices |
| `run_experiments_optimised.py` | Preferred experiment runner (caching, 20 experiments) |
| `run_experiments.py` | Original runner (no cache; same 20 experiments) |
| `build_indices.py` | Single-machine index builder (robust04 + OWI) |
| `build_indices_sharding.py` | Sharded index builder for cluster array jobs |
| `ir_datasets_owi.py` | Registers OWI dataset via DuckDB parquet reader; must be called before any `owi/*` load |

---

## No automated tests or linting

There is no pytest suite, no CI, no pre-commit, and no linter/formatter configured. `ir-tests.ipynb` is an exploratory notebook only.
