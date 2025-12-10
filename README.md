# IR Project

Implementation of the Build an IR system from Scratch project.

## Requirements

- Python >= 3.12

## Installation

### Using uv (recommended)

```bash
uv sync
```

### Using pip / Anaconda

The splade packages require `transformers==4.18.0`, but this project needs `transformers>=4.30.0`. 
To override this, install in two steps:

```bash
# Step 1: Install main dependencies (includes overridden transformers/tokenizers versions)
pip install -r requirements.txt

# Step 2: Install splade packages WITHOUT their dependencies (to avoid version conflicts)
pip install --no-deps git+https://github.com/naver/splade.git git+https://github.com/cmacdonald/pyt_splade.git
```

> **Note:** The `--no-deps` flag prevents pip from installing splade's strict `transformers==4.18.0` 
> requirement, allowing the newer version from step 1 to be used instead.
