# HuggingFace XET Rate Limits

## The problem

HuggingFace datasets backed by XET storage (the chunked CDN introduced in 2025)
hit a **per-IP rate limit of 500 requests / 5 minutes** on the
`xet-read-token` endpoint. This limit is applied **before** auth checks —
`HF_TOKEN` does not raise the ceiling for XET token requests.

For LeRobot datasets stored one-parquet-per-episode, this translates directly
to an episode count limit:

| Dataset              | Files | Behavior                             |
| -------------------- | ----- | ------------------------------------ |
| `lerobot/pusht`      | 418   | Usually works (just under ceiling)   |
| `pusht-synthetic-v3` | 20005 | Fails at file 500 (~1 min into load) |

Symptoms on a Vast.ai instance (shared NAT IP):

```
Fetching 20005 files: 2% | 500/20005 [1:01<...]
HfHubHTTPError: 429 Client Error: Too Many Requests
  https://huggingface.co/api/datasets/.../xet-read-token/...
  "We had to rate limit your IP (91.183.4.218)"
```

## Why HF_TOKEN doesn't help

The general Hub API bucket grants 10k req/5min to authenticated users. The
`xet-read-token` endpoint applies its own per-IP bucket that runs
independent of Bearer auth. Authenticated users on Vast's shared NAT IPs
share the same 500-req bucket with every other anonymous tenant on that
gateway.

## The fix: pre-stage the cache

The solution is to **avoid downloading from HF at job runtime**. Instead,
populate the HF cache before the training container starts. There are
several ways to do this:

### Option 1: sparkinstance `hf_cache_source` (recommended)

Configure your job to rsync the HF cache from a source machine to the
Vast instance at job startup, before the training container runs:

```yaml
# sparkinstance job config
hf_cache_source:
  type: rsync
  source: "neil@workstation.local:~/.cache/huggingface/hub/"
  repos:
    - "datasets--neiltan--pusht-synthetic-v3"
    - "datasets--lerobot--pusht"
  skip_if_present: true   # skip rsync if cache is already warm
  timeout_s: 600
```

The source machine needs:
- SSH reachability from Vast
- A warm HF cache (download the dataset once on a machine that doesn't
  hit the rate limit)
- Free disk for the cached files

### Option 2: Persistent HF_HOME volume on Vast

If the Vast host supports persistent storage across runs, mount it at
`HF_HOME` so subsequent jobs on the same host skip the download:

```bash
# In the Docker entrypoint
export HF_HOME=/persistent/hf_cache
```

### Option 3: Bake the dataset into the Docker image

For small, stable datasets (e.g., `lerobot/pusht` at 50 MB):

```dockerfile
# Dockerfile
ARG HF_TOKEN
RUN HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    lerobot/pusht --repo-type dataset \
    --local-dir /root/.cache/huggingface/hub/datasets--lerobot--pusht
```

Good for datasets that don't change often. Not ideal for large or frequently
updated datasets.

## DataPorter's role

DataPorter detects an empty HF cache at startup and logs a loud warning:

```
HF cache empty for neiltan/pusht-synthetic-v3 (cache dir does not exist:
/root/.cache/huggingface/hub/datasets--neiltan--pusht-synthetic-v3).
Runtime download may hit HF XET 500-req/5min per-IP rate limit for
repos with >500 files. Recommended: pre-sync the cache via
`rsync -az <source>:/root/.cache/huggingface/hub/ /root/.cache/huggingface/hub/`
or configure sparkinstance hf_cache_source in the job config.
```

After a successful dataset load, DataPorter writes a sentinel file
(`.dataporter_cache_complete`) in the cache directory. sparkinstance's
`skip_if_present` check looks for this file to decide whether to re-run
the rsync step on subsequent job starts.

## Helper API

DataPorter exposes three helpers for cache pre-flight logic:

```python
from dataporter import (
    check_hf_cache_populated,
    hf_cache_repo_path,
    write_cache_sentinel,
)

# Check if a dataset is in the HF cache
populated, reason = check_hf_cache_populated("lerobot/pusht")
if not populated:
    print(f"Cache is empty: {reason}")

# Get the canonical HF cache path for a repo
path = hf_cache_repo_path("lerobot/pusht")
# → PosixPath('~/.cache/huggingface/hub/datasets--lerobot--pusht')

# Write the sentinel after a successful load
write_cache_sentinel(path)
```

## What DataPorter does NOT do

- **Client-side rate limiting**: would cap throughput at the anonymous
  500/5min bucket, turning a 2-minute download into a 3+ hour one. Better
  to pre-stage the cache and avoid the rate limit entirely.
- **Re-uploading datasets with fewer files**: LeRobot's file layout
  (one parquet per episode) is baked into the library. Changing it
  requires a custom loader fork, which isn't worth it.
- **HF_TOKEN enforcement**: on Vast the token doesn't help for XET
  token requests, so requiring it would be misleading.
