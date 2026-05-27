"""LLM-MMI population-scale inference runner.

Implements Stages 2 (frozen policy ``pi_theta``) and 5 (deterministic
single-pass sampling) of the methodology (Method.md, sections "B.3" and
"B.6").  Stage 4 (the four-layer prompt) is supplied by
:mod:`llm_mmi.prompt`; Stage 6 (Cox PH on the aggregated 17-dim
representation) is implemented in ``eval/ukb.R`` and ``eval/other.R``.

Behaviourally this script preserves the multi-threaded, key-cycling,
breakpoint-resuming pipeline previously expressed as ``Prompts.py``; the
only added knobs are the ablation / backbone hooks driven by environment
variables (see :mod:`llm_mmi.config`).

Run as::

    python -m llm_mmi.inference
    LLM_MMI_PRESET=A1.2 python -m llm_mmi.inference
    LLM_MMI_BACKBONE=gpt-5-nano python -m llm_mmi.inference
"""

import glob
import itertools
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from . import config, prompt


# ---------------------------------------------------------------------------
# API-key cycling — one key per worker, round-robin from a token pool
# ---------------------------------------------------------------------------
_token_lock = threading.Lock()


def _load_api_keys(path):
    with open(path, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError(f"no API keys found in {path}")
    return keys


def _build_token_cycler(keys):
    cycle = itertools.cycle(keys)

    def _next():
        with _token_lock:
            return next(cycle)

    return _next


# ---------------------------------------------------------------------------
# Breakpoint resume — pick up where a previous run wrote out
# ---------------------------------------------------------------------------
def _load_existing_output(out_path, df):
    df_exist = pd.DataFrame()
    start_index = 0
    if os.path.exists(out_path):
        df_exist = pd.read_excel(out_path)
        if len(df_exist) > 0 and "ID" in df_exist.columns:
            last_id = df_exist.iloc[-1]["ID"]
            idx_list = df.index[df["ID"] == last_id].tolist()
            if idx_list:
                start_index = idx_list[0] + 1
    return df_exist, start_index


# ---------------------------------------------------------------------------
# Single-row inference — one forward pass of pi_theta
# ---------------------------------------------------------------------------
def _process_row(row, position, client, backbone, ablation):
    participant_id = row["ID"]
    narrative = str(row["input"]) if pd.notna(row["input"]) else ""

    user_prompt = prompt.assemble_prompt(narrative, ablation=ablation)

    if ablation["n_samples"] == 1:
        resp = client.chat.completions.create(
            model=backbone["model"],
            messages=[{"role": "user", "content": user_prompt}],
            temperature=config.TEMPERATURE,
            timeout=config.TIMEOUT_SECONDS,
        )
        content = resp.choices[0].message.content
    else:
        # A4.2 majority-voting branch — N stochastic samples retained as a
        # JSON-serialisable list; downstream aggregation is performed in R.
        contents = []
        for _ in range(ablation["n_samples"]):
            resp = client.chat.completions.create(
                model=backbone["model"],
                messages=[{"role": "user", "content": user_prompt}],
                temperature=max(config.TEMPERATURE, 0.7),
                timeout=config.TIMEOUT_SECONDS,
            )
            contents.append(resp.choices[0].message.content)
        content = "\n---SAMPLE-DELIM---\n".join(contents)

    return {
        "ID":       participant_id,
        "input":    narrative,
        "Response": content,
        "position": position,
    }


# ---------------------------------------------------------------------------
# Per-thread worker
# ---------------------------------------------------------------------------
def _worker_task(rows, worker_id, base_dir, get_token, backbone, ablation):
    if not rows:
        print(f"[worker {worker_id}] no work; exiting.")
        return 0

    api_key = get_token()
    client = OpenAI(api_key=api_key, base_url=backbone["base_url"])

    pbar = tqdm(total=len(rows),
                desc=f"worker {worker_id}",
                position=worker_id + 1,
                leave=True)

    out = []
    for position, row in rows:
        try:
            out.append(_process_row(row, position, client, backbone, ablation))
        except Exception as exc:
            row_id = row.get("ID") if hasattr(row, "get") else None
            print(f"[worker {worker_id}] position={position}, ID={row_id} "
                  f"failed: {exc!r}")
        finally:
            pbar.update(1)

    pbar.close()

    try:
        client.close()
    except Exception:
        pass

    if out:
        worker_path = os.path.join(base_dir, f"output_thread_{worker_id}.xlsx")
        pd.DataFrame(out).to_excel(worker_path, index=False)
        print(f"[worker {worker_id}] wrote {len(out)} rows to {worker_path}")
    else:
        print(f"[worker {worker_id}] produced no rows.")

    return len(out)


# ---------------------------------------------------------------------------
# Row partitioning + post-run integration
# ---------------------------------------------------------------------------
def _chunk_rows(rows, n_workers):
    buckets = [[] for _ in range(n_workers)]
    for idx, item in enumerate(rows):
        buckets[idx % n_workers].append(item)
    return buckets


def _integrate_outputs(base_dir, df_exist):
    worker_files = glob.glob(os.path.join(base_dir, "output_thread_*.xlsx"))
    if not worker_files:
        return df_exist, 0

    dfs = []
    for fp in worker_files:
        try:
            dfs.append(pd.read_excel(fp))
        except Exception as exc:
            print(f"failed to read {fp}: {exc!r}")
    if not dfs:
        return df_exist, 0

    df_new = pd.concat(dfs, ignore_index=True)
    combined = pd.concat([df_exist, df_new], ignore_index=True)
    if "position" in combined.columns:
        combined = combined.sort_values("position", na_position="first")
        combined = combined.drop(columns=["position"])
    return combined, len(df_new)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(os.getcwd(), config.OUTPUT_PATH)
    backbone   = config.POLICY_BACKBONES[config.ACTIVE_BACKBONE]
    ablation   = config.ABLATION

    print(f"LLM-MMI inference")
    print(f"  preset   : {config.ACTIVE_PRESET}")
    print(f"  backbone : {config.ACTIVE_BACKBONE} ({backbone['model']})")
    print(f"  data     : {config.DATA_PATH}")
    print(f"  output   : {out_path}")

    df = pd.read_excel(config.DATA_PATH)
    api_keys = _load_api_keys(config.API_KEY_PATH)
    get_token = _build_token_cycler(api_keys)

    df_exist, start_index = _load_existing_output(out_path, df)
    if start_index >= len(df):
        print("nothing to do; all rows already processed.")
        return

    rows_to_process = [
        (position, row)
        for position, (_, row) in enumerate(
            df.iloc[start_index:].iterrows(), start=start_index
        )
    ]
    if not rows_to_process:
        print("nothing to do.")
        return

    n_workers = min(config.THREAD_WORKERS, len(api_keys), len(rows_to_process))
    if n_workers <= 0:
        print("worker count is zero; aborting.")
        return

    buckets = _chunk_rows(rows_to_process, n_workers)
    print(f"  workers  : {n_workers}  (rows={len(rows_to_process)})")
    for i, b in enumerate(buckets):
        print(f"    - worker {i}: {len(b)} rows")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_worker_task, bucket, worker_id, base_dir,
                            get_token, backbone, ablation)
            for worker_id, bucket in enumerate(buckets)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"worker failed: {exc!r}")

    combined, new_count = _integrate_outputs(base_dir, df_exist)
    if new_count > 0:
        combined.to_excel(out_path, index=False)
        print(f"done; +{new_count} rows, total {len(combined)}; -> {out_path}")
        # Clean up per-worker shards once the merged file is on disk.
        for fp in glob.glob(os.path.join(base_dir, "output_thread_*.xlsx")):
            try:
                os.remove(fp)
            except OSError:
                pass
    else:
        print("no new rows produced.")


if __name__ == "__main__":
    main()
