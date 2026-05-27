"""LLM-MMI-LT population-scale inference runner (text + VLM branches).

Implements Stages 2 and 5 of the methodology for the LT task:

* the *text branch* runs :mod:`llm_mmi.lt.prompt` against a frozen policy
  ``pi_theta`` (one-pass greedy, ``T=0``) and emits a JSON conforming to
  :mod:`llm_mmi.lt.schema`;
* the *VLM branch* (when enabled by the active ablation) runs
  :mod:`llm_mmi.lt.vlm` over the CAM-selected keyframes from
  ``LT/build_tree.py`` + ``LT/CAM.py`` and emits a binary CT-derived risk.

Both outputs are written to disk per-worker and merged in
:mod:`llm_mmi.lt.fusion`.

Run as::

    python -m llm_mmi.lt.inference
    LLM_MMI_PRESET=A6.1 python -m llm_mmi.lt.inference        # text-only
    LT_BACKBONE=internvl3-14b python -m llm_mmi.lt.inference
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from . import config, prompt, vlm


# ---------------------------------------------------------------------------
# API-key cycling (one key per worker, round-robin)
# ---------------------------------------------------------------------------
_token_lock = threading.Lock()


def _load_api_keys(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError(f"no API keys found in {path}")
    return keys


def _build_token_cycler(keys: list[str]):
    cycle = itertools.cycle(keys)

    def _next():
        with _token_lock:
            return next(cycle)

    return _next


# ---------------------------------------------------------------------------
# Breakpoint resume
# ---------------------------------------------------------------------------
def _load_existing_output(out_path: str, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
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
# Keyframe lookup
# ---------------------------------------------------------------------------
def _load_keyframe_index(path: Optional[str]) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single-row inference: text + VLM
# ---------------------------------------------------------------------------
def _process_row(row,
                 position: int,
                 text_client: OpenAI,
                 vlm_client: Optional[OpenAI],
                 text_backbone: dict,
                 vlm_backbone: Optional[dict],
                 ablation: dict,
                 keyframe_index: dict) -> dict:
    pid = row["ID"]
    narrative = str(row["input"]) if pd.notna(row["input"]) else ""

    # ----- Text branch ----------------------------------------------------
    text_prompt = prompt.assemble_prompt(narrative, ablation=ablation)
    if ablation["n_samples"] == 1:
        resp = text_client.chat.completions.create(
            model=text_backbone["model"],
            messages=[{"role": "user", "content": text_prompt}],
            temperature=config.TEMPERATURE,
            timeout=config.TIMEOUT_SECONDS,
        )
        text_content = resp.choices[0].message.content
    else:
        contents = []
        for _ in range(ablation["n_samples"]):
            resp = text_client.chat.completions.create(
                model=text_backbone["model"],
                messages=[{"role": "user", "content": text_prompt}],
                temperature=max(config.TEMPERATURE, 0.7),
                timeout=config.TIMEOUT_SECONDS,
            )
            contents.append(resp.choices[0].message.content)
        text_content = "\n---SAMPLE-DELIM---\n".join(contents)

    # ----- VLM branch -----------------------------------------------------
    vlm_content: Optional[str] = None
    ct_risk_binary: Optional[int] = None
    ct_risk_label: Optional[str] = None
    if ablation.get("use_vlm_branch") and vlm_client is not None:
        patient_record = keyframe_index.get(str(pid)) or keyframe_index.get(pid)
        if patient_record:
            keyframes = vlm.select_keyframes_from_index(
                patient_record, topk=config.VLM_TOPK_KEYFRAMES,
            )
            if keyframes:
                messages = vlm.build_vlm_messages(
                    keyframes,
                    narrative_excerpt=narrative,
                )
                try:
                    vresp = vlm_client.chat.completions.create(
                        model=vlm_backbone["model"],
                        messages=messages,
                        temperature=config.TEMPERATURE,
                        timeout=config.TIMEOUT_SECONDS,
                    )
                    vlm_content = vresp.choices[0].message.content
                    parsed = vlm.parse_vlm_response(vlm_content)
                    ct_risk_binary = parsed["ct_risk_binary"]
                    ct_risk_label  = parsed["ct_risk_label"]
                except Exception as exc:                       # noqa: BLE001
                    vlm_content = f"<VLM_ERROR>{exc!r}</VLM_ERROR>"

    return {
        "ID":             pid,
        "input":          narrative,
        "Response":       text_content,
        "VLM_Response":   vlm_content,
        "ct-derived risk":  ct_risk_binary,
        "ct-derived label": ct_risk_label,
        "position":       position,
    }


# ---------------------------------------------------------------------------
# Per-thread worker
# ---------------------------------------------------------------------------
def _worker_task(rows,
                 worker_id: int,
                 base_dir: str,
                 get_token,
                 text_backbone: dict,
                 vlm_backbone: Optional[dict],
                 ablation: dict,
                 keyframe_index: dict) -> int:
    if not rows:
        print(f"[worker {worker_id}] no work; exiting.")
        return 0

    api_key = get_token()
    text_client = OpenAI(api_key=api_key, base_url=text_backbone["base_url"])
    vlm_client: Optional[OpenAI] = None
    if vlm_backbone is not None:
        # Same key is reused when both branches share an aggregator; for
        # InternVL3-14B on a local vLLM server, base_url differs but key
        # is typically a dummy and the gateway ignores it.
        vlm_client = OpenAI(api_key=api_key, base_url=vlm_backbone["base_url"])

    pbar = tqdm(total=len(rows),
                desc=f"worker {worker_id}",
                position=worker_id + 1,
                leave=True)
    out = []
    for position, row in rows:
        try:
            out.append(_process_row(row, position,
                                    text_client, vlm_client,
                                    text_backbone, vlm_backbone,
                                    ablation, keyframe_index))
        except Exception as exc:                              # noqa: BLE001
            row_id = row.get("ID") if hasattr(row, "get") else None
            print(f"[worker {worker_id}] position={position}, ID={row_id} "
                  f"failed: {exc!r}")
        finally:
            pbar.update(1)
    pbar.close()

    try:
        text_client.close()
    except Exception:
        pass
    if vlm_client is not None:
        try:
            vlm_client.close()
        except Exception:
            pass

    if out:
        worker_path = os.path.join(base_dir, f"lt_thread_{worker_id}.xlsx")
        pd.DataFrame(out).to_excel(worker_path, index=False)
        print(f"[worker {worker_id}] wrote {len(out)} rows to {worker_path}")
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
    worker_files = glob.glob(os.path.join(base_dir, "lt_thread_*.xlsx"))
    if not worker_files:
        return df_exist, 0
    dfs = []
    for fp in worker_files:
        try:
            dfs.append(pd.read_excel(fp))
        except Exception as exc:                              # noqa: BLE001
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(os.getcwd(), config.OUTPUT_PATH)
    text_backbone = config.POLICY_BACKBONES[config.ACTIVE_BACKBONE]
    ablation = config.ABLATION

    vlm_backbone: Optional[dict] = None
    if ablation.get("use_vlm_branch"):
        try:
            vlm_backbone = vlm.get_vlm_backbone()
        except RuntimeError as exc:
            print(f"[warn] disabling VLM branch: {exc}")
            ablation = dict(ablation, use_vlm_branch=False)

    print(f"LLM-MMI-LT inference")
    print(f"  preset       : {config.ACTIVE_PRESET}")
    print(f"  text backbone: {config.ACTIVE_BACKBONE} ({text_backbone['model']})")
    if vlm_backbone is not None:
        print(f"  vlm backbone : {config.VLM_BACKBONE} ({vlm_backbone['model']})")
    else:
        print(f"  vlm branch   : disabled")
    print(f"  data         : {config.DATA_PATH}")
    print(f"  output       : {out_path}")
    print(f"  keyframes    : {config.KEYFRAME_INDEX}")

    df = pd.read_excel(config.DATA_PATH)
    api_keys = _load_api_keys(config.API_KEY_PATH)
    get_token = _build_token_cycler(api_keys)
    keyframe_index = _load_keyframe_index(config.KEYFRAME_INDEX)

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
    print(f"  workers      : {n_workers}  (rows={len(rows_to_process)})")
    for i, b in enumerate(buckets):
        print(f"    - worker {i}: {len(b)} rows")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_worker_task, bucket, worker_id, base_dir,
                            get_token, text_backbone, vlm_backbone,
                            ablation, keyframe_index)
            for worker_id, bucket in enumerate(buckets)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:                          # noqa: BLE001
                print(f"worker failed: {exc!r}")

    combined, new_count = _integrate_outputs(base_dir, df_exist)
    if new_count > 0:
        combined.to_excel(out_path, index=False)
        print(f"done; +{new_count} rows, total {len(combined)}; -> {out_path}")
        for fp in glob.glob(os.path.join(base_dir, "lt_thread_*.xlsx")):
            try:
                os.remove(fp)
            except OSError:
                pass
    else:
        print("no new rows produced.")


if __name__ == "__main__":
    main()
