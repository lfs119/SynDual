#!/usr/bin/env python3
"""
Stage 2G: AiZynthFinder retrosynthetic solvability evaluation.

Purpose
-------
Evaluate whether the final generated products from SynDual routes can be solved
by an independent AiZynthFinder retrosynthesis planner.

This is a planner-solvability / retrosynthetic-accessibility assessment only.
A solved route does NOT prove experimental feasibility.

Recommended input
-----------------
The Stage 2F-0 step CSV:
    runs/stage2f0_full/reference_reaction_support_steps.csv

For each route, the product of the highest step_idx is treated as the final
generated product. Unique canonical targets are deduplicated before AiZynthFinder
search and then mapped back to route records.

Outputs
-------
- aizynth_unique_products.csv
- aizynth_route_records.csv
- aizynth_top_routes.jsonl
- aizynth_summary.json
- aizynth_cache.jsonl (resumable cache)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

try:
    from aizynthfinder.aizynthfinder import AiZynthFinder
except Exception as exc:
    raise RuntimeError(
        "AiZynthFinder could not be imported. Install/configure aizynthfinder "
        "before running Stage 2G."
    ) from exc


# -----------------------------------------------------------------------------
# Chemistry helpers
# -----------------------------------------------------------------------------
_BH2_QUERY = Chem.MolFromSmarts("[B;H2]")
_BORONIC_REPL = Chem.MolFromSmiles("B(O)O")


def canonicalize_smiles(smiles: Any, *, isomeric: bool = True) -> Optional[str]:
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return None
    s = str(smiles).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def convert_bh2_to_boronic(smiles: str) -> tuple[str, bool]:
    """
    Mirror the preprocessing used in SynLaD's evaluate_samples.py.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, False
    if _BH2_QUERY is None or _BORONIC_REPL is None:
        return smiles, False
    if not mol.HasSubstructMatch(_BH2_QUERY):
        return smiles, False
    try:
        new_mol = Chem.ReplaceSubstructs(
            mol, _BH2_QUERY, _BORONIC_REPL, replaceAll=True
        )[0]
        Chem.SanitizeMol(new_mol)
        out = Chem.MolToSmiles(
            new_mol, canonical=True, isomericSmiles=True
        )
        return out, out != smiles
    except Exception:
        return smiles, False


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return float(x)
    try:
        return float(str(x))
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    y = safe_float(x)
    return None if y is None else int(y)


def bool_from_any(x: Any) -> Optional[bool]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    xs = []
    for v in values:
        f = safe_float(v)
        if f is not None and np.isfinite(f):
            xs.append(float(f))
    if not xs:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(xs),
        "mean": float(np.mean(xs)),
        "median": float(np.median(xs)),
        "p95": float(np.percentile(xs, 95)),
        "max": float(np.max(xs)),
    }


def fraction_metric(values: Iterable[Any]) -> dict[str, Any]:
    vals = [bool_from_any(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"count": 0, "numerator": 0, "fraction": None}
    n = int(sum(vals))
    return {"count": len(vals), "numerator": n, "fraction": n / len(vals)}


def detect_column(df: pd.DataFrame, requested: str, candidates: list[str]) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise KeyError(f"Column '{requested}' not found. Available: {list(df.columns)}")
        return requested
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not auto-detect column. Tried {candidates}. "
        f"Available: {list(df.columns)}"
    )


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------
def prepare_targets_from_f0_steps(
    path: Path,
    route_id_column: str,
    product_column: str,
) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)

    route_col = detect_column(
        df,
        route_id_column,
        ["route_id", "test_index", "sample_index", "index", "id"],
    )

    prod_col = detect_column(
        df,
        product_column,
        [
            "product_canonical_isomeric",
            "generated_product_raw",
            "generated_product",
            "product_smiles",
        ],
    )

    if "step_idx" not in df.columns:
        raise KeyError("F0 step CSV must contain 'step_idx'.")

    tmp = df.copy()
    tmp["_step_idx_num"] = pd.to_numeric(tmp["step_idx"], errors="coerce")
    tmp = tmp.sort_values([route_col, "_step_idx_num"], kind="stable")
    final_df = tmp.groupby(route_col, as_index=False, sort=False).tail(1).copy()

    final_df = final_df.rename(
        columns={
            route_col: "route_id",
            prod_col: "final_product_input",
        }
    )

    keep = ["route_id", "final_product_input"]
    for c in [
        "reference_exact_reaction_support",
        "train_seen_exact_reaction",
        "train_novel_exact_reaction",
        "reaction_evaluable",
    ]:
        if c in final_df.columns:
            keep.append(c)
    return final_df[keep].reset_index(drop=True), route_col, prod_col


def prepare_targets_from_smiles_csv(
    path: Path,
    route_id_column: str,
    smiles_column: str,
) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    prod_col = detect_column(
        df,
        smiles_column,
        [
            "generated_final_product_smiles",
            "final_product_smiles",
            "generated_product",
            "product_smiles",
            "canonical_product_smiles",
        ],
    )
    if route_id_column == "auto":
        route_col = None
        for c in ["route_id", "test_index", "sample_index", "index", "id"]:
            if c in df.columns:
                route_col = c
                break
        if route_col is None:
            df = df.copy()
            df["route_id"] = np.arange(len(df), dtype=int)
            route_col = "route_id"
    else:
        route_col = detect_column(df, route_id_column, [])

    out = df.rename(
        columns={route_col: "route_id", prod_col: "final_product_input"}
    ).copy()
    return out, route_col, prod_col


def add_canonical_targets(df: pd.DataFrame, convert_bh2: bool) -> pd.DataFrame:
    out = df.copy()
    canonical = []
    query = []
    converted = []
    valid = []

    for smi in out["final_product_input"].tolist():
        c = canonicalize_smiles(smi, isomeric=True)
        canonical.append(c)
        if c is None:
            query.append(None)
            converted.append(False)
            valid.append(False)
            continue
        if convert_bh2:
            q, flag = convert_bh2_to_boronic(c)
        else:
            q, flag = c, False
        q = canonicalize_smiles(q, isomeric=True)
        query.append(q)
        converted.append(bool(flag))
        valid.append(q is not None)

    out["final_product_canonical"] = canonical
    out["aizynth_query_smiles"] = query
    out["bh2_to_boronic_converted"] = converted
    out["aizynth_target_valid"] = valid
    return out


# -----------------------------------------------------------------------------
# Optional route-level annotations from F1/F2
# -----------------------------------------------------------------------------
def merge_annotation(
    base: pd.DataFrame,
    path: Optional[Path],
    *,
    prefix: str,
    recognized: list[str],
    route_id_column: str = "auto",
) -> pd.DataFrame:
    if path is None:
        return base
    ann = pd.read_csv(path)
    rid = detect_column(
        ann,
        route_id_column,
        ["route_id", "test_index", "sample_index", "index", "id"],
    )
    cols = [rid] + [c for c in recognized if c in ann.columns]
    ann = ann[cols].copy()
    rename = {rid: "route_id"}
    rename.update({c: f"{prefix}{c}" for c in cols if c != rid})
    ann = ann.rename(columns=rename)
    return base.merge(ann, on="route_id", how="left", validate="m:1")


# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------
def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = rec.get("cache_key")
            if key:
                cache[key] = rec
    return cache


def append_cache(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")
        f.flush()


def make_cache_key(
    smiles: str,
    config_sha256: str,
    stock: str,
    expansion_policy: str,
    filter_policy: Optional[str],
) -> str:
    payload = "|".join(
        [smiles, config_sha256, stock, expansion_policy, filter_policy or "none"]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# AiZynthFinder
# -----------------------------------------------------------------------------
def init_finder(
    config: Path,
    stock: str,
    expansion_policy: str,
    filter_policy: Optional[str],
) -> AiZynthFinder:
    finder = AiZynthFinder(configfile=str(config))
    finder.stock.select(stock)
    finder.expansion_policy.select(expansion_policy)
    if filter_policy and filter_policy.lower() != "none":
        finder.filter_policy.select(filter_policy)
    return finder


def solved_from_stats(stats: dict[str, Any]) -> bool:
    n_solved = safe_int(stats.get("number_of_solved_routes"))
    if n_solved is not None:
        return n_solved > 0
    val = bool_from_any(stats.get("is_solved"))
    return bool(val) if val is not None else False


def run_one_target(
    finder: AiZynthFinder,
    smiles: str,
    *,
    keep_top_route: bool,
    show_search_progress: bool,
) -> dict[str, Any]:
    t0 = time.time()
    try:
        finder.target_smiles = smiles
        finder.tree_search(show_progress=show_search_progress)
        finder.build_routes()
        stats = dict(finder.extract_statistics())
        solved = solved_from_stats(stats)

        top_route = None
        if keep_top_route and len(finder.routes) > 0:
            try:
                top_route = finder.routes.dict_with_scores()[0]
            except Exception:
                try:
                    top_route = finder.routes.dicts[0]
                except Exception:
                    top_route = None

        stock_info = None
        try:
            stock_info = finder.stock_info()
        except Exception:
            pass

        return {
            "success": True,
            "solved": bool(solved),
            "error_type": None,
            "error_message": None,
            "elapsed_wall_s": float(time.time() - t0),
            "stats": stats,
            "top_route": top_route,
            "stock_info": stock_info,
        }
    except Exception as exc:
        return {
            "success": False,
            "solved": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "elapsed_wall_s": float(time.time() - t0),
            "stats": {},
            "top_route": None,
            "stock_info": None,
        }


def flatten_result(
    query_smiles: str,
    cache_key: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    stats = result.get("stats") or {}
    row = {
        "aizynth_query_smiles": query_smiles,
        "cache_key": cache_key,
        "aizynth_call_success": bool(result.get("success", False)),
        "aizynth_solved": bool(result.get("solved", False)),
        "error_type": result.get("error_type"),
        "error_message": result.get("error_message"),
        "elapsed_wall_s": result.get("elapsed_wall_s"),
        "search_time_s": stats.get("search_time"),
        "first_solution_time_s": stats.get("first_solution_time"),
        "first_solution_iteration": stats.get("first_solution_iteration"),
        "number_of_nodes": stats.get("number_of_nodes"),
        "max_transforms": stats.get("max_transforms"),
        "max_children": stats.get("max_children"),
        "number_of_routes": stats.get("number_of_routes"),
        "number_of_solved_routes": stats.get("number_of_solved_routes"),
        "top_score": stats.get("top_score"),
        "top_route_is_solved": stats.get("is_solved"),
        "top_route_number_of_steps": stats.get("number_of_steps"),
        "top_route_number_of_precursors": stats.get("number_of_precursors"),
        "top_route_number_of_precursors_in_stock": stats.get(
            "number_of_precursors_in_stock"
        ),
        "precursors_in_stock": stats.get("precursors_in_stock"),
        "precursors_not_in_stock": stats.get("precursors_not_in_stock"),
        "precursors_availability": stats.get("precursors_availability"),
    }
    return row


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def stratified_solved(df: pd.DataFrame, field: str) -> dict[str, Any]:
    if field not in df.columns:
        return {}
    out: dict[str, Any] = {}
    vals = df[field].map(bool_from_any)
    for flag, name in [(True, "true"), (False, "false")]:
        sub = df[vals == flag]
        out[name] = {
            "records": int(len(sub)),
            "solved": fraction_metric(sub["aizynth_solved"]) if len(sub) else {
                "count": 0, "numerator": 0, "fraction": None
            },
        }
    return out


def build_summary(
    route_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    *,
    input_path: Path,
    config_path: Path,
    input_sha256: str,
    config_sha256: str,
    input_mode: str,
    route_id_source_column: str,
    product_source_column: str,
    stock: str,
    expansion_policy: str,
    filter_policy: Optional[str],
    convert_bh2: bool,
    finder: AiZynthFinder,
) -> dict[str, Any]:
    valid_route = route_df[route_df["aizynth_target_valid"] == True].copy()
    attempted_unique = unique_df[unique_df["aizynth_attempted"] == True].copy()

    solved_unique = attempted_unique[
        attempted_unique["aizynth_call_success"] == True
    ].copy()

    search_cfg = {}
    try:
        search_cfg = {
            "algorithm": finder.config.search.algorithm,
            "time_limit": finder.config.search.time_limit,
            "iteration_limit": finder.config.search.iteration_limit,
            "return_first": finder.config.search.return_first,
            "exclude_target_from_stock": finder.config.search.exclude_target_from_stock,
        }
    except Exception:
        pass

    post_cfg = {}
    try:
        post_cfg = {
            "min_routes": finder.config.post_processing.min_routes,
            "max_routes": finder.config.post_processing.max_routes,
            "all_routes": finder.config.post_processing.all_routes,
            "route_scorers": finder.config.post_processing.route_scorers,
        }
    except Exception:
        pass

    summary = {
        "evaluation": {
            "stage": "stage2g_aizynthfinder_retrosynthetic_solvability",
            "policy": "independent_aizynthfinder_tree_search_on_final_generated_products",
            "interpretation": (
                "Planner solvability / retrosynthetic accessibility only. "
                "A solved AiZynthFinder route does not prove experimental feasibility."
            ),
        },
        "counts": {
            "route_records": int(len(route_df)),
            "valid_final_product_records": int(route_df["aizynth_target_valid"].sum()),
            "invalid_final_product_records": int((~route_df["aizynth_target_valid"]).sum()),
            "unique_valid_query_products": int(
                route_df.loc[
                    route_df["aizynth_target_valid"] == True,
                    "aizynth_query_smiles",
                ].nunique()
            ),
            "unique_products_attempted": int(
                unique_df["aizynth_attempted"].sum()
            ),
            "unique_products_not_attempted_due_limit": int(
                (~unique_df["aizynth_attempted"]).sum()
            ),
        },
        "solvability": {
            "unique_product_level": fraction_metric(
                attempted_unique.loc[
                    attempted_unique["aizynth_call_success"] == True,
                    "aizynth_solved",
                ]
            ),
            "route_record_level": fraction_metric(
                valid_route.loc[
                    valid_route["aizynth_attempted"] == True,
                    "aizynth_solved",
                ]
            ),
            "call_success_unique_product_level": fraction_metric(
                attempted_unique["aizynth_call_success"]
            ),
        },
        "search_statistics_unique_products": {
            "search_time_s": numeric_summary(solved_unique["search_time_s"]),
            "elapsed_wall_s": numeric_summary(solved_unique["elapsed_wall_s"]),
            "first_solution_time_s_solved_only": numeric_summary(
                solved_unique.loc[
                    solved_unique["aizynth_solved"] == True,
                    "first_solution_time_s",
                ]
            ),
            "top_route_number_of_steps_solved_only": numeric_summary(
                solved_unique.loc[
                    solved_unique["aizynth_solved"] == True,
                    "top_route_number_of_steps",
                ]
            ),
            "number_of_routes": numeric_summary(solved_unique["number_of_routes"]),
            "number_of_solved_routes": numeric_summary(
                solved_unique["number_of_solved_routes"]
            ),
        },
        "stratified_route_record_solvability": {
            "f1_all_steps_combined_exact_or_template_supported": stratified_solved(
                valid_route, "f1_all_steps_combined_exact_or_template_supported"
            ),
            "f2_all_steps_conservative_sanity_pass": stratified_solved(
                valid_route, "f2_qc_all_steps_conservative_sanity_pass"
            ),
        },
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "input_mode": input_mode,
            "route_id_source_column": route_id_source_column,
            "product_source_column": product_source_column,
        },
        "aizynthfinder": {
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "stock": stock,
            "expansion_policy": expansion_policy,
            "filter_policy": filter_policy,
            "bh2_to_boronic_conversion": convert_bh2,
            "search_config": search_cfg,
            "post_processing_config": post_cfg,
        },
    }
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2G: AiZynthFinder retrosynthetic solvability."
    )
    p.add_argument("--input", required=True, help="F0 step CSV or a SMILES CSV.")
    p.add_argument(
        "--input-mode",
        choices=["f0_steps", "smiles_csv"],
        default="f0_steps",
    )
    p.add_argument("--route-id-column", default="auto")
    p.add_argument("--product-column", default="auto")
    p.add_argument("--config", required=True, help="AiZynthFinder config.yml")
    p.add_argument("--stock", default="zinc")
    p.add_argument("--expansion-policy", default="uspto")
    p.add_argument("--filter-policy", default="uspto")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--cache-jsonl",
        default=None,
        help="Resumable per-unique-target cache. Default: OUTPUT_DIR/aizynth_cache.jsonl",
    )
    p.add_argument(
        "--limit-unique",
        type=int,
        default=None,
        help="Optional number of unique products to evaluate.",
    )
    p.add_argument(
        "--sample-unique",
        action="store_true",
        help="When --limit-unique is used, randomly sample unique products.",
    )
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument(
        "--no-bh2-conversion",
        action="store_true",
        help="Disable SynLaD-style [BH2] -> B(O)O preprocessing.",
    )
    p.add_argument(
        "--no-top-route-json",
        action="store_true",
        help="Do not save the top-ranked AiZynthFinder route.",
    )
    p.add_argument("--show-search-progress", action="store_true")
    p.add_argument("--quiet-rdkit", action="store_true")
    p.add_argument(
        "--f1-route-csv",
        default=None,
        help="Optional Stage 2F-1 route CSV for stratified analysis.",
    )
    p.add_argument(
        "--f2-route-csv",
        default=None,
        help="Optional Stage 2F-2 route CSV for stratified analysis.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    config_path = Path(args.config).resolve()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quiet_rdkit:
        RDLogger.DisableLog("rdApp.*")

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if args.input_mode == "f0_steps":
        route_df, route_id_src, product_src = prepare_targets_from_f0_steps(
            input_path, args.route_id_column, args.product_column
        )
    else:
        route_df, route_id_src, product_src = prepare_targets_from_smiles_csv(
            input_path, args.route_id_column, args.product_column
        )

    route_df = add_canonical_targets(
        route_df, convert_bh2=not args.no_bh2_conversion
    )

    # Optional route annotations.
    route_df = merge_annotation(
        route_df,
        Path(args.f1_route_csv).resolve() if args.f1_route_csv else None,
        prefix="f1_",
        recognized=[
            "all_steps_exact_reference_supported",
            "any_step_template_executable",
            "all_steps_combined_exact_or_template_supported",
            "any_step_unsupported_after_f1",
        ],
    )
    route_df = merge_annotation(
        route_df,
        Path(args.f2_route_csv).resolve() if args.f2_route_csv else None,
        prefix="f2_",
        recognized=[
            "qc_all_steps_evaluable",
            "qc_all_steps_conservative_sanity_pass",
            "qc_any_high_alert_step",
        ],
    )

    valid_queries = sorted(
        set(
            route_df.loc[
                route_df["aizynth_target_valid"] == True, "aizynth_query_smiles"
            ].dropna()
        )
    )

    selected_queries = list(valid_queries)
    if args.limit_unique is not None and args.limit_unique < len(selected_queries):
        if args.sample_unique:
            rng = random.Random(args.seed)
            selected_queries = rng.sample(selected_queries, args.limit_unique)
            selected_queries = sorted(selected_queries)
        else:
            selected_queries = selected_queries[: args.limit_unique]

    selected_set = set(selected_queries)

    config_sha = sha256_file(config_path)
    input_sha = sha256_file(input_path)

    finder = init_finder(
        config_path,
        args.stock,
        args.expansion_policy,
        None if args.filter_policy.lower() == "none" else args.filter_policy,
    )

    cache_path = (
        Path(args.cache_jsonl).resolve()
        if args.cache_jsonl
        else outdir / "aizynth_cache.jsonl"
    )
    cache = load_cache(cache_path)

    unique_rows: list[dict[str, Any]] = []
    top_route_records: list[dict[str, Any]] = []

    for q in tqdm(valid_queries, desc="AiZynthFinder targets"):
        attempted = q in selected_set
        key = make_cache_key(
            q,
            config_sha,
            args.stock,
            args.expansion_policy,
            None if args.filter_policy.lower() == "none" else args.filter_policy,
        )

        if not attempted:
            unique_rows.append(
                {
                    "aizynth_query_smiles": q,
                    "cache_key": key,
                    "aizynth_attempted": False,
                    "aizynth_call_success": None,
                    "aizynth_solved": None,
                }
            )
            continue

        if key in cache:
            rec = cache[key]
            result = rec["result"]
        else:
            result = run_one_target(
                finder,
                q,
                keep_top_route=not args.no_top_route_json,
                show_search_progress=args.show_search_progress,
            )
            rec = {
                "cache_key": key,
                "query_smiles": q,
                "config_sha256": config_sha,
                "stock": args.stock,
                "expansion_policy": args.expansion_policy,
                "filter_policy": None
                if args.filter_policy.lower() == "none"
                else args.filter_policy,
                "result": result,
            }
            append_cache(cache_path, rec)
            cache[key] = rec

        flat = flatten_result(q, key, result)
        flat["aizynth_attempted"] = True
        unique_rows.append(flat)

        if result.get("top_route") is not None:
            top_route_records.append(
                {
                    "aizynth_query_smiles": q,
                    "cache_key": key,
                    "aizynth_solved": bool(result.get("solved", False)),
                    "top_route": result.get("top_route"),
                    "stock_info": result.get("stock_info"),
                }
            )

    unique_df = pd.DataFrame(unique_rows)

    route_df = route_df.merge(
        unique_df,
        on="aizynth_query_smiles",
        how="left",
        validate="m:1",
    )

    # Invalid targets were never attempted.
    if "aizynth_attempted" not in route_df.columns:
        route_df["aizynth_attempted"] = False
    route_df["aizynth_attempted"] = route_df["aizynth_attempted"].fillna(False)

    # Save data.
    unique_path = outdir / "aizynth_unique_products.csv"
    route_path = outdir / "aizynth_route_records.csv"
    top_routes_path = outdir / "aizynth_top_routes.jsonl"
    summary_path = outdir / "aizynth_summary.json"

    unique_df.to_csv(unique_path, index=False)
    route_df.to_csv(route_path, index=False)

    if not args.no_top_route_json:
        with top_routes_path.open("w", encoding="utf-8") as f:
            for rec in top_route_records:
                f.write(
                    json.dumps(rec, ensure_ascii=False, default=json_default)
                    + "\n"
                )

    summary = build_summary(
        route_df,
        unique_df,
        input_path=input_path,
        config_path=config_path,
        input_sha256=input_sha,
        config_sha256=config_sha,
        input_mode=args.input_mode,
        route_id_source_column=route_id_src,
        product_source_column=product_src,
        stock=args.stock,
        expansion_policy=args.expansion_policy,
        filter_policy=None
        if args.filter_policy.lower() == "none"
        else args.filter_policy,
        convert_bh2=not args.no_bh2_conversion,
        finder=finder,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=json_default)

    print("\nStage 2G complete")
    print(f"Unique products: {unique_path}")
    print(f"Route records:   {route_path}")
    if not args.no_top_route_json:
        print(f"Top routes:      {top_routes_path}")
    print(f"Summary:         {summary_path}")
    print(f"Cache:           {cache_path}")
    print(json.dumps(summary["solvability"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
