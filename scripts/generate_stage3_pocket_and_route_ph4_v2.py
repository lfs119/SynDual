from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from syndual.bb_constraints import BuildingBlockTokenTrie
from syndual.chemistry import canonicalize_smiles, is_valid_smiles
from syndual.model_syn_pocket_ph4_v2 import PocketConditionedSynthesisModel, move_pocket_to_device
from syndual.pocket_data_ph4_v2 import PocketStore, build_stage3_index, collate_pockets
from syndual.reactant_constraints import CompositeRouteTokenConstraint, CurrentGraphReactantConstraint
from syndual.route_serialization import (
    decode_route_ids_to_source_tokens,
    final_product_from_route,
    format_route_tokens,
    molecule_tokens_from_route,
    route_format_valid,
)
from syndual.utils import load_yaml, resolve_device, resolve_path, seed_everything


ALL_CONDITIONS = (
    "correct_ab",
    "wrong_a",
    "wrong_b",
    "wrong_both",
    "zero_condition",
    "unconditional",
)


def canonical_equal(a, b):
    if a is None or b is None:
        return False
    ca = canonicalize_smiles(a, isomeric=True)
    cb = canonicalize_smiles(b, isomeric=True)
    return ca is not None and cb is not None and ca == cb


def pocket_batch(store, pair_id, device):
    entry = store[int(pair_id)]
    item = {
        "aa_ids": entry.aa_ids,
        "coords": entry.coords,
        "core_mask": entry.core_mask,
        "pair_id": entry.pair_id,
    }
    return move_pocket_to_device(collate_pockets([item]), device)


def _parse_conditions(text: Optional[str]) -> list[str]:
    if not text:
        return list(ALL_CONDITIONS)
    values = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in values if x not in ALL_CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}; allowed={ALL_CONDITIONS}")
    return values


def _stage3_checkpoint_from_cfg(cfg: dict, base: Path) -> Path:
    model_cfg = cfg["model"]
    stage3_value = model_cfg.get("stage3_checkpoint")
    if stage3_value is None:
        stage3_output = model_cfg.get("stage3_output_dir")
        if stage3_output is None:
            raise ValueError("Provide model.stage3_checkpoint or model.stage3_output_dir.")
        stage3_value = str(Path(stage3_output) / "best")
    return Path(resolve_path(stage3_value, base))


def _stage2_checkpoint_from_cfg(cfg: dict, base: Path) -> Path:
    model_cfg = cfg["model"]
    value = model_cfg.get("stage2_checkpoint", model_cfg.get("checkpoint"))
    if value is None:
        raise ValueError("Provide model.stage2_checkpoint in Stage3 config.")
    return Path(resolve_path(value, base))


def _default_negative_manifest(stage3_ckpt: Path, split: str) -> Path:
    parent = stage3_ckpt.parent if stage3_ckpt.name == "best" else stage3_ckpt
    return parent / f"eval_{split}_per_record.csv"


def _load_negative_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Negative manifest not found: {path}\n"
            "Run detailed V3A evaluation first or pass --negative-manifest."
        )
    frame = pd.read_csv(path)
    required = {
        "dual_target_record_id",
        "pair_id_a",
        "pair_id_b",
        "neg_pair_id_a",
        "neg_pair_id_b",
        "neg_both_pair_id_a",
        "neg_both_pair_id_b",
        "negative_strategy_a",
        "negative_strategy_b",
        "negative_strategy_both",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Negative manifest missing columns: {missing}")
    frame = frame.drop_duplicates("dual_target_record_id", keep="first").copy()
    frame["_record_key"] = frame["dual_target_record_id"].astype(str)
    return frame


def _merge_stage3_index_with_manifest(index: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    idx = index.copy()
    idx["_record_key"] = idx["dual_target_record_id"].astype(str)
    neg_cols = [
        "_record_key",
        "neg_pair_id_a",
        "neg_pair_id_b",
        "neg_both_pair_id_a",
        "neg_both_pair_id_b",
        "negative_strategy_a",
        "negative_strategy_b",
        "negative_strategy_both",
    ]
    merged = idx.merge(manifest[neg_cols], on="_record_key", how="left", validate="many_to_one")
    required_neg = [
        "neg_pair_id_a",
        "neg_pair_id_b",
        "neg_both_pair_id_a",
        "neg_both_pair_id_b",
    ]
    missing_rows = merged[required_neg].isna().any(axis=1)
    if bool(missing_rows.any()):
        bad = merged.loc[missing_rows, "dual_target_record_id"].astype(str).tolist()[:10]
        raise RuntimeError(
            f"{int(missing_rows.sum())} records have no V3A evaluation negatives; examples={bad}"
        )
    return merged


def _select_records(frame: pd.DataFrame, limit: Optional[int], seed: int, record_id: Optional[str]) -> pd.DataFrame:
    if record_id is not None:
        hit = frame[frame["dual_target_record_id"].astype(str) == str(record_id)]
        if hit.empty:
            raise KeyError(f"record-id={record_id} not found")
        return hit.iloc[:1].copy()

    if limit is None or int(limit) >= len(frame):
        return frame.copy()

    return (
        frame.sample(n=int(limit), random_state=int(seed), replace=False)
        .sort_values("dual_target_record_id", key=lambda s: s.astype(str))
        .reset_index(drop=True)
    )


def _seed_for_record(base_seed: int, record_id: object) -> int:
    digest = hashlib.sha1(str(record_id).encode("utf-8")).hexdigest()[:8]
    return (int(base_seed) + int(digest, 16)) % (2**31 - 1)


def _reset_torch_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _generation_kwargs(cfg: dict, num_samples: int) -> dict:
    gen = cfg.get("generation", {})
    return {
        "num_samples": int(num_samples),
        "max_length": int(gen.get("smiles_max_length", 160)),
        "temperature": float(gen.get("smiles_temperature", 1.0)),
        "top_k": gen.get("smiles_top_k"),
        "top_p": gen.get("smiles_top_p", 0.95),
        "do_sample": bool(gen.get("smiles_do_sample", True)),
        "start_token_id": gen.get("start_token_id"),
        "end_token_id": gen.get("end_token_id"),
    }


def _generate_for_condition(model, store, row, condition: str, device: torch.device, kwargs: dict):
    pair_a = int(row["pair_id_a"])
    pair_b = int(row["pair_id_b"])
    metadata = {
        "true_pair_id_a": pair_a,
        "true_pair_id_b": pair_b,
        "used_pair_id_a": None,
        "used_pair_id_b": None,
        "negative_strategy": None,
    }

    if condition == "correct_ab":
        used_a, used_b = pair_a, pair_b
        metadata.update(used_pair_id_a=used_a, used_pair_id_b=used_b)
        return model.generate_smiles(
            pocket_batch(store, used_a, device), pocket_batch(store, used_b, device), **kwargs
        ), metadata

    if condition == "wrong_a":
        used_a, used_b = int(row["neg_pair_id_a"]), pair_b
        metadata.update(
            used_pair_id_a=used_a,
            used_pair_id_b=used_b,
            negative_strategy=str(row["negative_strategy_a"]),
        )
        return model.generate_smiles(
            pocket_batch(store, used_a, device), pocket_batch(store, used_b, device), **kwargs
        ), metadata

    if condition == "wrong_b":
        used_a, used_b = pair_a, int(row["neg_pair_id_b"])
        metadata.update(
            used_pair_id_a=used_a,
            used_pair_id_b=used_b,
            negative_strategy=str(row["negative_strategy_b"]),
        )
        return model.generate_smiles(
            pocket_batch(store, used_a, device), pocket_batch(store, used_b, device), **kwargs
        ), metadata

    if condition == "wrong_both":
        used_a, used_b = int(row["neg_both_pair_id_a"]), int(row["neg_both_pair_id_b"])
        metadata.update(
            used_pair_id_a=used_a,
            used_pair_id_b=used_b,
            negative_strategy=str(row["negative_strategy_both"]),
        )
        return model.generate_smiles(
            pocket_batch(store, used_a, device), pocket_batch(store, used_b, device), **kwargs
        ), metadata

    if condition == "zero_condition":
        return model.generate_zero_condition_smiles(**kwargs), metadata

    if condition == "unconditional":
        return model.generate_unconditional_smiles(**kwargs), metadata

    raise ValueError(f"Unknown condition: {condition}")


def _route_settings(stage2e_cfg: Optional[dict], cfg: dict) -> tuple[dict, bool]:
    if stage2e_cfg is not None:
        evaluation = dict(stage2e_cfg.get("evaluation", {}))
        require_forward_only = bool(stage2e_cfg.get("data", {}).get("require_forward_only", True))
        return evaluation, require_forward_only

    evaluation = dict(cfg.get("evaluation", cfg.get("generation", {})))
    require_forward_only = bool(
        cfg.get("data", {}).get("require_forward_only", evaluation.get("route_forward_only", True))
    )
    return evaluation, require_forward_only


def _make_bb_constraint(route_tokenizer, evaluation: dict, base: Path):
    if not bool(evaluation.get("bb_constrained_decoding", False)):
        return None
    bb_path = evaluation.get("building_block_vocab_path")
    if not bb_path:
        raise ValueError("bb_constrained_decoding=true requires building_block_vocab_path")
    return BuildingBlockTokenTrie.from_vocab_path(
        tokenizer=route_tokenizer,
        path=resolve_path(bb_path, base),
    )


def _empty_route_result() -> dict:
    return {
        "route_tokens": [],
        "route_text": "",
        "route_decode_error": None,
        "route_format_valid": False,
        "route_all_molecules_valid": False,
        "route_final_product": None,
        "route_final_product_matches_smiles": False,
        "route_success": False,
        "synthesis_pass": False,
        "validity_aware_decoding": None,
    }


def _generate_route_for_smiles(
    *, model, tokenizer, route_tokenizer, stage2_cfg, evaluation, require_forward_only,
    bb_constraint, smiles: str, device: torch.device,
) -> dict:
    if not is_valid_smiles(smiles):
        return _empty_route_result()

    reactant_constraint = None
    if bool(evaluation.get("reactant_graph_constrained_decoding", True)):
        reactant_constraint = CurrentGraphReactantConstraint(
            tokenizer=route_tokenizer,
            terminate_invalid_prefix_with_eos=True,
        )

    constraints = [x for x in (bb_constraint, reactant_constraint) if x is not None]
    route_constraint = CompositeRouteTokenConstraint(constraints) if constraints else None

    enc = tokenizer(
        [smiles],
        add_special_tokens=True,
        truncation=True,
        max_length=int(model.backbone.config.max_position_embeddings),
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    validity = dict(evaluation.get("validity_aware_decoding", {}))

    with torch.inference_mode():
        route_ids, diag = model.stage2.generate_route_validity_aware_v2(
            product_input_ids=ids,
            product_attention_mask=mask,
            route_tokenizer=route_tokenizer,
            max_length=int(stage2_cfg["route"]["max_length"]),
            do_sample=bool(evaluation.get("route_do_sample", False)),
            temperature=float(evaluation.get("route_temperature", 1.0)),
            top_k=evaluation.get("route_top_k"),
            forward_only=bool(require_forward_only),
            token_constraint=route_constraint,
            product_state_tracker=reactant_constraint,
            primary_do_sample=bool(validity.get("primary_do_sample", False)),
            repair_beam_width=int(validity.get("beam_width", validity.get("repair_beam_width", 4))),
            repair_candidate_pool_size=int(
                validity.get("candidate_pool_size", validity.get("repair_candidate_pool_size", 8))
            ),
            repair_max_tokens=int(
                validity.get("max_molecule_tokens", validity.get("repair_max_tokens", 160))
            ),
            repair_length_penalty=float(
                validity.get("length_penalty", validity.get("repair_length_penalty", 0.6))
            ),
            repair_temperature=float(validity.get("repair_temperature", 1.0)),
            repair_top_k=validity.get("repair_top_k"),
        )

    generated_ids = route_ids[0].detach().cpu().tolist()
    decode_error = None
    try:
        route_tokens = decode_route_ids_to_source_tokens(generated_ids, route_tokenizer, strict=True)
    except Exception as exc:
        decode_error = f"{type(exc).__name__}: {exc}"
        route_tokens = decode_route_ids_to_source_tokens(generated_ids, route_tokenizer, strict=False)

    fmt = route_format_valid(route_tokens, require_forward_only=bool(require_forward_only))
    molecules = []
    final_product = None
    if fmt:
        try:
            molecules = molecule_tokens_from_route(route_tokens)
            final_product = final_product_from_route(route_tokens)
        except Exception:
            pass

    all_molecules_valid = bool(molecules) and all(is_valid_smiles(x) for x in molecules)
    final_matches = canonical_equal(final_product, smiles)
    route_success = bool(fmt and all_molecules_valid)
    synthesis_pass = bool(route_success and final_matches)

    return {
        "route_tokens": route_tokens,
        "route_text": format_route_tokens(route_tokens),
        "route_decode_error": decode_error,
        "route_format_valid": bool(fmt),
        "route_all_molecules_valid": bool(all_molecules_valid),
        "route_final_product": final_product,
        "route_final_product_matches_smiles": bool(final_matches),
        "route_success": route_success,
        "synthesis_pass": synthesis_pass,
        "validity_aware_decoding": diag,
    }


def _summarize(rows: list[dict]) -> dict:
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_type"])].append(row)

    summary = {"n_rows": len(rows), "by_condition": {}}
    for condition, items in sorted(by_condition.items()):
        n = len(items)
        valid = [x for x in items if bool(x.get("valid_smiles"))]
        canonical = [x.get("canonical_smiles") for x in valid if x.get("canonical_smiles")]
        route_success = sum(bool(x.get("route_success")) for x in items)
        synthesis_pass = sum(bool(x.get("synthesis_pass")) for x in items)
        summary["by_condition"][condition] = {
            "n_generated": n,
            "n_valid": len(valid),
            "validity_rate": len(valid) / n if n else None,
            "n_unique_valid": len(set(canonical)),
            "uniqueness_among_valid": len(set(canonical)) / len(valid) if valid else None,
            "route_success": route_success,
            "route_success_rate_generated": route_success / n if n else None,
            "synthesis_pass": synthesis_pass,
            "synthesis_pass_rate_generated": synthesis_pass / n if n else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_pocket.yaml")
    parser.add_argument("--stage2e-config", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--num-molecules", type=int, default=16)
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    parser.add_argument("--negative-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-route", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    seed = int(cfg.get("seed", 20260812))
    seed_everything(seed)
    device = resolve_device(cfg.get("device", "auto"))

    stage2e_cfg = None
    stage2e_base = base
    if args.stage2e_config:
        stage2e_path = Path(args.stage2e_config).resolve()
        stage2e_cfg = load_yaml(stage2e_path)
        stage2e_base = stage2e_path.parent

    stage2_ckpt = _stage2_checkpoint_from_cfg(cfg, base)
    stage3_ckpt = _stage3_checkpoint_from_cfg(cfg, base)
    model, tokenizer, route_tokenizer, stage2_cfg, _ = PocketConditionedSynthesisModel.from_pretrained(
        stage2_checkpoint=stage2_ckpt,
        stage3_checkpoint=stage3_ckpt,
        device=device,
    )
    model.eval()

    store = PocketStore(
        resolve_path(cfg["data"]["pocket_context_csv"], base),
        max_residues=cfg["pocket"].get("max_residues", 128),
        use_core_flag=cfg["pocket"].get("use_core_flag", False),
    )

    index, _ = build_stage3_index(
        resolve_path(cfg["data"]["split_root"], base),
        args.split,
        store,
        strict_metadata=bool(cfg["data"].get("strict_pocket_metadata", True)),
    )

    negative_manifest = (
        Path(args.negative_manifest).resolve()
        if args.negative_manifest
        else _default_negative_manifest(stage3_ckpt, args.split)
    )
    manifest = _load_negative_manifest(negative_manifest)
    records = _merge_stage3_index_with_manifest(index, manifest)
    records = _select_records(records, args.limit_records, seed, args.record_id)

    conditions = _parse_conditions(args.conditions)
    n_samples = int(args.num_molecules)
    if n_samples <= 0:
        raise ValueError("--num-molecules must be > 0")

    gen_kwargs = _generation_kwargs(cfg, n_samples)
    if not gen_kwargs["do_sample"] and n_samples > 1:
        raise ValueError(
            "Multiple V3A molecules requested with smiles_do_sample=false. "
            "Deterministic argmax would duplicate the same SMILES. "
            "Set generation.smiles_do_sample=true or use --num-molecules 1."
        )

    route_eval, require_forward_only = _route_settings(stage2e_cfg, cfg)
    route_base = stage2e_base if stage2e_cfg is not None else base
    bb_constraint = None if args.skip_route else _make_bb_constraint(route_tokenizer, route_eval, route_base)

    default_output = Path("../outputs") / "stage3_v3A_generation" / f"{args.split}_{len(records)}records_{n_samples}samples"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (base / default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"generation_{args.split}.jsonl"
    csv_path = output_dir / f"generation_{args.split}.csv"
    summary_path = output_dir / f"generation_{args.split}_summary.json"
    manifest_out = output_dir / f"generation_{args.split}_records.csv"
    records.drop(columns=["_record_key"], errors="ignore").to_csv(manifest_out, index=False)

    rows = []
    route_cache = {}
    total_groups = len(records) * len(conditions)
    group_no = 0

    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        for _, rec in records.iterrows():
            record_id = rec["dual_target_record_id"]
            paired_seed = _seed_for_record(seed, record_id)

            for condition in conditions:
                group_no += 1
                print(f"[{group_no}/{total_groups}] split={args.split} record={record_id} condition={condition}")

                # Common-random-number pairing across conditions for the same record.
                _reset_torch_seed(paired_seed, device)
                with torch.inference_mode():
                    smiles_list, condition_meta = _generate_for_condition(
                        model=model,
                        store=store,
                        row=rec,
                        condition=condition,
                        device=device,
                        kwargs=gen_kwargs,
                    )

                if len(smiles_list) != n_samples:
                    raise RuntimeError(
                        f"Expected {n_samples} SMILES, got {len(smiles_list)} "
                        f"for record={record_id}, condition={condition}"
                    )

                for sample_idx, smi in enumerate(smiles_list):
                    valid = is_valid_smiles(smi)
                    canonical = canonicalize_smiles(smi, isomeric=True) if valid else None
                    item = {
                        "split": args.split,
                        "dual_target_record_id": record_id,
                        "ligand_strict_key": rec.get("ligand_strict_key"),
                        "target_id1": rec.get("target_id1"),
                        "target_id2": rec.get("target_id2"),
                        "pair_id_a": int(rec["pair_id_a"]),
                        "pair_id_b": int(rec["pair_id_b"]),
                        "condition_type": condition,
                        "negative_strategy_a": rec.get("negative_strategy_a"),
                        "negative_strategy_b": rec.get("negative_strategy_b"),
                        "negative_strategy_both": rec.get("negative_strategy_both"),
                        **condition_meta,
                        "generation_seed": int(paired_seed),
                        "sample_idx": int(sample_idx),
                        "smiles": smi,
                        "canonical_smiles": canonical,
                        "valid_smiles": bool(valid),
                    }

                    if args.skip_route:
                        item.update(route_success=False, synthesis_pass=False, route_skipped=True)
                    elif valid:
                        route_key = canonical or smi
                        if route_key not in route_cache:
                            route_cache[route_key] = _generate_route_for_smiles(
                                model=model,
                                tokenizer=tokenizer,
                                route_tokenizer=route_tokenizer,
                                stage2_cfg=stage2_cfg,
                                evaluation=route_eval,
                                require_forward_only=require_forward_only,
                                bb_constraint=bb_constraint,
                                smiles=smi,
                                device=device,
                            )
                        item.update(route_cache[route_key])
                        item["route_skipped"] = False
                    else:
                        item.update(_empty_route_result())
                        item["route_skipped"] = False

                    rows.append(item)
                    jsonl_handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
                    jsonl_handle.flush()

    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    try:
        frame.to_parquet(output_dir / f"generation_{args.split}.parquet", index=False)
    except Exception as exc:
        print(f"[warning] parquet not written: {type(exc).__name__}: {exc}")

    summary = _summarize(rows)
    summary.update(
        {
            "split": args.split,
            "n_records": int(len(records)),
            "n_conditions": int(len(conditions)),
            "conditions": conditions,
            "num_molecules_per_condition": n_samples,
            "expected_rows": int(len(records) * len(conditions) * n_samples),
            "seed": seed,
            "paired_condition_sampling": True,
            "negative_manifest": str(negative_manifest),
            "stage2_checkpoint": str(stage2_ckpt),
            "stage3_checkpoint": str(stage3_ckpt),
            "stage2e_config": str(Path(args.stage2e_config).resolve()) if args.stage2e_config else None,
            "skip_route": bool(args.skip_route),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone.")
    print("Rows:", len(rows))
    print("JSONL:", jsonl_path)
    print("CSV:", csv_path)
    print("Summary:", summary_path)
    print("Record manifest:", manifest_out)


if __name__ == "__main__":
    main()
