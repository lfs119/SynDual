from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from syndual.model_syn import SynthesisAwareModel
from syndual.model_syn_pocket_ph4_v2 import PocketConditionedSynthesisModel
from syndual.molformer_prefix import get_backbone_hidden_size, get_backbone_max_positions
from syndual.pocket_adapter_ph4_v2 import FactorizedDualPocketAdapter
from syndual.pocket_data_ph4_v2 import PocketStore, build_stage3_index
from syndual.ph4_local_aux import (
    PH4LocalTargetStore,
    Stage3PocketCollatorPH4V2,
    Stage3PocketDatasetPH4V2,
    balanced_local_ph4_bce,
    evaluate_local_ph4_grounding,
)
from syndual.stage3_evaluation import evaluate_model, move_batch_to_device, save_detailed_evaluation
from syndual.stage3_selection import selection_value
from syndual.utils import load_yaml, resolve_device, resolve_path, save_json, seed_everything


def ranking_hinge(pos_nll: torch.Tensor, neg_nll: torch.Tensor, margin: float) -> torch.Tensor:
    return torch.relu(float(margin) + pos_nll - neg_nll).mean()


def mean_tensor(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("No tensor values were supplied")
    return torch.stack(values).mean()


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash parameters + registered buffers."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        x = tensor.detach().cpu().contiguous()
        h.update(str(x.dtype).encode("utf-8"))
        h.update(str(tuple(x.shape)).encode("utf-8"))
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def _core_adapter_kwargs(saved_cfg: dict, backbone_dim: int) -> dict:
    """Map V3A / PH4_v1 / PH4_v2 configs into the current PH4_v2 adapter core.

    Baseline evaluation never uses any PH4 head. For PH4_v1, the historical
    global ph4_head parameters are deliberately ignored while all shared
    encoder/pool/fusion/conditioning parameters are restored.
    """
    sig = inspect.signature(FactorizedDualPocketAdapter.__init__)
    allowed = set(sig.parameters) - {"self"}
    out = {
        k: v for k, v in dict(saved_cfg).items()
        if k in allowed and not k.startswith("ph4_")
    }
    out["backbone_dim"] = int(backbone_dim)
    out["ph4_local_aux_enabled"] = False
    return out


def _load_legacy_core_stage3_model(
    stage3_checkpoint: Path,
    device: torch.device,
):
    """Load V3A/PH4_v1 core adapter into the current wrapper for invariant QC."""
    stage3_checkpoint = Path(stage3_checkpoint).resolve()
    cfg_path = stage3_checkpoint / "stage3_config.json"
    state_path = stage3_checkpoint / "pocket_adapter.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not state_path.exists():
        raise FileNotFoundError(state_path)

    stage3_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    stage2_checkpoint = Path(stage3_cfg["stage2_checkpoint"]).expanduser()
    if not stage2_checkpoint.is_absolute():
        # Old checkpoints normally serialize an absolute Stage2 path. This is
        # only a compatibility fallback.
        stage2_checkpoint = (stage3_checkpoint / stage2_checkpoint).resolve()

    stage2, tokenizer, _, _ = SynthesisAwareModel.from_pretrained(
        stage2_checkpoint, device=device
    )
    stage2.eval()

    adapter_cfg = _core_adapter_kwargs(
        stage3_cfg["adapter"],
        get_backbone_hidden_size(stage2.backbone),
    )
    adapter = FactorizedDualPocketAdapter(**adapter_cfg).to(device)

    raw_state = torch.load(state_path, map_location="cpu")
    # PH4_v1 has an extra global ph4_head.* branch. It is irrelevant to
    # conditional generation and entirely irrelevant to zero/unconditional.
    core_state = {
        k: v for k, v in raw_state.items()
        if not k.startswith("ph4_head.") and not k.startswith("ph4_local_head.")
    }
    load_result = adapter.load_state_dict(core_state, strict=False)
    bad_missing = [
        k for k in load_result.missing_keys
        if not k.startswith("ph4_local_head.")
    ]
    bad_unexpected = [
        k for k in load_result.unexpected_keys
        if not k.startswith("ph4_head.")
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Core adapter load mismatch for {stage3_checkpoint}: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )

    model = PocketConditionedSynthesisModel(
        stage2_model=stage2,
        pocket_adapter=adapter,
        product_tokenizer=tokenizer,
        conditioning_mode=str(stage3_cfg.get("conditioning_mode", "position_preserving")),
    ).to(device)
    model.eval()
    return model, stage2_checkpoint, stage3_cfg


def _build_fresh_ph4v2_model(
    stage2_checkpoint: Path,
    adapter_cfg: dict,
    adapter_state: dict,
    product_tokenizer,
    device: torch.device,
    conditioning_mode: str,
):
    """Fresh Stage2 + supplied PH4_v2 adapter state."""
    stage2, tokenizer, _, _ = SynthesisAwareModel.from_pretrained(
        stage2_checkpoint, device=device
    )
    stage2.eval()

    cfg = dict(adapter_cfg)
    cfg["backbone_dim"] = get_backbone_hidden_size(stage2.backbone)
    adapter = FactorizedDualPocketAdapter(**cfg).to(device)
    adapter.load_state_dict(adapter_state, strict=True)

    model = PocketConditionedSynthesisModel(
        stage2_model=stage2,
        pocket_adapter=adapter,
        product_tokenizer=tokenizer,
        conditioning_mode=conditioning_mode,
    ).to(device)
    model.eval()
    return model


def _baseline_frame_from_eval(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dual_target_record_id",
        "ligand_strict_key",
        "unconditional_nll",
        "zero_condition_nll",
    ]
    out = frame[cols].copy()
    if out["dual_target_record_id"].duplicated().any():
        raise RuntimeError("Evaluation baseline frame contains duplicate record IDs.")
    return out


def _baseline_summary(frame: pd.DataFrame) -> dict:
    lig = frame.groupby("ligand_strict_key", sort=True)[
        ["unconditional_nll", "zero_condition_nll"]
    ].mean(numeric_only=True)
    return {
        "record_micro": {
            "n_records": int(len(frame)),
            "unconditional_nll": float(frame["unconditional_nll"].mean()),
            "zero_condition_nll": float(frame["zero_condition_nll"].mean()),
            "zero_penalty": float(
                (frame["zero_condition_nll"] - frame["unconditional_nll"]).mean()
            ),
        },
        "ligand_macro": {
            "n_ligands": int(len(lig)),
            "unconditional_nll": float(lig["unconditional_nll"].mean()),
            "zero_condition_nll": float(lig["zero_condition_nll"].mean()),
            "zero_penalty": float(
                (lig["zero_condition_nll"] - lig["unconditional_nll"]).mean()
            ),
        },
    }


def _compare_baselines(
    left: pd.DataFrame,
    right: pd.DataFrame,
    atol: float,
    label: str,
) -> dict:
    a = left[
        ["dual_target_record_id", "unconditional_nll", "zero_condition_nll"]
    ].copy()
    b = right[
        ["dual_target_record_id", "unconditional_nll", "zero_condition_nll"]
    ].copy()
    merged = a.merge(
        b,
        on="dual_target_record_id",
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        bad = merged.loc[
            merged["_merge"] != "both",
            ["dual_target_record_id", "_merge"],
        ]
        raise RuntimeError(
            f"{label}: record mismatch; "
            f"examples={bad.head(20).to_dict(orient='records')}"
        )

    report = {
        "label": label,
        "atol": float(atol),
        "n_records": int(len(merged)),
    }
    passed = True
    for col in ("unconditional_nll", "zero_condition_nll"):
        diff = (
            merged[f"{col}_left"].astype(float)
            - merged[f"{col}_right"].astype(float)
        ).abs()
        report[col] = {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "n_above_atol": int((diff > float(atol)).sum()),
        }
        passed = passed and bool((diff <= float(atol)).all())
    report["pass"] = bool(passed)
    return report


def _historical_baseline_snapshot(metrics_path: Path | None) -> dict | None:
    if metrics_path is None or not Path(metrics_path).exists():
        return None
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    out = {}
    for split in ("val", "test"):
        if split not in data:
            continue
        out[split] = {}
        for agg in ("record_micro", "ligand_macro"):
            block = data[split].get(agg, {})
            out[split][agg] = {
                "unconditional_nll": block.get("unconditional_nll"),
                "zero_condition_nll": block.get("zero_condition_nll"),
            }
    return out


def _run_triple_baseline_preflight(
    *,
    v3a_checkpoint: Path,
    ph4v1_checkpoint: Path,
    ph4v2_stage2_checkpoint: Path,
    ph4v2_adapter_cfg: dict,
    ph4v2_adapter_state: dict,
    product_tokenizer,
    loader,
    device: torch.device,
    conditioning_mode: str,
    atol: float,
    output_dir: Path,
    historical_v3a_metrics: Path | None = None,
    historical_ph4v1_metrics: Path | None = None,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Re-evaluate V3A, PH4_v1 and pre-train PH4_v2 under ONE evaluator.

    The unchanged evaluate_model() is used for all three models on the exact
    same test DataLoader. Therefore the comparison includes the same legacy
    forward order and batch composition. Training may start only if both
    zero-condition and unconditional per-record NLLs agree across all three.
    """
    frames: dict[str, pd.DataFrame] = {}
    hashes = {}
    stage2_paths = {}

    # V3A
    v3a_model, v3a_stage2_path, _ = _load_legacy_core_stage3_model(
        v3a_checkpoint, device
    )
    v3a_summary, v3a_full = evaluate_model(v3a_model, loader, device)
    frames["v3a"] = _baseline_frame_from_eval(v3a_full)
    hashes["v3a"] = module_state_sha256(v3a_model.stage2)
    stage2_paths["v3a"] = str(v3a_stage2_path)
    del v3a_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # PH4_v1
    ph4v1_model, ph4v1_stage2_path, _ = _load_legacy_core_stage3_model(
        ph4v1_checkpoint, device
    )
    ph4v1_summary, ph4v1_full = evaluate_model(ph4v1_model, loader, device)
    frames["ph4_v1"] = _baseline_frame_from_eval(ph4v1_full)
    hashes["ph4_v1"] = module_state_sha256(ph4v1_model.stage2)
    stage2_paths["ph4_v1"] = str(ph4v1_stage2_path)
    del ph4v1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # PH4_v2 candidate before any optimization. Use a fresh Stage2 and a copy
    # of the exact adapter initialization that will be trained.
    ph4v2_model = _build_fresh_ph4v2_model(
        stage2_checkpoint=ph4v2_stage2_checkpoint,
        adapter_cfg=ph4v2_adapter_cfg,
        adapter_state=ph4v2_adapter_state,
        product_tokenizer=product_tokenizer,
        device=device,
        conditioning_mode=conditioning_mode,
    )
    ph4v2_summary, ph4v2_full = evaluate_model(ph4v2_model, loader, device)
    frames["ph4_v2_pretrain"] = _baseline_frame_from_eval(ph4v2_full)
    hashes["ph4_v2_pretrain"] = module_state_sha256(ph4v2_model.stage2)
    stage2_paths["ph4_v2_pretrain"] = str(Path(ph4v2_stage2_checkpoint).resolve())
    del ph4v2_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for name, frame in frames.items():
        frame.to_csv(output_dir / f"baseline_preflight_{name}_per_record.csv", index=False)

    comparisons = {
        "v3a_vs_ph4_v1": _compare_baselines(
            frames["v3a"], frames["ph4_v1"], atol, "v3a_vs_ph4_v1"
        ),
        "v3a_vs_ph4_v2_pretrain": _compare_baselines(
            frames["v3a"], frames["ph4_v2_pretrain"], atol, "v3a_vs_ph4_v2_pretrain"
        ),
        "ph4_v1_vs_ph4_v2_pretrain": _compare_baselines(
            frames["ph4_v1"], frames["ph4_v2_pretrain"], atol, "ph4_v1_vs_ph4_v2_pretrain"
        ),
    }

    same_stage2_hash = len(set(hashes.values())) == 1
    report = {
        "protocol": (
            "V3A, PH4_v1 and pre-train PH4_v2 re-evaluated with the same "
            "unchanged evaluate_model(), same test DataLoader, same batch size/order, "
            "same CUDA device and fresh Stage2 loads."
        ),
        "stage2_paths": stage2_paths,
        "stage2_hashes": hashes,
        "same_stage2_hash": bool(same_stage2_hash),
        "summaries": {
            name: _baseline_summary(frame)
            for name, frame in frames.items()
        },
        "comparisons": comparisons,
        "historical_metrics_diagnostic_only": {
            "v3a": _historical_baseline_snapshot(historical_v3a_metrics),
            "ph4_v1": _historical_baseline_snapshot(historical_ph4v1_metrics),
        },
    }
    report["pass"] = bool(
        same_stage2_hash
        and all(x["pass"] for x in comparisons.values())
    )
    save_json(report, output_dir / "triple_baseline_preflight_report.json")
    return report, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_pocket.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    seed = int(cfg.get("seed", 20260812))
    seed_everything(seed)
    device = resolve_device(cfg.get("device", "auto"))

    stage2_checkpoint = resolve_path(cfg["model"]["stage2_checkpoint"], base)
    split_root = resolve_path(cfg["data"]["split_root"], base)
    pocket_csv = resolve_path(cfg["data"]["pocket_context_csv"], base)
    output_dir = resolve_path(cfg["model"]["stage3_output_dir"], base)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = cfg.get("baseline_guard", {})
    baseline_enabled = bool(baseline_cfg.get("enabled", True))
    baseline_atol = float(baseline_cfg.get("atol", 1e-6))
    baseline_abort = bool(baseline_cfg.get("abort_on_mismatch", True))
    v3a_checkpoint = (
        resolve_path(baseline_cfg["v3a_checkpoint"], base)
        if baseline_cfg.get("v3a_checkpoint") else None
    )
    ph4v1_checkpoint = (
        resolve_path(baseline_cfg["ph4_v1_checkpoint"], base)
        if baseline_cfg.get("ph4_v1_checkpoint") else None
    )
    historical_v3a_metrics = (
        resolve_path(baseline_cfg["historical_v3a_metrics"], base)
        if baseline_cfg.get("historical_v3a_metrics") else None
    )
    historical_ph4v1_metrics = (
        resolve_path(baseline_cfg["historical_ph4_v1_metrics"], base)
        if baseline_cfg.get("historical_ph4_v1_metrics") else None
    )

    ph4_cfg = cfg.get("ph4_v2_auxiliary", {})
    ph4_enabled = bool(ph4_cfg.get("enabled", True))
    ph4_lambda = float(ph4_cfg.get("lambda", 0.01))
    negative_ph4_policy = str(ph4_cfg.get("negative_policy", "positive_only")).lower()
    local_positive_fraction_weight = float(ph4_cfg.get("positive_fraction_weight", 0.5))
    if negative_ph4_policy not in {"positive_only", "own_label"}:
        raise ValueError("ph4_v2_auxiliary.negative_policy must be 'positive_only' or 'own_label'")
    ph4_target_csv = None
    if ph4_enabled:
        if "ph4_local_target_csv" not in cfg.get("data", {}):
            raise KeyError("PH4_v2 local auxiliary training requires data.ph4_local_target_csv")
        ph4_target_csv = resolve_path(cfg["data"]["ph4_local_target_csv"], base)

    stage2, product_tokenizer, _, _ = SynthesisAwareModel.from_pretrained(
        stage2_checkpoint, device=device
    )
    stage2.eval()

    adapter_cfg = dict(cfg["adapter"])
    # Keep architecture serialized in the adapter config so save/load remains exact.
    adapter_cfg["ph4_local_aux_enabled"] = bool(ph4_enabled)
    adapter_cfg["ph4_local_num_types"] = int(ph4_cfg.get("num_types", adapter_cfg.get("ph4_local_num_types", 6)))
    adapter_cfg["ph4_local_hidden_dim"] = int(ph4_cfg.get("hidden_dim", adapter_cfg.get("ph4_local_hidden_dim", 256)))
    adapter_cfg["ph4_local_dropout"] = float(ph4_cfg.get("dropout", adapter_cfg.get("ph4_local_dropout", 0.0)))
    adapter_cfg["backbone_dim"] = get_backbone_hidden_size(stage2.backbone)
    adapter = FactorizedDualPocketAdapter(**adapter_cfg).to(device)
    model = PocketConditionedSynthesisModel(
        stage2_model=stage2,
        pocket_adapter=adapter,
        product_tokenizer=product_tokenizer,
        conditioning_mode=cfg["model"].get("conditioning_mode", "position_preserving"),
    ).to(device)

    # Keep the exact initialization for PH4_v2 preflight and frozen-state audit.
    ph4v2_initial_adapter_state = {
        k: v.detach().cpu().clone()
        for k, v in model.pocket_adapter.state_dict().items()
    }
    stage2_hash_before_training = module_state_sha256(model.stage2)

    store = PocketStore(
        pocket_csv,
        max_residues=cfg["pocket"].get("max_residues", 128),
        use_core_flag=cfg["pocket"].get("use_core_flag", False),
    )
    ph4_store = PH4LocalTargetStore(ph4_target_csv) if ph4_enabled else None
    indexes = {}
    reports = {}
    for split in ("train", "val", "test"):
        indexes[split], reports[split] = build_stage3_index(
            split_root,
            split,
            store,
            strict_metadata=bool(cfg["data"].get("strict_pocket_metadata", True)),
        )
        indexes[split].to_csv(output_dir / f"stage3_{split}_index.csv", index=False)
    save_json(reports, output_dir / "alignment_reports.json")

    # V3 does not consume K virtual positions. The full Stage-2 positional
    # capacity remains available to the original SMILES token sequence.
    max_pos = get_backbone_max_positions(stage2.backbone)
    max_smiles_cfg = int(cfg["training"].get("max_smiles_length", max_pos))
    max_smiles_length = min(max_smiles_cfg, max_pos)
    if max_smiles_length < 2:
        raise ValueError("Invalid max_smiles_length")

    if not ph4_enabled:
        raise ValueError("This PH4_v2 training script expects ph4_v2_auxiliary.enabled=true; use the V3A script for no-PH4 ablations.")
    collator = Stage3PocketCollatorPH4V2(product_tokenizer, max_smiles_length=max_smiles_length)
    datasets = {}
    loaders = {}
    for split in ("train", "val", "test"):
        datasets[split] = Stage3PocketDatasetPH4V2(
            indexes[split],
            store,
            ph4_store=ph4_store,
            negative_ph4_policy=negative_ph4_policy,
            include_hard_negatives=True,
            negative_seed=seed + {"train": 0, "val": 10000, "test": 20000}[split],
        )
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(cfg["training"].get("num_workers", 0)),
            pin_memory=(device.type == "cuda"),
            collate_fn=collator,
            drop_last=False,
        )

    negative_report = {
        split: datasets[split].negative_sampler.coverage_report(epoch=0)
        for split in ("train", "val", "test")
    }
    save_json(negative_report, output_dir / "hard_negative_coverage.json")
    print("Hard-negative coverage:", json.dumps(negative_report, ensure_ascii=False))

    ph4_coverage_report = {
        split: datasets[split].ph4_coverage_report(epoch=0)
        for split in ("train", "val", "test")
    }
    save_json(ph4_coverage_report, output_dir / "ph4_v2_local_label_and_negative_coverage.json")
    print("PH4_v2 local coverage / negative audit:", json.dumps(ph4_coverage_report, ensure_ascii=False))

    # PH4_v2.2: derive static PH4-type weights from TRAIN positive pockets only.
    # Validation/test PH4 label statistics are never used for optimization.
    train_positive_entries = []
    for row in indexes["train"].itertuples(index=False):
        train_positive_entries.append(store[int(getattr(row, "pair_id_a"))])
        train_positive_entries.append(store[int(getattr(row, "pair_id_b"))])

    ph4_type_weight_report = ph4_store.sqrt_coverage_type_weights(train_positive_entries)
    ph4_type_weights = ph4_type_weight_report["weights"]
    save_json(ph4_type_weight_report, output_dir / "ph4_v2_sqrt_coverage_type_weights.json")
    print(
        "PH4_v2 sqrt-coverage type weights (train positive only):",
        json.dumps(ph4_type_weight_report, ensure_ascii=False),
    )

    preflight_report = None
    preflight_frames = None
    if baseline_enabled:
        if v3a_checkpoint is None or ph4v1_checkpoint is None:
            raise ValueError(
                "baseline_guard.enabled=true requires both "
                "baseline_guard.v3a_checkpoint and baseline_guard.ph4_v1_checkpoint"
            )
        datasets["test"].set_epoch(0)
        preflight_report, preflight_frames = _run_triple_baseline_preflight(
            v3a_checkpoint=v3a_checkpoint,
            ph4v1_checkpoint=ph4v1_checkpoint,
            ph4v2_stage2_checkpoint=stage2_checkpoint,
            ph4v2_adapter_cfg=adapter_cfg,
            ph4v2_adapter_state=ph4v2_initial_adapter_state,
            product_tokenizer=product_tokenizer,
            loader=loaders["test"],
            device=device,
            conditioning_mode=cfg["model"].get("conditioning_mode", "position_preserving"),
            atol=baseline_atol,
            output_dir=output_dir,
            historical_v3a_metrics=historical_v3a_metrics,
            historical_ph4v1_metrics=historical_ph4v1_metrics,
        )
        print("Triple baseline preflight:", json.dumps(preflight_report, ensure_ascii=False, indent=2))
        if baseline_abort and not preflight_report["pass"]:
            raise RuntimeError(
                "PH4_v2.4 triple-baseline preflight failed: V3A, PH4_v1 and "
                "pre-train PH4_v2 do not share the same zero/unconditional baseline "
                "under one identical evaluation protocol. Training aborted before epoch 1. "
                f"See {output_dir / 'triple_baseline_preflight_report.json'}"
            )

        # Reference-model loading/evaluation consumes RNG. Restore the original
        # training seed so PH4_v2 optimization keeps the intended deterministic
        # train-order schedule. Adapter weights themselves were never modified.
        seed_everything(seed)

    optimizer = torch.optim.AdamW(
        model.pocket_adapter.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )

    # Keep FP32 by default: the real GP-MoLFormer-Uniq Stage-3 run produced
    # non-finite train losses under FP16 autocast.
    use_amp = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    epochs = int(cfg["training"].get("epochs", 30))
    patience = int(cfg["training"].get("early_stopping_patience", 6))

    rank_cfg = cfg.get("ranking", {})
    rank_enabled = bool(rank_cfg.get("enabled", True))
    rank_lambda = float(rank_cfg.get("lambda", 0.2))
    rank_margin = float(rank_cfg.get("margin", 0.05))
    rank_types = [str(x).lower() for x in rank_cfg.get("types", ["a", "b", "both"])]
    allowed = {"a", "b", "both"}
    if any(x not in allowed for x in rank_types):
        raise ValueError(f"ranking.types must be a subset of {sorted(allowed)}")

    single_cfg = cfg.get("single_grounding", {})
    single_enabled = bool(single_cfg.get("enabled", True))
    single_lambda = float(single_cfg.get("lambda", 0.2))
    single_sides = [str(x).lower() for x in single_cfg.get("sides", ["a", "b"])]
    if any(x not in {"a", "b"} for x in single_sides):
        raise ValueError("single_grounding.sides must be a subset of ['a', 'b']")

    # OOD-motivated curriculum: ground reusable individual-pocket semantics
    # briefly before applying the full joint + ranking objective. Set
    # warmup_epochs=0 for an ablation without this curriculum.
    warmup_epochs = int(single_cfg.get("warmup_epochs", 3))
    warmup_joint_weight = float(single_cfg.get("warmup_joint_weight", 0.25))
    warmup_single_lambda = float(single_cfg.get("warmup_single_lambda", 1.0))
    warmup_rank_weight = float(single_cfg.get("warmup_rank_weight", 0.0))

    # V3.1: make the reusable A/B grounding branches explicitly discriminative.
    single_rank_cfg = cfg.get("single_ranking", {})
    single_rank_enabled = bool(single_rank_cfg.get("enabled", True))
    single_rank_lambda = float(single_rank_cfg.get("lambda", 0.20))
    single_rank_margin = float(single_rank_cfg.get("margin", 0.05))
    single_rank_sides = [str(x).lower() for x in single_rank_cfg.get("sides", ["a", "b"])]
    if any(x not in {"a", "b"} for x in single_rank_sides):
        raise ValueError("single_ranking.sides must be a subset of ['a', 'b']")
    single_rank_warmup_weight = float(single_rank_cfg.get("warmup_weight", 0.5))

    best_val = float("inf")
    bad_epochs = 0
    log_path = output_dir / "train_log.jsonl"
    if log_path.exists() and bool(cfg["training"].get("overwrite_train_log", True)):
        log_path.unlink()

    for epoch in range(1, epochs + 1):
        datasets["train"].set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        in_warmup = single_enabled and epoch <= warmup_epochs
        joint_weight = warmup_joint_weight if in_warmup else 1.0
        effective_rank_lambda = rank_lambda * (warmup_rank_weight if in_warmup else 1.0)
        effective_single_lambda = warmup_single_lambda if in_warmup else single_lambda
        effective_single_rank_lambda = single_rank_lambda * (single_rank_warmup_weight if in_warmup else 1.0)

        sums = {
            "joint": 0.0,
            "single_a": 0.0,
            "single_b": 0.0,
            "single_aux": 0.0,
            "rank": 0.0,
            "single_rank": 0.0,
            "ph4_weighted": 0.0,
            "ph4_dims": 0,
            "total": 0.0,
            "delta_a": 0.0,
            "delta_b": 0.0,
            "delta_both": 0.0,
            "single_delta_a": 0.0,
            "single_delta_b": 0.0,
        }
        seen = 0

        for step, batch in enumerate(loaders["train"], start=1):
            batch = move_batch_to_device(batch, device)
            bs = batch["product_input_ids"].shape[0]

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Encode the positive A/B pockets once and reuse the factorized
                # state for joint and A-only/B-only auxiliary supervision.
                state = model.build_condition_state(batch["pocket_a"], batch["pocket_b"])

                # PH4_v2 is residue-local auxiliary supervision only. The shared
                # head reads invariant residue embeddings before query pooling.
                # Targets are never fed into the generator. Types absent from the
                # reference ligand are masked, while local zero residues within a
                # present type provide spatial negative supervision.
                ph4_terms = [
                    (
                        model.pocket_adapter.predict_local_ph4_from_state(state, "a"),
                        batch["ph4_local_target_a"], batch["ph4_local_mask_a"],
                    ),
                    (
                        model.pocket_adapter.predict_local_ph4_from_state(state, "b"),
                        batch["ph4_local_target_b"], batch["ph4_local_mask_b"],
                    ),
                ]

                pos = model.forward_smiles_from_state(
                    batch["product_input_ids"],
                    batch["product_attention_mask"],
                    batch["product_labels"],
                    state,
                    condition="dual",
                )
                joint_loss = pos["loss"]
                pos_nll = pos["per_record_nll"]

                single_outputs = {}
                single_terms = []
                if single_enabled:
                    for side in single_sides:
                        single_outputs[side] = model.forward_single_smiles(
                            batch["product_input_ids"],
                            batch["product_attention_mask"],
                            batch["product_labels"],
                            state,
                            side=side,
                        )
                        single_terms.append(single_outputs[side]["loss"])
                    single_loss = mean_tensor(single_terms)
                else:
                    single_loss = joint_loss.new_zeros(())

                neg_outputs = {}
                neg_states = {}
                rank_terms = []
                need_neg_a = (
                    (rank_enabled and "a" in rank_types)
                    or (single_rank_enabled and "a" in single_rank_sides)
                    or (negative_ph4_policy == "own_label")
                )
                need_neg_b = (
                    (rank_enabled and "b" in rank_types)
                    or (single_rank_enabled and "b" in single_rank_sides)
                    or (negative_ph4_policy == "own_label")
                )

                if need_neg_a:
                    neg_a_out = model.forward_smiles(
                        batch["product_input_ids"], batch["product_attention_mask"], batch["product_labels"],
                        batch["pocket_a_neg"], batch["pocket_b"],
                    )
                    neg_outputs["a"] = neg_a_out["per_record_nll"]
                    neg_states["a"] = neg_a_out["state"]
                    if rank_enabled and "a" in rank_types:
                        rank_terms.append(ranking_hinge(pos_nll, neg_outputs["a"], rank_margin))

                if need_neg_b:
                    neg_b_out = model.forward_smiles(
                        batch["product_input_ids"], batch["product_attention_mask"], batch["product_labels"],
                        batch["pocket_a"], batch["pocket_b_neg"],
                    )
                    neg_outputs["b"] = neg_b_out["per_record_nll"]
                    neg_states["b"] = neg_b_out["state"]
                    if rank_enabled and "b" in rank_types:
                        rank_terms.append(ranking_hinge(pos_nll, neg_outputs["b"], rank_margin))

                if (rank_enabled and "both" in rank_types) or negative_ph4_policy == "own_label":
                    neg_both_out = model.forward_smiles(
                        batch["product_input_ids"], batch["product_attention_mask"], batch["product_labels"],
                        batch["pocket_both_a_neg"], batch["pocket_both_b_neg"],
                    )
                    neg_outputs["both"] = neg_both_out["per_record_nll"]
                    neg_states["both"] = neg_both_out["state"]
                    if rank_enabled and "both" in rank_types:
                        rank_terms.append(ranking_hinge(pos_nll, neg_outputs["both"], rank_margin))
                rank_loss = mean_tensor(rank_terms) if rank_terms else joint_loss.new_zeros(())

                single_neg_outputs = {}
                single_rank_terms = []
                if single_rank_enabled:
                    for side in single_rank_sides:
                        if side not in single_outputs:
                            raise RuntimeError(f"single_ranking side={side!r} requires single_grounding for that side")
                        single_neg_outputs[side] = model.forward_single_smiles(
                            batch["product_input_ids"], batch["product_attention_mask"], batch["product_labels"],
                            neg_states[side], side=side,
                        )["per_record_nll"]
                        single_rank_terms.append(ranking_hinge(
                            single_outputs[side]["per_record_nll"], single_neg_outputs[side], single_rank_margin
                        ))
                    single_rank_loss = mean_tensor(single_rank_terms) if single_rank_terms else joint_loss.new_zeros(())
                else:
                    single_rank_loss = joint_loss.new_zeros(())

                # Formal PH4_v2 uses positive_only. Optional own_label is an
                # ablation only: a hard-negative pocket may receive its OWN local
                # pair-specific target, never the current positive ligand's target.
                if negative_ph4_policy == "own_label":
                    ph4_terms.extend([
                        (model.pocket_adapter.predict_local_ph4_from_state(neg_states["a"], "a"),
                         batch["ph4_local_target_a_neg"], batch["ph4_local_mask_a_neg"]),
                        (model.pocket_adapter.predict_local_ph4_from_state(neg_states["b"], "b"),
                         batch["ph4_local_target_b_neg"], batch["ph4_local_mask_b_neg"]),
                        (model.pocket_adapter.predict_local_ph4_from_state(neg_states["both"], "a"),
                         batch["ph4_local_target_both_a_neg"], batch["ph4_local_mask_both_a_neg"]),
                        (model.pocket_adapter.predict_local_ph4_from_state(neg_states["both"], "b"),
                         batch["ph4_local_target_both_b_neg"], batch["ph4_local_mask_both_b_neg"]),
                    ])
                ph4_loss, ph4_stats = balanced_local_ph4_bce(
                    ph4_terms,
                    positive_fraction_weight=local_positive_fraction_weight,
                    type_weights=ph4_type_weights,
                )
                ph4_n_dims = int(ph4_stats["n_supervised"])

                total_loss = (
                    joint_weight * joint_loss
                    + effective_single_lambda * single_loss
                    + effective_rank_lambda * rank_loss
                    + effective_single_rank_lambda * single_rank_loss
                    + ph4_lambda * ph4_loss
                )
                scaled_loss = total_loss / grad_accum

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    "Non-finite Stage-3 PH4_v2 loss detected. "
                    f"epoch={epoch} step={step} records={batch['dual_target_record_id']} "
                    f"ligands={batch['ligand_strict_key']} joint={float(joint_loss.detach())} "
                    f"single={float(single_loss.detach())} rank={float(rank_loss.detach())} "
                    f"single_rank={float(single_rank_loss.detach())} "
                    f"ph4={float(ph4_loss.detach())} ph4_dims={ph4_n_dims}"
                )

            scaler.scale(scaled_loss).backward()
            if step % grad_accum == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.pocket_adapter.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            sums["joint"] += float(joint_loss.detach()) * bs
            if "a" in single_outputs:
                sums["single_a"] += float(single_outputs["a"]["loss"].detach()) * bs
            if "b" in single_outputs:
                sums["single_b"] += float(single_outputs["b"]["loss"].detach()) * bs
            sums["single_aux"] += float(single_loss.detach()) * bs
            sums["rank"] += float(rank_loss.detach()) * bs
            sums["single_rank"] += float(single_rank_loss.detach()) * bs
            sums["ph4_weighted"] += float(ph4_loss.detach()) * bs
            sums["ph4_dims"] += int(ph4_n_dims)
            sums["total"] += float(total_loss.detach()) * bs
            for name in ["a", "b", "both"]:
                if name in neg_outputs:
                    sums[f"delta_{name}"] += float((neg_outputs[name] - pos_nll).detach().mean()) * bs
            for name in ["a", "b"]:
                if name in single_neg_outputs:
                    sums[f"single_delta_{name}"] += float((
                        single_neg_outputs[name] - single_outputs[name]["per_record_nll"]
                    ).detach().mean()) * bs
            seen += bs

        datasets["val"].set_epoch(0)
        val_summary, _ = evaluate_model(model, loaders["val"], device)
        gates = model.pocket_adapter.gate_values()
        record = {
            "epoch": epoch,
            "warmup": bool(in_warmup),
            "joint_weight": joint_weight,
            "single_lambda": effective_single_lambda,
            "rank_lambda": effective_rank_lambda,
            "single_rank_lambda": effective_single_rank_lambda,
            "ph4_v2_lambda": ph4_lambda,
            "ph4_v2_positive_fraction_weight": local_positive_fraction_weight,
            "ph4_v2_type_weighting": "sqrt_coverage_train_positive_only",
            "ph4_v2_type_weights": ph4_type_weight_report["weights_by_type"],
            "ph4_v2_negative_policy": negative_ph4_policy,
            "train_joint_nll": sums["joint"] / max(seen, 1),
            "train_single_a_nll": sums["single_a"] / max(seen, 1) if "a" in single_sides else None,
            "train_single_b_nll": sums["single_b"] / max(seen, 1) if "b" in single_sides else None,
            "train_single_aux_loss": sums["single_aux"] / max(seen, 1),
            "train_rank_loss": sums["rank"] / max(seen, 1),
            "train_single_rank_loss": sums["single_rank"] / max(seen, 1),
            "train_ph4_v2_local_loss": (
                sums["ph4_weighted"] / max(seen, 1)
            ),
            "train_ph4_v2_supervised_dimensions": int(sums["ph4_dims"]),
            "train_total_loss": sums["total"] / max(seen, 1),
            "train_delta_a_nll": sums["delta_a"] / max(seen, 1),
            "train_delta_b_nll": sums["delta_b"] / max(seen, 1),
            "train_delta_both_nll": sums["delta_both"] / max(seen, 1),
            "train_single_delta_a_nll": sums["single_delta_a"] / max(seen, 1),
            "train_single_delta_b_nll": sums["single_delta_b"] / max(seen, 1),
            **gates,
            "val": val_summary,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        # Warm-up epochs are grounding-only curriculum and are not eligible
        # for final checkpoint selection. This prevents an early single-pocket
        # checkpoint from winning simply because its objective differs from the
        # full PH4_v2 objective.
        if in_warmup:
            continue

        current = selection_value(val_summary, cfg)
        if current < best_val:
            best_val = current
            bad_epochs = 0
            model.save_adapter(
                output_dir / "best",
                stage2_checkpoint=stage2_checkpoint,
                adapter_config={k: v for k, v in adapter_cfg.items() if k != "backbone_dim"},
                extra_config={
                    "stage3_variant": "ph4_v2_local_propensity",
                    "data": {
                        "split_root": str(split_root),
                        "pocket_context_csv": str(pocket_csv),
                        "ph4_local_target_csv": str(ph4_target_csv),
                    },
                    "ph4_v2_auxiliary": {
                        **ph4_cfg,
                        "negative_policy": negative_ph4_policy,
                        "positive_fraction_weight": local_positive_fraction_weight,
                        "type_weighting": "sqrt_coverage_train_positive_only",
                        "derived_type_weights": ph4_type_weight_report["weights_by_type"],
                        "derived_type_supervised_counts": ph4_type_weight_report["n_supervised_by_type"],
                    },
                    "negative_ph4_v2_semantics": (
                        "Formal PH4_v2: positive_only. Hard negatives are ranking controls; "
                        "PH4 is never copied from the current positive ligand onto a negative pocket. "
                        "Optional own_label uses only the negative pair's own annotation."
                    ),
                    "ranking": rank_cfg,
                    "single_grounding": single_cfg,
                    "single_ranking": single_rank_cfg,
                    "model_selection": cfg.get("model_selection", {}),
                    "best_epoch": epoch,
                    "best_selection_value": current,
                    "best_val_metrics": val_summary,
                },
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    best_path = output_dir / "best" / "pocket_adapter.pt"
    if not best_path.exists():
        raise RuntimeError(
            "No PH4_v2 checkpoint was eligible for selection. Ensure training.epochs "
            "is greater than single_grounding.warmup_epochs."
        )
    best_state = torch.load(best_path, map_location="cpu")

    # Frozen Stage2 state must remain byte-identical through training.
    stage2_hash_after_training = module_state_sha256(model.stage2)
    frozen_stage2_report = {
        "before_training": stage2_hash_before_training,
        "after_training": stage2_hash_after_training,
        "match": bool(stage2_hash_before_training == stage2_hash_after_training),
    }
    save_json(frozen_stage2_report, output_dir / "stage2_frozen_state_audit.json")
    if baseline_abort and not frozen_stage2_report["match"]:
        raise RuntimeError(
            "Frozen Stage2 state changed during PH4_v2 training. "
            "See stage2_frozen_state_audit.json"
        )

    final = {
        "stage3_variant": "ph4_v2_local_propensity_v2_4_triple_baseline_guard",
        "adapter_type": model.adapter_type,
        "conditioning_mode": model.conditioning_mode,
        "max_smiles_length": max_smiles_length,
        "ranking": rank_cfg,
        "single_grounding": single_cfg,
        "single_ranking": single_rank_cfg,
        "ph4_v2_auxiliary": {
            **ph4_cfg,
            "negative_policy": negative_ph4_policy,
            "positive_fraction_weight": local_positive_fraction_weight,
            "type_weighting": "sqrt_coverage_train_positive_only",
            "derived_type_weights": ph4_type_weight_report["weights_by_type"],
            "derived_type_supervised_counts": ph4_type_weight_report["n_supervised_by_type"],
        },
        "model_selection": cfg.get("model_selection", {}),
        "baseline_guard": {
            "enabled": baseline_enabled,
            "atol": baseline_atol,
            "v3a_checkpoint": str(v3a_checkpoint) if v3a_checkpoint else None,
            "ph4_v1_checkpoint": str(ph4v1_checkpoint) if ph4v1_checkpoint else None,
            "protocol": "same unchanged evaluate_model and same test DataLoader for all versions",
            "preflight_pass": bool(preflight_report["pass"]) if preflight_report else None,
        },
        "stage2_frozen_state_audit": frozen_stage2_report,
    }

    final_test_baseline = None
    for split in ("val", "test"):
        datasets[split].set_epoch(0)

        # Authoritative final evaluation uses a FRESH Stage2 and the UNCHANGED
        # legacy evaluate_model(), exactly like the V3A/PH4_v1 comparison.
        eval_model = _build_fresh_ph4v2_model(
            stage2_checkpoint=stage2_checkpoint,
            adapter_cfg=adapter_cfg,
            adapter_state=best_state,
            product_tokenizer=product_tokenizer,
            device=device,
            conditioning_mode=cfg["model"].get("conditioning_mode", "position_preserving"),
        )
        summary, frame = evaluate_model(eval_model, loaders[split], device)

        final[split] = save_detailed_evaluation(
            summary, frame, output_dir, split
        )
        final[f"{split}_ph4_v2_local_grounding"] = evaluate_local_ph4_grounding(
            eval_model, loaders[split], device, move_batch_to_device
        )

        if split == "test":
            final_test_baseline = _baseline_frame_from_eval(frame)

        del eval_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final PH4_v2 baseline must still equal BOTH same-condition references.
    if baseline_enabled and preflight_frames is not None and final_test_baseline is not None:
        post_report = {
            "final_vs_v3a": _compare_baselines(
                final_test_baseline,
                preflight_frames["v3a"],
                baseline_atol,
                "final_ph4_v2_vs_v3a",
            ),
            "final_vs_ph4_v1": _compare_baselines(
                final_test_baseline,
                preflight_frames["ph4_v1"],
                baseline_atol,
                "final_ph4_v2_vs_ph4_v1",
            ),
        }
        post_report["pass"] = bool(
            post_report["final_vs_v3a"]["pass"]
            and post_report["final_vs_ph4_v1"]["pass"]
        )
        final["baseline_guard"]["post_training"] = post_report
        save_json(post_report, output_dir / "triple_baseline_posttraining_report.json")

        if baseline_abort and not post_report["pass"]:
            save_json(final, output_dir / "final_metrics_baseline_guard_failed.json")
            raise RuntimeError(
                "PH4_v2.4 final baseline differs from same-condition V3A/PH4_v1. "
                "Authoritative final_metrics.json was not written."
            )

    save_json(final, output_dir / "final_metrics.json")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
