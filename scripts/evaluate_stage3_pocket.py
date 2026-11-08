from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from syndual.model_syn_pocket import PocketConditionedSynthesisModel
from syndual.molformer_prefix import get_backbone_max_positions
from syndual.pocket_data import PocketStore, Stage3PocketCollator, Stage3PocketDataset, build_stage3_index
from syndual.stage3_evaluation import evaluate_model, save_detailed_evaluation
from syndual.utils import load_yaml, resolve_device, resolve_path, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_pocket.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    device = resolve_device(cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 20260812))

    stage2_ckpt = resolve_path(cfg["model"]["stage2_checkpoint"], base)
    stage3_ckpt = resolve_path(
        cfg["model"].get("stage3_checkpoint", cfg["model"]["stage3_output_dir"] + "/best"),
        base,
    )
    output_dir = resolve_path(cfg["model"]["stage3_output_dir"], base)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, _, _, _ = PocketConditionedSynthesisModel.from_pretrained(
        stage2_checkpoint=stage2_ckpt,
        stage3_checkpoint=stage3_ckpt,
        device=device,
        conditioning_mode=cfg["model"].get("conditioning_mode", "position_preserving"),
    )

    store = PocketStore(
        resolve_path(cfg["data"]["pocket_context_csv"], base),
        max_residues=cfg["pocket"].get("max_residues", 128),
        use_core_flag=cfg["pocket"].get("use_core_flag", False),
    )
    index, report = build_stage3_index(
        resolve_path(cfg["data"]["split_root"], base),
        args.split,
        store,
        strict_metadata=bool(cfg["data"].get("strict_pocket_metadata", True)),
    )

    dataset = Stage3PocketDataset(
        index,
        store,
        include_hard_negatives=True,
        negative_seed=seed + {"train": 0, "val": 10000, "test": 20000}[args.split],
    )
    dataset.set_epoch(0)
    negative_coverage = dataset.negative_sampler.coverage_report(epoch=0)

    # Position-preserving v3 uses the full original GP-MoLFormer context.
    max_len = min(
        int(cfg["training"].get("max_smiles_length", get_backbone_max_positions(model.backbone))),
        get_backbone_max_positions(model.backbone),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        collate_fn=Stage3PocketCollator(tokenizer, max_len),
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    summary, frame = evaluate_model(model, loader, device)
    summary = save_detailed_evaluation(summary, frame, output_dir, args.split)
    summary["alignment"] = report
    summary["hard_negative_coverage"] = negative_coverage
    summary["adapter_type"] = model.adapter_type
    summary["conditioning_mode"] = model.conditioning_mode
    summary["gates"] = model.pocket_adapter.gate_values()

    out = output_dir / f"eval_{args.split}.json"
    save_json(summary, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved:", out)
    print("Per-record CSV:", summary["output_tables"]["per_record_csv"])
    print("Ligand-macro source CSV:", summary["output_tables"]["by_ligand_csv"])
    print("Target-pair CSV:", summary["output_tables"]["by_target_pair_csv"])


if __name__ == "__main__":
    main()
