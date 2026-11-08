from __future__ import annotations

import argparse
from pathlib import Path
import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from syndual.data import (
    RouteCollator,
    RouteJSONLDataset,
)
from syndual.model_syn_stage2D import SynthesisAwareModel
from syndual.training import (
    build_scheduler,
    evaluate_model_syn,
    train_model_syn_epoch,
)
from syndual.utils import (
    ensure_dir,
    load_yaml,
    make_grad_scaler,
    resolve_device,
    save_json,
    seed_everything,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stage2_joint.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))
    device = resolve_device("auto")

    model, product_tokenizer, route_tokenizer, model_cfg = (
        SynthesisAwareModel.from_pretrained(
            cfg["model"]["checkpoint"],
            device=device,
        )
    )

    model.set_backbone_trainable(True)

    data_root = Path(
        cfg["data"]["root"]
    )

    train_ds = RouteJSONLDataset(
        data_root / cfg["data"]["train_file"],
        smiles_columns=
            cfg["data"]["smiles_columns"],
        route_token_fields=
            cfg["data"].get("route_token_fields"),
        route_model_mask_fields=
            cfg["data"].get("route_model_mask_fields"),
        strict=True,
        require_forward_only=bool(
            cfg["data"].get("require_forward_only", True)
        ),
    )

    val_ds = RouteJSONLDataset(
        data_root / cfg["data"]["val_file"],
        smiles_columns=
            cfg["data"]["smiles_columns"],
        route_token_fields=
            cfg["data"].get("route_token_fields"),
        route_model_mask_fields=
            cfg["data"].get("route_model_mask_fields"),
        strict=True,
        require_forward_only=bool(
            cfg["data"].get("require_forward_only", True)
        ),
    )

    joint_route_loss_mode = str(
        cfg["training"].get(
            "route_loss_mode",
            model_cfg["route"].get("loss_mode", "all_tokens"),
        )
    )
    model_cfg["route"]["loss_mode"] = joint_route_loss_mode

    collator = RouteCollator(
        product_tokenizer=
            product_tokenizer,
        route_tokenizer=
            route_tokenizer,
        max_product_length=int(
            model.backbone.config.
                max_position_embeddings
        ),
        max_route_length=int(
            model_cfg["route"]["max_length"]
        ),
        loss_mode=joint_route_loss_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(
            cfg["training"]["batch_size"]
        ),
        shuffle=True,
        num_workers=int(
            cfg["training"]["num_workers"]
        ),
        pin_memory=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(
            cfg["training"]["eval_batch_size"]
        ),
        shuffle=False,
        num_workers=int(
            cfg["training"]["num_workers"]
        ),
        pin_memory=True,
        collate_fn=collator,
    )

    optimizer = AdamW(
        [
            {
                "params":
                    model.backbone.parameters(),
                "lr":
                    float(
                        cfg["training"][
                            "backbone_learning_rate"
                        ]
                    ),
            },
            {
                "params":
                    model.route_decoder.parameters(),
                "lr":
                    float(
                        cfg["training"][
                            "route_learning_rate"
                        ]
                    ),
            },
        ],
        weight_decay=float(
            cfg["training"]["weight_decay"]
        ),
    )

    grad_accum = int(
        cfg["training"][
            "gradient_accumulation_steps"
        ]
    )

    total_steps = (
        math.ceil(
            len(train_loader)
            / grad_accum
        )
        * int(
            cfg["training"]["epochs"]
        )
    )

    scheduler = build_scheduler(
        optimizer,
        total_steps,
        float(
            cfg["training"]["warmup_ratio"]
        ),
    )

    scaler = make_grad_scaler(
        cfg["training"]["precision"],
        device,
    )

    output_dir = ensure_dir(
        cfg["training"]["output_dir"]
    )

    route_w = float(
        cfg["training"]["route_loss_weight"]
    )

    smiles_w = float(
        cfg["training"]["smiles_loss_weight"]
    )

    best_val = float("inf")
    history = []

    for epoch in range(
        1,
        int(cfg["training"]["epochs"]) + 1,
    ):
        train_metrics = train_model_syn_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            precision=
                cfg["training"]["precision"],
            scaler=scaler,
            gradient_accumulation_steps=
                grad_accum,
            max_grad_norm=float(
                cfg["training"][
                    "max_grad_norm"
                ]
            ),
            route_loss_weight=route_w,
            smiles_loss_weight=smiles_w,
            log_every=int(
                cfg["training"]["log_every"]
            ),
        )

        val_metrics = evaluate_model_syn(
            model=model,
            loader=val_loader,
            device=device,
            precision=
                cfg["training"]["precision"],
            route_loss_weight=route_w,
            smiles_loss_weight=smiles_w,
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(row)
        print(row)

        save_json(
            {"history": history},
            output_dir / "history.json",
        )

        model.save_pretrained(
            output_dir / "last",
            product_tokenizer,
            route_tokenizer,
            model_cfg,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]

            model.save_pretrained(
                output_dir / "best",
                product_tokenizer,
                route_tokenizer,
                model_cfg,
            )

    print(
        "Best joint validation loss:",
        best_val,
    )


if __name__ == "__main__":
    main()
