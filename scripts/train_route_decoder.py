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
    build_route_tokenizer,
)
from syndual.model_syn_stage2D import SynthesisAwareModel
from syndual.molformer import (
    load_gp_molformer,
    load_tokenizer,
    resolve_stage1_checkpoint,
)
from syndual.route_decoder import LightweightRouteDecoder
from syndual.route_serialization import ROUTE_SCHEMA_VERSION
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
        default="configs/stage2_route.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))

    device = resolve_device("auto")
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

    model_source, tokenizer_source = (
        resolve_stage1_checkpoint(
            cfg["backbone"]["checkpoint"],
            fallback_model=
                cfg["backbone"][
                    "fallback_pretrained_model"
                ],
            fallback_tokenizer=
                cfg["backbone"][
                    "fallback_tokenizer_model"
                ],
        )
    )

    product_tokenizer = load_tokenizer(
        tokenizer_source
    )

    backbone = load_gp_molformer(
        model_source
    )

    # Build the route tokenizer from the same base tokenizer used by the
    # molecular checkpoint, then add route-control tokens locally.
    route_tokenizer = build_route_tokenizer(
        tokenizer_source
    )

    route_decoder = LightweightRouteDecoder(
        vocab_size=len(route_tokenizer),
        memory_dim=int(
            backbone.config.hidden_size
        ),
        d_model=int(
            cfg["route"]["d_model"]
        ),
        num_layers=int(
            cfg["route"]["num_layers"]
        ),
        num_heads=int(
            cfg["route"]["num_heads"]
        ),
        dim_feedforward=int(
            cfg["route"]["dim_feedforward"]
        ),
        dropout=float(
            cfg["route"]["dropout"]
        ),
        max_length=int(
            cfg["route"]["max_length"]
        ),
        pad_token_id=int(
            route_tokenizer.pad_token_id
        ),
    )

    model = SynthesisAwareModel(
        backbone=backbone,
        route_decoder=route_decoder,
    ).to(device)

    model.set_backbone_trainable(
        not bool(
            cfg["backbone"]["freeze"]
        )
    )

    collator = RouteCollator(
        product_tokenizer=
            product_tokenizer,
        route_tokenizer=
            route_tokenizer,
        max_product_length=int(
            cfg["backbone"][
                "max_product_length"
            ]
        ),
        max_route_length=int(
            cfg["route"]["max_length"]
        ),
        loss_mode=str(
            cfg["route"].get("loss_mode", "all_tokens")
        ),
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

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = AdamW(
        trainable,
        lr=float(
            cfg["training"]["learning_rate"]
        ),
        weight_decay=float(
            cfg["training"]["weight_decay"]
        ),
    )

    grad_accum = int(
        cfg["training"][
            "gradient_accumulation_steps"
        ]
    )

    steps_per_epoch = math.ceil(
        len(train_loader) / grad_accum
    )

    total_steps = (
        steps_per_epoch
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

    checkpoint_config = {
        "model_type":
            "SynDual-model_syn",
        "route": {
            "max_length":
                int(
                    cfg["route"]["max_length"]
                ),
            "d_model":
                int(
                    cfg["route"]["d_model"]
                ),
            "num_layers":
                int(
                    cfg["route"]["num_layers"]
                ),
            "num_heads":
                int(
                    cfg["route"]["num_heads"]
                ),
            "dim_feedforward":
                int(
                    cfg["route"][
                        "dim_feedforward"
                    ]
                ),
            "dropout":
                float(
                    cfg["route"]["dropout"]
                ),
            "loss_mode":
                str(
                    cfg["route"].get("loss_mode", "all_tokens")
                ),
            "schema_version":
                ROUTE_SCHEMA_VERSION,
            "lexicalization":
                "source molecule token -> MoLFormer sub-tokens + <MOL_END>",
            "forward_only":
                bool(
                    cfg["data"].get("require_forward_only", True)
                ),
            "base_tokenizer_source":
                str(tokenizer_source),
        },
    }

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
            route_loss_weight=1.0,
            smiles_loss_weight=0.0,
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
            route_loss_weight=1.0,
            smiles_loss_weight=0.0,
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
            checkpoint_config,
        )

        if (
            val_metrics["route_loss"]
            < best_val
        ):
            best_val = (
                val_metrics[
                    "route_loss"
                ]
            )

            model.save_pretrained(
                output_dir / "best",
                product_tokenizer,
                route_tokenizer,
                checkpoint_config,
            )

    print(
        "Best route validation loss:",
        best_val,
    )


if __name__ == "__main__":
    main()
