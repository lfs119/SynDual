from __future__ import annotations

import argparse
from pathlib import Path
import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from syndual.checkpoint import save_stage1_checkpoint
from syndual.data import (
    MoleculeCSVDataset,
    MoleculeCollator,
)
from syndual.molformer import (
    load_gp_molformer,
    load_tokenizer,
)
from syndual.training import (
    build_scheduler,
    evaluate_causal_lm,
    train_causal_lm_epoch,
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
        default="configs/stage1_set2_smiles.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))

    data_root = Path(
        cfg["data"]["root"]
    )

    train_ds = MoleculeCSVDataset(
        data_root / cfg["data"]["train_file"],
        cfg["data"]["smiles_columns"],
    )

    val_ds = MoleculeCSVDataset(
        data_root / cfg["data"]["val_file"],
        cfg["data"]["smiles_columns"],
    )

    tokenizer = load_tokenizer(
        cfg["model"]["tokenizer_model"]
    )

    collator = MoleculeCollator(
        tokenizer,
        max_length=int(
            cfg["model"]["max_length"]
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

    device = resolve_device("auto")

    model = load_gp_molformer(
        cfg["model"]["pretrained_model"],
        deterministic_eval=bool(
            cfg["model"].get(
                "deterministic_eval",
                False,
            )
        ),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
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
        len(train_loader)
        / grad_accum
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

    best_val = float("inf")
    history = []

    for epoch in range(
        1,
        int(cfg["training"]["epochs"]) + 1,
    ):
        train_loss = train_causal_lm_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            precision=cfg["training"]["precision"],
            scaler=scaler,
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=float(
                cfg["training"]["max_grad_norm"]
            ),
            log_every=int(
                cfg["training"]["log_every"]
            ),
        )

        val_loss = evaluate_causal_lm(
            model=model,
            loader=val_loader,
            device=device,
            precision=cfg["training"]["precision"],
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": math.exp(
                min(val_loss, 20)
            ),
        }

        history.append(row)
        print(row)

        save_json(
            {"history": history},
            output_dir / "history.json",
        )

        save_stage1_checkpoint(
            model,
            tokenizer,
            output_dir / "last",
            trainer_state=row,
        )

        if val_loss < best_val:
            best_val = val_loss

            save_stage1_checkpoint(
                model,
                tokenizer,
                output_dir / "best",
                trainer_state={
                    **row,
                    "best_val_loss":
                        best_val,
                },
            )

    print(
        "Best validation loss:",
        best_val,
    )


if __name__ == "__main__":
    main()
