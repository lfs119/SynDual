from __future__ import annotations

import argparse
from pathlib import Path
import json

from syndual.bb_constraints import BuildingBlockTokenTrie
from syndual.reactant_constraints import (
    CompositeRouteTokenConstraint,
    CurrentGraphReactantConstraint,
)
from syndual.chemistry import (
    canonicalize_smiles,
    is_valid_smiles,
)
from syndual.model_syn import SynthesisAwareModel
from syndual.molformer import generate_unconditional
from syndual.route_serialization import (
    decode_route_ids_to_source_tokens,
    final_product_from_route,
    format_route_tokens,
    molecule_tokens_from_route,
    route_format_valid,
)
from syndual.utils import (
    ensure_dir,
    load_yaml,
    resolve_device,
    seed_everything,
)


def choose_checkpoint(cfg):
    primary = Path(cfg["model"]["checkpoint"])
    fallback = Path(cfg["model"]["fallback_checkpoint"])

    if primary.exists():
        return primary
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"No model_syn checkpoint found: {primary} or {fallback}"
    )


def _canonical_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    ca = canonicalize_smiles(a, isomeric=True)
    cb = canonicalize_smiles(b, isomeric=True)
    return ca is not None and cb is not None and ca == cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/inference.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))
    device = resolve_device("auto")

    checkpoint = choose_checkpoint(cfg)

    model, product_tokenizer, route_tokenizer, model_cfg = (
        SynthesisAwareModel.from_pretrained(
            checkpoint,
            device=device,
        )
    )
    model.eval()

    requested = int(cfg["generation"]["num_molecules"])
    forward_only = bool(cfg["generation"].get("route_forward_only", True))

    bb_constraint = None
    if bool(cfg["generation"].get("bb_constrained_decoding", True)):
        bb_vocab_path = cfg["generation"].get("building_block_vocab_path")
        if not bb_vocab_path:
            raise ValueError(
                "generation.bb_constrained_decoding=true requires "
                "generation.building_block_vocab_path."
            )
        bb_constraint = BuildingBlockTokenTrie.from_vocab_path(
            tokenizer=route_tokenizer,
            path=bb_vocab_path,
        )
        print("BB-constrained decoding:", bb_constraint.metadata())

    reactant_constraint = None
    if bool(cfg["generation"].get("reactant_graph_constrained_decoding", True)):
        reactant_constraint = CurrentGraphReactantConstraint(
            tokenizer=route_tokenizer,
            terminate_invalid_prefix_with_eos=True,
        )
        print(
            "Reactant-graph constrained decoding:",
            reactant_constraint.metadata(),
        )

    route_token_constraint = CompositeRouteTokenConstraint(
        [bb_constraint, reactant_constraint]
    )

    collected = []
    attempts = 0

    # Oversample in small batches until enough valid molecules are obtained.
    while len(collected) < requested:
        attempts += 1

        if attempts > 50:
            raise RuntimeError(
                "Could not obtain the requested number of generated molecules."
            )

        batch_n = max(16, requested - len(collected))

        smiles_batch = generate_unconditional(
            model.backbone,
            product_tokenizer,
            num_samples=batch_n,
            max_length=int(cfg["generation"]["smiles_max_length"]),
            temperature=float(cfg["generation"]["smiles_temperature"]),
        )

        for smiles in smiles_batch:
            valid = is_valid_smiles(smiles)

            if bool(cfg["generation"]["require_valid_smiles"]) and not valid:
                continue

            collected.append(
                {
                    "smiles": smiles,
                    "valid_smiles": valid,
                }
            )

            if len(collected) >= requested:
                break

    for row in collected:
        enc = product_tokenizer(
            [row["smiles"]],
            add_special_tokens=True,
            truncation=True,
            max_length=int(model.backbone.config.max_position_embeddings),
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        validity_cfg = cfg["generation"].get("validity_aware_decoding", {})

        route_ids, validity_aware_diag = model.generate_route_validity_aware_v2(
            product_input_ids=input_ids,
            product_attention_mask=attention_mask,
            route_tokenizer=route_tokenizer,
            max_length=int(model_cfg["route"]["max_length"]),
            do_sample=bool(cfg["generation"]["route_do_sample"]),
            temperature=float(cfg["generation"]["route_temperature"]),
            top_k=cfg["generation"].get("route_top_k"),
            forward_only=forward_only,
            token_constraint=route_token_constraint,
            product_state_tracker=reactant_constraint,
            primary_do_sample=bool(validity_cfg.get("primary_do_sample", False)),
            repair_beam_width=int(validity_cfg.get("repair_beam_width", 4)),
            repair_candidate_pool_size=int(validity_cfg.get("repair_candidate_pool_size", 8)),
            repair_max_tokens=int(validity_cfg.get("repair_max_tokens", 160)),
            repair_length_penalty=float(validity_cfg.get("repair_length_penalty", 0.6)),
            repair_temperature=float(validity_cfg.get("repair_temperature", 1.0)),
            repair_top_k=validity_cfg.get("repair_top_k"),
        )

        generated_ids = route_ids[0].detach().cpu().tolist()
        reactant_reference_audit = (
            reactant_constraint.audit_generated_ids(generated_ids)
            if reactant_constraint is not None
            else None
        )

        decode_error = None
        try:
            route_tokens = decode_route_ids_to_source_tokens(
                generated_ids,
                route_tokenizer,
                strict=True,
            )
        except Exception as exc:
            decode_error = f"{type(exc).__name__}: {exc}"
            route_tokens = decode_route_ids_to_source_tokens(
                generated_ids,
                route_tokenizer,
                strict=False,
            )

        format_valid = route_format_valid(
            route_tokens,
            require_forward_only=forward_only,
        )

        route_molecules = []
        final_product = None
        if format_valid:
            try:
                route_molecules = molecule_tokens_from_route(route_tokens)
                final_product = final_product_from_route(route_tokens)
            except Exception:
                route_molecules = []
                final_product = None

        row.update(
            {
                "route_tokens": route_tokens,
                "route_text": format_route_tokens(route_tokens),
                "route_decode_error": decode_error,
                "route_format_valid": format_valid,
                "route_all_molecules_valid": (
                    bool(route_molecules)
                    and all(is_valid_smiles(smi) for smi in route_molecules)
                ),
                "route_final_product": final_product,
                "route_final_product_matches_smiles": _canonical_equal(
                    final_product,
                    row["smiles"],
                ),
                "reactant_reference_audit": reactant_reference_audit,
                "validity_aware_decoding": validity_aware_diag,
            }
        )

    output_file = Path(cfg["generation"]["output_file"])
    ensure_dir(output_file.parent)

    with output_file.open("w", encoding="utf-8") as handle:
        for row in collected:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("Saved:", output_file)


if __name__ == "__main__":
    main()
