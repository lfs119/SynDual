from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from syndual.chemistry import (
    canonicalize_smiles,
    is_valid_smiles,
    uniqueness_fraction,
    validity_fraction,
)
from syndual.data import (
    RouteCollator,
    RouteJSONLDataset,
)
from syndual.model_syn_stage2D import SynthesisAwareModel
from syndual.molformer import generate_unconditional
from syndual.route_serialization import (
    count_route_steps,
    decode_route_ids_to_source_tokens,
    format_route_tokens,
    molecule_tokens_from_route,
    parse_route_actions,
)
from syndual.training import evaluate_model_syn
from syndual.utils import (
    ensure_dir,
    load_yaml,
    resolve_device,
    save_json,
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
        f"Neither checkpoint exists: {primary} / {fallback}"
    )


def _canonical_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    ca = canonicalize_smiles(a, isomeric=True)
    cb = canonicalize_smiles(b, isomeric=True)
    return ca is not None and cb is not None and ca == cb


def _canonical_key(smiles: str | None) -> str | None:
    """Canonical molecular identity used for graph/vocabulary comparisons."""
    if smiles is None:
        return None
    return canonicalize_smiles(smiles, isomeric=True)


def _resolve_data_path(raw_path: str | Path, data_root: Path) -> Path:
    """
    Resolve a configured path.

    Absolute paths are used directly. Relative paths are first interpreted
    relative to the current working directory; if they do not exist there,
    they are interpreted relative to data_root.
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return data_root / path


def _extract_bb_vocab_from_payload(payload) -> list[str]:
    """Extract a building-block SMILES list from common serialized payloads."""
    if isinstance(payload, dict):
        if "building_block_vocab" in payload:
            payload = payload["building_block_vocab"]
        elif "building_blocks" in payload:
            payload = payload["building_blocks"]
        else:
            raise KeyError(
                "Building-block payload is a dict but has neither "
                "'building_block_vocab' nor 'building_blocks'."
            )

    if not isinstance(payload, (list, tuple, set)):
        raise TypeError(
            "Building-block vocabulary must resolve to a list/tuple/set of SMILES; "
            f"got {type(payload).__name__}."
        )

    return [str(x).strip() for x in payload if str(x).strip()]


def _load_building_block_vocab(path: Path) -> tuple[set[str], dict]:
    """
    Load the official/declared SynLaD building-block vocabulary and canonicalize it.

    Supported inputs:
      - .pkl/.pickle: original SynLaD dataset pickle containing
        {'dataset': ..., 'building_block_vocab': [...]}
      - .json: a list, or a dict with building_block_vocab/building_blocks
      - .csv/.tsv: SMILES column if present, otherwise first column
      - .txt/.smi/.smiles: first whitespace-separated field per non-empty line

    Membership is evaluated on canonical isomeric SMILES, so an equivalent valid
    SMILES spelling is counted as the same building block.
    """
    if not path.exists():
        raise FileNotFoundError(f"Building-block vocabulary file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        smiles_list = _extract_bb_vocab_from_payload(payload)

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        smiles_list = _extract_bb_vocab_from_payload(payload)

    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        candidate_columns = [
            "smiles",
            "building_block_smiles",
            "building_block",
            "canonical_smiles",
        ]
        column = next((c for c in candidate_columns if c in df.columns), df.columns[0])
        smiles_list = [str(x).strip() for x in df[column].dropna().tolist()]

    elif suffix in {".txt", ".smi", ".smiles"}:
        smiles_list = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                smiles_list.append(line.split()[0])

    else:
        raise ValueError(
            "Unsupported building-block vocabulary format: "
            f"{path.suffix}. Use pkl/pickle/json/csv/tsv/txt/smi/smiles."
        )

    canonical_vocab: set[str] = set()
    invalid_entries: list[str] = []

    for smi in smiles_list:
        key = _canonical_key(smi)
        if key is None:
            invalid_entries.append(smi)
        else:
            canonical_vocab.add(key)

    if invalid_entries:
        preview = invalid_entries[:10]
        raise ValueError(
            f"Building-block vocabulary contains {len(invalid_entries)} invalid SMILES. "
            f"Examples: {preview}"
        )

    if not canonical_vocab:
        raise ValueError(f"Building-block vocabulary is empty after loading: {path}")

    metadata = {
        "path": str(path),
        "raw_entries": len(smiles_list),
        "canonical_unique_entries": len(canonical_vocab),
        "invalid_entries": 0,
    }
    return canonical_vocab, metadata


def _select_generation_indices(
    dataset_size: int,
    evaluation_cfg: dict,
    seed: int,
    cli_mode: str | None = None,
    cli_n: int | None = None,
) -> tuple[list[int], dict]:
    """
    Select free-generation examples without first-N bias.

    generation_mode:
      - "random": reproducible random subset of generation_examples
      - "full": evaluate every test record

    generation_examples <= 0 under random mode is interpreted as full test set.
    """
    mode = str(
        cli_mode
        if cli_mode is not None
        else evaluation_cfg.get("generation_mode", "random")
    ).lower()

    if mode not in {"random", "full"}:
        raise ValueError(
            f"evaluation.generation_mode must be 'random' or 'full'; got {mode!r}."
        )

    requested_n = (
        int(cli_n)
        if cli_n is not None
        else int(evaluation_cfg.get("generation_examples", 100))
    )

    if mode == "full" or requested_n <= 0 or requested_n >= dataset_size:
        indices = list(range(dataset_size))
        effective_mode = "full"
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(dataset_size), k=requested_n))
        effective_mode = "random"

    metadata = {
        "mode": effective_mode,
        "requested_examples": requested_n,
        "evaluated_examples": len(indices),
        "test_dataset_size": dataset_size,
        "selection_seed": seed if effective_mode == "random" else None,
    }
    return indices, metadata


def _route_role_metrics(
    tokens,
    condition_smiles: str,
    building_block_vocab: set[str],
    require_forward_only: bool = True,
):
    """
    Evaluate a decoded source-level SynLaD route.

    reaction_step_strict_causal_graph_consistency is deliberately a route-graph
    consistency metric, NOT a chemical reaction-feasibility metric.

    A forward step is causally consistent only when:
      1. every reactant is a valid molecule;
      2. every reactant has already been introduced by a previous <BB> action
         or produced by a previous causally valid reaction;
      3. the current product is a valid molecule.

    Crucially, a product is added to the available synthesis graph ONLY if that
    reaction step is itself consistent. Therefore an invalid upstream step causes
    downstream references to that unavailable product to fail as well.
    """
    result = {
        # Grammar after source-token recovery.
        "recovered_format_valid": False,

        # All molecular strings.
        "all_generated_molecules_valid": False,
        "invalid_generated_molecules": [],

        # Building blocks: SMILES validity + official-vocabulary membership.
        "building_block_count": 0,
        "valid_building_block_count": 0,
        "building_block_vocab_member_count": 0,
        "all_building_blocks_valid": False,
        "all_building_blocks_in_vocab": False,
        "invalid_building_blocks": [],
        "out_of_vocab_building_blocks": [],

        # Intermediates.
        "intermediate_count": 0,
        "valid_intermediate_count": 0,
        "all_intermediates_valid": None,

        # Final product.
        "final_product_valid": False,
        "generated_final_product": None,

        # Product-condition agreement.
        "final_product_matches_condition": False,
        "conditional_final_product_match": None,

        # Strict causal reaction-graph consistency.
        "reaction_steps_total": 0,
        "reaction_steps_strict_causal_consistent": 0,
        "all_reaction_steps_strict_causal_consistent": False,
        "first_inconsistent_reaction_step": None,
    }

    try:
        actions = parse_route_actions(
            tokens,
            require_forward_only=require_forward_only,
        )
    except Exception:
        return result

    result["recovered_format_valid"] = True

    building_blocks = [
        action["molecule"]
        for action in actions
        if action["type"] == "BB"
    ]

    reaction_actions = [
        action
        for action in actions
        if action["type"] == "FWD_RXN"
    ]

    reaction_products = [
        action["product"]
        for action in reaction_actions
    ]

    # For the current forward, single-terminal-product route representation,
    # every reaction product before the terminal product is treated as an
    # intermediate. This preserves the previous SynDual metric definition.
    intermediates = reaction_products[:-1] if len(reaction_products) > 1 else []
    final_product = reaction_products[-1] if reaction_products else None
    result["generated_final_product"] = final_product

    # ------------------------------------------------------------------
    # 2. All-molecule validity
    # ------------------------------------------------------------------
    try:
        all_molecules = molecule_tokens_from_route(tokens)
    except Exception:
        all_molecules = []

    molecule_validity = [is_valid_smiles(smi) for smi in all_molecules]
    result["all_generated_molecules_valid"] = (
        bool(all_molecules) and all(molecule_validity)
    )
    result["invalid_generated_molecules"] = [
        smi
        for smi, valid in zip(all_molecules, molecule_validity)
        if not valid
    ]

    # ------------------------------------------------------------------
    # 3. Building-block validity AND vocabulary membership
    # ------------------------------------------------------------------
    bb_keys = [_canonical_key(smi) for smi in building_blocks]
    bb_validity = [key is not None for key in bb_keys]
    bb_membership = [
        key is not None and key in building_block_vocab
        for key in bb_keys
    ]

    result["building_block_count"] = len(building_blocks)
    result["valid_building_block_count"] = sum(bb_validity)
    result["building_block_vocab_member_count"] = sum(bb_membership)
    result["all_building_blocks_valid"] = (
        len(building_blocks) > 0 and all(bb_validity)
    )
    result["all_building_blocks_in_vocab"] = (
        len(building_blocks) > 0 and all(bb_membership)
    )
    result["invalid_building_blocks"] = [
        smi
        for smi, valid in zip(building_blocks, bb_validity)
        if not valid
    ]
    result["out_of_vocab_building_blocks"] = [
        smi
        for smi, valid, in_vocab in zip(
            building_blocks,
            bb_validity,
            bb_membership,
        )
        if valid and not in_vocab
    ]

    # ------------------------------------------------------------------
    # 4. Intermediate validity
    # ------------------------------------------------------------------
    intermediate_validity = [is_valid_smiles(smi) for smi in intermediates]
    result["intermediate_count"] = len(intermediates)
    result["valid_intermediate_count"] = sum(intermediate_validity)
    result["all_intermediates_valid"] = (
        all(intermediate_validity) if intermediates else None
    )

    # ------------------------------------------------------------------
    # 5. Final-product validity
    # ------------------------------------------------------------------
    final_valid = final_product is not None and is_valid_smiles(final_product)
    result["final_product_valid"] = final_valid

    # ------------------------------------------------------------------
    # 6. Conditional final-product match
    # ------------------------------------------------------------------
    if final_valid:
        matched = _canonical_equal(final_product, condition_smiles)
        result["final_product_matches_condition"] = matched
        result["conditional_final_product_match"] = matched

    # ------------------------------------------------------------------
    # 7. STRICT CAUSAL reaction-graph consistency
    # ------------------------------------------------------------------
    available_molecules: set[str] = set()
    total_steps = 0
    consistent_steps = 0

    for action in actions:
        if action["type"] == "BB":
            key = _canonical_key(action["molecule"])
            if key is not None:
                # Graph availability follows the generated route declaration.
                # BB-vocabulary membership is evaluated separately above.
                available_molecules.add(key)
            continue

        if action["type"] != "FWD_RXN":
            continue

        total_steps += 1
        reactant_keys = [_canonical_key(smi) for smi in action["reactants"]]
        product_key = _canonical_key(action["product"])

        reactants_valid = (
            len(reactant_keys) > 0
            and all(key is not None for key in reactant_keys)
        )
        reactants_available = (
            reactants_valid
            and all(key in available_molecules for key in reactant_keys)
        )
        product_valid = product_key is not None

        step_consistent = reactants_available and product_valid

        if step_consistent:
            consistent_steps += 1
            # STRICT CAUSAL propagation: only products of valid upstream steps
            # become available to later reactions.
            available_molecules.add(product_key)
        elif result["first_inconsistent_reaction_step"] is None:
            result["first_inconsistent_reaction_step"] = total_steps

    result["reaction_steps_total"] = total_steps
    result["reaction_steps_strict_causal_consistent"] = consistent_steps
    result["all_reaction_steps_strict_causal_consistent"] = (
        total_steps > 0 and consistent_steps == total_steps
    )

    return result


def _strict_gate_route_metrics(route_metrics: dict, strict_format_valid: bool) -> dict:
    """
    Gate primary evaluation metrics on strict decoding + valid route grammar.

    When strict decoding fails, the strict=False recovered tokens are retained for
    diagnostics, but they do NOT receive credit in the main route-generation metrics.
    """
    if strict_format_valid:
        return dict(route_metrics)

    gated = dict(route_metrics)
    gated.update(
        {
            "all_generated_molecules_valid": False,
            "building_block_count": 0,
            "valid_building_block_count": 0,
            "building_block_vocab_member_count": 0,
            "all_building_blocks_valid": False,
            "all_building_blocks_in_vocab": False,
            "intermediate_count": 0,
            "valid_intermediate_count": 0,
            "all_intermediates_valid": None,
            "final_product_valid": False,
            "final_product_matches_condition": False,
            "conditional_final_product_match": None,
            "reaction_steps_total": 0,
            "reaction_steps_strict_causal_consistent": 0,
            "all_reaction_steps_strict_causal_consistent": False,
        }
    )
    return gated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eval_model_syn.yaml",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["random", "full"],
        default=None,
        help="Override evaluation.generation_mode.",
    )
    parser.add_argument(
        "--generation-examples",
        type=int,
        default=None,
        help="Override evaluation.generation_examples for random evaluation.",
    )
    parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="output directory for evaluation results. Overrides evaluation.output_dir in config.",
        )
    
    parser.add_argument(
                "--checkpoint",
                type=str,
                default=None,
                help="Override model.checkpoint in config.",
            )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device("auto")

    checkpoint = choose_checkpoint(cfg)

    model, product_tokenizer, route_tokenizer, model_cfg = (
        SynthesisAwareModel.from_pretrained(
            checkpoint,
            device=device,
        )
    )
    model.eval()

    data_root = Path(cfg["data"]["root"])
    require_forward_only = bool(
        cfg["data"].get("require_forward_only", True)
    )

    # ------------------------------------------------------------------
    # Official SynLaD building-block vocabulary.
    # Fail hard rather than silently approximating it from test references.
    # ------------------------------------------------------------------
    bb_vocab_raw_path = cfg["evaluation"].get(
        "building_block_vocab_path",
        cfg["data"].get("building_block_vocab_path"),
    )
    if not bb_vocab_raw_path:
        raise ValueError(
            "BB vocabulary membership evaluation requires "
            "evaluation.building_block_vocab_path (or data.building_block_vocab_path). "
            "Point it to an original SynLaD pathway_data_*.pkl containing "
            "'building_block_vocab', or to an exported JSON/CSV/TXT vocabulary."
        )

    bb_vocab_path = _resolve_data_path(bb_vocab_raw_path, data_root)
    building_block_vocab, bb_vocab_metadata = _load_building_block_vocab(
        bb_vocab_path
    )

    test_ds = RouteJSONLDataset(
        data_root / cfg["data"]["test_file"],
        smiles_columns=cfg["data"]["smiles_columns"],
        route_token_fields=cfg["data"].get("route_token_fields"),
        route_model_mask_fields=cfg["data"].get("route_model_mask_fields"),
        strict=True,
        require_forward_only=require_forward_only,
    )

    loss_mode = str(
        cfg["evaluation"].get(
            "route_loss_mode",
            model_cfg["route"].get("loss_mode", "all_tokens"),
        )
    )

    collator = RouteCollator(
        product_tokenizer,
        route_tokenizer,
        max_product_length=int(model.backbone.config.max_position_embeddings),
        max_route_length=int(model_cfg["route"]["max_length"]),
        loss_mode=loss_mode,
    )

    loader = DataLoader(
        test_ds,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        shuffle=False,
        collate_fn=collator,
    )

    with torch.inference_mode():
        teacher = evaluate_model_syn(
            model=model,
            loader=loader,
            device=device,
            precision="fp32",
            route_loss_weight=1.0,
            smiles_loss_weight=0.0,
        )

    teacher["route_perplexity"] = float(
        math.exp(min(float(teacher["route_loss"]), 20.0))
    )
    teacher["route_loss_mode"] = loss_mode

    generation_indices, generation_selection = _select_generation_indices(
        dataset_size=len(test_ds),
        evaluation_cfg=cfg["evaluation"],
        seed=seed,
        cli_mode=args.generation_mode,
        cli_n=args.generation_examples,
    )

    generated_rows = []

    for i in generation_indices:
        item = test_ds[i]

        enc = product_tokenizer(
            [item["smiles"]],
            add_special_tokens=True,
            truncation=True,
            max_length=int(model.backbone.config.max_position_embeddings),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.inference_mode():
            ids = model.generate_route(
                product_input_ids=input_ids,
                product_attention_mask=attention_mask,
                route_tokenizer=route_tokenizer,
                max_length=int(model_cfg["route"]["max_length"]),
                do_sample=bool(cfg["evaluation"]["route_do_sample"]),
                temperature=float(cfg["evaluation"]["route_temperature"]),
                top_k=cfg["evaluation"].get("route_top_k"),
                forward_only=require_forward_only,
            )

        generated_ids = ids[0].detach().cpu().tolist()

        # --------------------------------------------------------------
        # STRICT decode validity.
        # strict=False recovery is diagnostic only and cannot earn credit
        # in the primary metrics.
        # --------------------------------------------------------------
        decode_error = None
        recovery_decode_error = None
        strict_decode_valid = True

        try:
            pred_tokens = decode_route_ids_to_source_tokens(
                generated_ids,
                route_tokenizer,
                strict=True,
            )
        except Exception as exc:
            strict_decode_valid = False
            decode_error = f"{type(exc).__name__}: {exc}"
            try:
                pred_tokens = decode_route_ids_to_source_tokens(
                    generated_ids,
                    route_tokenizer,
                    strict=False,
                )
            except Exception as recovery_exc:
                recovery_decode_error = (
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                )
                pred_tokens = []

        target_tokens = list(item["route_tokens"])

        recovered_metrics = _route_role_metrics(
            pred_tokens,
            condition_smiles=item["smiles"],
            building_block_vocab=building_block_vocab,
            require_forward_only=require_forward_only,
        )

        strict_format_valid = (
            strict_decode_valid
            and recovered_metrics["recovered_format_valid"]
        )
        route_metrics = _strict_gate_route_metrics(
            recovered_metrics,
            strict_format_valid=strict_format_valid,
        )

        target_steps = count_route_steps(target_tokens)
        generated_steps = (
            count_route_steps(pred_tokens)
            if strict_format_valid
            else 0
        )

        generated_rows.append(
            {
                "test_index": i,
                "smiles": item["smiles"],
                "target_route_tokens": format_route_tokens(target_tokens),
                "generated_route_tokens": format_route_tokens(pred_tokens),

                # Strict decoding / grammar diagnostics.
                "strict_decode_valid": strict_decode_valid,
                "decode_error": decode_error,
                "recovery_decode_error": recovery_decode_error,
                "recovered_format_valid": recovered_metrics[
                    "recovered_format_valid"
                ],

                # 1. Primary route-format validity = strict decode + grammar.
                "format_valid": strict_format_valid,

                # 2. All-molecule validity.
                "all_generated_molecules_valid": route_metrics[
                    "all_generated_molecules_valid"
                ],

                # 3. Building-block SMILES validity + vocabulary membership.
                "building_block_count": route_metrics["building_block_count"],
                "valid_building_block_count": route_metrics[
                    "valid_building_block_count"
                ],
                "building_block_vocab_member_count": route_metrics[
                    "building_block_vocab_member_count"
                ],
                "all_building_blocks_valid": route_metrics[
                    "all_building_blocks_valid"
                ],
                "all_building_blocks_in_vocab": route_metrics[
                    "all_building_blocks_in_vocab"
                ],

                # 4. Intermediate validity.
                "intermediate_count": route_metrics["intermediate_count"],
                "valid_intermediate_count": route_metrics[
                    "valid_intermediate_count"
                ],
                "all_intermediates_valid": route_metrics[
                    "all_intermediates_valid"
                ],

                # 5. Final-product validity.
                "final_product_valid": route_metrics["final_product_valid"],
                "generated_final_product": (
                    recovered_metrics["generated_final_product"]
                    if strict_format_valid
                    else None
                ),

                # 6. Conditional final-product match.
                "final_product_matches_condition": route_metrics[
                    "final_product_matches_condition"
                ],
                "conditional_final_product_match": route_metrics[
                    "conditional_final_product_match"
                ],

                # 7. Strict causal reaction-graph consistency.
                "reaction_steps_total": route_metrics[
                    "reaction_steps_total"
                ],
                "reaction_steps_strict_causal_consistent": route_metrics[
                    "reaction_steps_strict_causal_consistent"
                ],
                "all_reaction_steps_strict_causal_consistent": route_metrics[
                    "all_reaction_steps_strict_causal_consistent"
                ],
                "first_inconsistent_reaction_step": (
                    recovered_metrics["first_inconsistent_reaction_step"]
                    if strict_format_valid
                    else None
                ),

                # 8. Exact route reconstruction. Strict decode is required.
                "exact_match": (
                    strict_format_valid and pred_tokens == target_tokens
                ),

                # Supplemental diagnostics.
                "target_steps": target_steps,
                "generated_steps": generated_steps,
                "step_count_matches_target": (
                    strict_format_valid and generated_steps == target_steps
                ),
                "invalid_generated_molecules": json.dumps(
                    recovered_metrics["invalid_generated_molecules"],
                    ensure_ascii=False,
                ),
                "invalid_building_blocks": json.dumps(
                    recovered_metrics["invalid_building_blocks"],
                    ensure_ascii=False,
                ),
                "out_of_vocab_building_blocks": json.dumps(
                    recovered_metrics["out_of_vocab_building_blocks"],
                    ensure_ascii=False,
                ),
            }
        )

    with torch.inference_mode():
        smiles_samples = generate_unconditional(
            model.backbone,
            product_tokenizer,
            num_samples=int(cfg["evaluation"]["unconditional_smiles_samples"]),
            max_length=int(model.backbone.config.max_position_embeddings),
            temperature=1.0,
        )
    
    if args.output_dir is not None:
        cfg["evaluation"]["output_dir"] = args.output_dir
        
    output_dir = ensure_dir(cfg["evaluation"]["output_dir"])

    pd.DataFrame(generated_rows).to_csv(
        output_dir / "route_generation_examples.csv",
        index=False,
    )
    pd.DataFrame({"smiles": smiles_samples}).to_csv(
        output_dir / "unconditional_smiles.csv",
        index=False,
    )

    def fraction(key):
        return (
            sum(bool(x[key]) for x in generated_rows)
            / max(len(generated_rows), 1)
        )

    def conditional_fraction(numerator_key, denominator_key):
        eligible = [
            x
            for x in generated_rows
            if bool(x[denominator_key])
        ]
        if not eligible:
            return None
        return (
            sum(bool(x[numerator_key]) for x in eligible)
            / len(eligible)
        )

    def optional_fraction(key):
        eligible = [
            x[key]
            for x in generated_rows
            if x[key] is not None
        ]
        if not eligible:
            return None
        return sum(bool(v) for v in eligible) / len(eligible)

    def molecule_level_fraction(valid_count_key, total_count_key):
        valid = sum(int(x[valid_count_key]) for x in generated_rows)
        total = sum(int(x[total_count_key]) for x in generated_rows)
        if total == 0:
            return None
        return valid / total

    def reaction_step_consistency_fraction():
        consistent = sum(
            int(x["reaction_steps_strict_causal_consistent"])
            for x in generated_rows
        )
        total = sum(int(x["reaction_steps_total"]) for x in generated_rows)
        if total == 0:
            return None
        return consistent / total

    summary = {
        "checkpoint": str(checkpoint),
        "route_schema_version": model_cfg["route"].get("schema_version"),
        "teacher_forced": teacher,
        "generation_selection": generation_selection,
        "building_block_vocabulary": bb_vocab_metadata,
        "route_generation": {
            "n": len(generated_rows),

            # Strict decode is separately visible, and the primary route-format
            # metric below requires BOTH strict decode and valid route grammar.
            "strict_decode_validity": fraction("strict_decode_valid"),

            # 1. Strict route-format validity.
            "route_format_validity": fraction("format_valid"),

            # Diagnostic only: grammar after strict=False recovery.
            "recovered_route_format_validity": fraction(
                "recovered_format_valid"
            ),

            # 2. All-molecule validity.
            "all_molecule_validity": fraction(
                "all_generated_molecules_valid"
            ),

            # 3. Building-block validity and official vocabulary membership.
            "building_block_smiles_route_validity": fraction(
                "all_building_blocks_valid"
            ),
            "building_block_smiles_validity": molecule_level_fraction(
                "valid_building_block_count",
                "building_block_count",
            ),
            "building_block_vocab_route_membership": fraction(
                "all_building_blocks_in_vocab"
            ),
            "building_block_vocab_membership": molecule_level_fraction(
                "building_block_vocab_member_count",
                "building_block_count",
            ),

            # 4. Intermediate validity. One-step routes are excluded from
            # intermediate_route_validity denominator.
            "intermediate_route_validity": optional_fraction(
                "all_intermediates_valid"
            ),
            "intermediate_molecule_validity": molecule_level_fraction(
                "valid_intermediate_count",
                "intermediate_count",
            ),

            # 5. Final-product validity.
            "final_product_validity": fraction("final_product_valid"),

            # 6. Conditional final-product match. Denominator = strictly
            # decoded/formatted routes with a valid generated final product.
            "conditional_final_product_match": conditional_fraction(
                "final_product_matches_condition",
                "final_product_valid",
            ),
            "raw_final_product_match": fraction(
                "final_product_matches_condition"
            ),

            # 7. Strict causal synthesis-graph consistency.
            "reaction_step_strict_causal_graph_consistency": (
                reaction_step_consistency_fraction()
            ),
            "all_steps_strict_causal_consistent_route_fraction": fraction(
                "all_reaction_steps_strict_causal_consistent"
            ),

            # 8. Exact serialized-route reconstruction.
            "route_exact_match": fraction("exact_match"),

            # Supplemental diagnostics.
            "step_count_match_fraction": fraction(
                "step_count_matches_target"
            ),
            "decode_error_fraction": (
                sum(x["decode_error"] is not None for x in generated_rows)
                / max(len(generated_rows), 1)
            ),
            "recovery_decode_error_fraction": (
                sum(
                    x["recovery_decode_error"] is not None
                    for x in generated_rows
                )
                / max(len(generated_rows), 1)
            ),
        },
        "evaluation_stage": {
            "stage": cfg["evaluation"].get(
                "stage",
                "stage2_route_only",
            ),

            "product_backbone": (
                "Joint-finetuned GP-MoLFormer"
                if cfg["evaluation"].get(
                    "joint_finetuning_performed",
                    False,
                )
                else "GP-MoLFormer Stage-1 checkpoint"
            ),

            "product_backbone_frozen": bool(
                cfg["evaluation"].get(
                    "product_backbone_frozen",
                    True,
                )
            ),

            "route_decoder_trained": True,

            "joint_finetuning_performed": bool(
                cfg["evaluation"].get(
                    "joint_finetuning_performed",
                    False,
                )
            ),

            # These are EVALUATION weights,
            # not necessarily training weights.
            "evaluation_smiles_loss_weight": 0.0,
            "evaluation_route_loss_weight": 1.0,
        },
        "unconditional_smiles": {
            "n": len(smiles_samples),
            "validity": validity_fraction(smiles_samples),
            "uniqueness": uniqueness_fraction(smiles_samples),
        },
    }

    save_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
