from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PH4_TYPES = ("HBD", "HBA", "Cation", "Anion", "Aromatic", "Hydrophobe")


def normalize_pair_id(value) -> int:
    if pd.isna(value):
        raise ValueError("protein_ligand_pair_id is missing")
    return int(float(value))


def clean_icode(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", ".", "?"} else text


def pocket_key(chain, num, icode, name) -> tuple[str, int, str, str]:
    return (str(chain).strip(), int(num), clean_icode(icode), str(name).strip().upper())


def parse_prolif_residue(text: str) -> tuple[str | None, int | None, str, str | None]:
    """Parse common ProLIF residue strings such as ASP123.A or ASP123A.A.

    Returns (resname, residue_num, icode, chain). The function is deliberately
    permissive because ProLIF/MDAnalysis string formatting can vary slightly.
    """
    s = str(text).strip().strip("[](){}<>")
    # Typical ProLIF ResidueId: ASP123.A ; insertion code may appear as ASP123B.A
    m = re.match(r"^([A-Za-z]{3,4})(-?\d+)([A-Za-z]?)(?:\.([^\.\s]+))?$", s)
    if m:
        return m.group(1).upper(), int(m.group(2)), m.group(3) or "", m.group(4)

    # Fallback: find leading residue name + number, optional .CHAIN elsewhere.
    m = re.search(r"([A-Za-z]{3,4})\s*(-?\d+)([A-Za-z]?)", s)
    if not m:
        return None, None, "", None
    chain = None
    if "." in s:
        chain = s.rsplit(".", 1)[-1].strip() or None
    return m.group(1).upper(), int(m.group(2)), m.group(3) or "", chain


def build_lookup(pocket_pair: pd.DataFrame):
    exact = {}
    loose = defaultdict(list)
    for row in pocket_pair.itertuples(index=False):
        key = pocket_key(row.chain_id, row.residue_num, row.icode, row.residue_name)
        exact[key] = key
        loose[(key[1], key[3])].append(key)
    return exact, loose


def match_residue(text: str, exact, loose):
    name, num, icode, chain = parse_prolif_residue(text)
    if name is None or num is None:
        return None, "parse_failed"

    candidates = []
    if chain is not None:
        candidates.append((str(chain), int(num), clean_icode(icode), str(name).upper()))
    # Your preprocessing fixes protein chain to A, but retain generic fallbacks.
    candidates.extend([
        ("A", int(num), clean_icode(icode), str(name).upper()),
        ("", int(num), clean_icode(icode), str(name).upper()),
    ])
    for key in candidates:
        if key in exact:
            return exact[key], "exact"

    xs = loose.get((int(num), str(name).upper()), [])
    if len(xs) == 1:
        return xs[0], "unique_num_name"
    if icode:
        ys = [x for x in xs if x[2] == clean_icode(icode)]
        if len(ys) == 1:
            return ys[0], "unique_num_name_icode"
    return None, "not_in_8A_pocket"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ph4", required=True, help="full_ligand_pharmacophore_v3.csv")
    ap.add_argument("--pocket-context", required=True, help="pocket_context_8A.csv")
    ap.add_argument("--output", required=True, help="ph4_local_targets_v2.csv")
    ap.add_argument(
        "--pair-model-qc", default=None,
        help="Optional pair_model_ph4_completeness_v3.csv; if supplied, only feature_complete=True models are used.",
    )
    args = ap.parse_args()

    full_path = Path(args.full_ph4)
    pocket_path = Path(args.pocket_context)
    out_path = Path(args.output)

    ph4 = pd.read_csv(full_path)
    pocket = pd.read_csv(pocket_path)

    req_ph4 = {
        "protein_ligand_pair_id", "model_index", "ph4_key", "ph4_type",
        "protein_residues_json",
    }
    req_pocket = {
        "protein_ligand_pair_id", "chain_id", "residue_num", "icode", "residue_name",
    }
    miss = sorted(req_ph4 - set(ph4.columns))
    if miss:
        raise KeyError(f"Full PH4 CSV missing columns: {miss}")
    miss = sorted(req_pocket - set(pocket.columns))
    if miss:
        raise KeyError(f"Pocket CSV missing columns: {miss}")

    ph4 = ph4.copy()
    pocket = pocket.copy()
    ph4["_pair_id"] = ph4["protein_ligand_pair_id"].map(normalize_pair_id)
    pocket["_pair_id"] = pocket["protein_ligand_pair_id"].map(normalize_pair_id)
    ph4["model_index"] = ph4["model_index"].astype(int)

    qc_note = "not_applied"
    if args.pair_model_qc:
        qc = pd.read_csv(args.pair_model_qc)
        needed = {"protein_ligand_pair_id", "model_index", "feature_complete"}
        if not needed.issubset(qc.columns):
            raise KeyError(f"QC CSV must contain {sorted(needed)}")
        qc = qc.copy()
        qc["_pair_id"] = qc["protein_ligand_pair_id"].map(normalize_pair_id)
        qc["model_index"] = qc["model_index"].astype(int)
        good = qc[qc["feature_complete"].fillna(False).astype(bool)][["_pair_id", "model_index"]].drop_duplicates()
        before = len(ph4)
        ph4 = ph4.merge(good, on=["_pair_id", "model_index"], how="inner")
        qc_note = f"feature_complete_only:{before}->{len(ph4)} rows"

    rows = []
    unmatched_counts = defaultdict(int)
    unmatched_examples = defaultdict(list)
    pair_reports = []

    pocket_groups = {int(pid): g.copy() for pid, g in pocket.groupby("_pair_id", sort=False)}

    for pair_id, pair_group in ph4.groupby("_pair_id", sort=False):
        pair_id = int(pair_id)
        pk = pocket_groups.get(pair_id)
        if pk is None or pk.empty:
            continue
        exact, loose = build_lookup(pk)

        # Denominator for a type = pair models in which at least one feature of
        # that type was successfully projected. This mirrors the feature-model
        # denominator logic used by the existing consensus PH4 calculation.
        type_models = {}
        for t in PH4_TYPES:
            type_models[t] = set(
                int(x) for x in pair_group.loc[pair_group["ph4_type"] == t, "model_index"].unique()
            )

        # Binary model-level event: for a given residue/type/model, did ANY
        # ligand PH4 feature of this type interact with that residue?
        active_models = defaultdict(set)  # (residue_key, type) -> {model_index}
        seen_res_strings = set()

        for row in pair_group.itertuples(index=False):
            t = str(row.ph4_type)
            if t not in PH4_TYPES:
                continue
            try:
                residues = json.loads(row.protein_residues_json) if pd.notna(row.protein_residues_json) else []
            except Exception:
                residues = []
            if not isinstance(residues, list):
                residues = []
            for residue_text in set(str(x) for x in residues):
                seen_res_strings.add(residue_text)
                key, status = match_residue(residue_text, exact, loose)
                if key is None:
                    unmatched_counts[status] += 1
                    if len(unmatched_examples[status]) < 20:
                        unmatched_examples[status].append({"pair_id": pair_id, "residue": residue_text})
                    continue
                active_models[(key, t)].add(int(row.model_index))

        supervised_types = [t for t in PH4_TYPES if type_models[t]]
        n_positive_cells = 0
        for pkrow in pk.itertuples(index=False):
            key = pocket_key(pkrow.chain_id, pkrow.residue_num, pkrow.icode, pkrow.residue_name)
            for t in PH4_TYPES:
                denom_models = type_models[t]
                mask = int(bool(denom_models))
                if mask:
                    numerator = len(active_models.get((key, t), set()) & denom_models)
                    target = float(numerator / len(denom_models))
                else:
                    numerator = 0
                    target = 0.0
                n_positive_cells += int(mask and target > 0)
                rows.append({
                    "protein_ligand_pair_id": pair_id,
                    "chain_id": key[0],
                    "residue_num": key[1],
                    "icode": key[2],
                    "residue_name": key[3],
                    "ph4_type": t,
                    "target": target,
                    "mask": mask,
                    "n_type_models": int(len(denom_models)),
                    "n_active_models": int(numerator),
                })

        pair_reports.append({
            "protein_ligand_pair_id": pair_id,
            "n_pocket_residues": int(len(pk)),
            "n_pair_models": int(pair_group["model_index"].nunique()),
            "supervised_types": supervised_types,
            "n_supervised_types": int(len(supervised_types)),
            "n_positive_local_cells": int(n_positive_cells),
            "n_interacting_residue_strings": int(len(seen_res_strings)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No PH4_v2 local targets were built")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    report = {
        "source_full_ph4": str(full_path),
        "source_pocket_context": str(pocket_path),
        "pair_model_qc": str(args.pair_model_qc) if args.pair_model_qc else None,
        "pair_model_qc_policy": qc_note,
        "target_semantics": (
            "For each protein-ligand pair, pocket residue and ligand PH4 type: "
            "fraction of usable Boltz-2 pair models in which any feature of that type "
            "interacts with that residue. Types absent from the reference ligand are masked, not zero-labeled."
        ),
        "ph4_types": list(PH4_TYPES),
        "n_rows": int(len(out)),
        "n_pairs": int(out["protein_ligand_pair_id"].nunique()),
        "n_pocket_residues": int(out[["protein_ligand_pair_id", "chain_id", "residue_num", "icode", "residue_name"]].drop_duplicates().shape[0]),
        "n_supervised_cells": int(out["mask"].sum()),
        "n_positive_cells": int(((out["mask"] == 1) & (out["target"] > 0)).sum()),
        "positive_fraction_among_supervised": float(((out["mask"] == 1) & (out["target"] > 0)).sum() / max(int(out["mask"].sum()), 1)),
        "per_type": {},
        "unmatched_interaction_residue_counts": dict(unmatched_counts),
        "unmatched_interaction_residue_examples": dict(unmatched_examples),
    }
    for t in PH4_TYPES:
        x = out[(out["ph4_type"] == t) & (out["mask"] == 1)]
        report["per_type"][t] = {
            "n_supervised": int(len(x)),
            "n_positive": int((x["target"] > 0).sum()),
            "positive_fraction": float((x["target"] > 0).mean()) if len(x) else None,
            "mean_target_positive": float(x.loc[x["target"] > 0, "target"].mean()) if bool((x["target"] > 0).any()) else None,
        }

    Path(str(out_path) + ".report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(pair_reports).to_csv(str(out_path) + ".pair_report.csv", index=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
