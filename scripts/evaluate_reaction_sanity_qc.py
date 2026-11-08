#!/usr/bin/env python3
"""
Stage 2F-2: model-free chemical sanity / reaction-center QC.

This script performs structural diagnostics on each generated reaction step.
It intentionally does NOT label reactions as chemically valid/invalid.  The
checks are designed to surface obvious or unusual structural behavior that can
be audited further.

Main diagnostics
----------------
* molecule parsing / sanitization
* product fragmentation
* heavy-atom / exact-mass / formal-charge changes
* element inventory accounting
* reactant-to-product atom coverage using pairwise MCS mappings
* approximate bond-formation / bond-loss / bond-order-change counts
* approximate reaction-center size
* configurable alert flags and alert severity

Because source reaction records may omit reagents/catalysts and because atom
mapping is unavailable here, inventory and MCS-derived quantities are QC
surrogates, not conservation laws or reaction feasibility tests.

Inputs
------
Normally use the Stage 2F-0 step CSV:
    reference_reaction_support_steps.csv
Optionally merge Stage 2F-1 results:
    template_executability_steps.csv

Outputs
-------
* reaction_sanity_qc_steps.csv
* reaction_sanity_qc_routes.csv
* reaction_sanity_qc_summary.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFMCS, rdMolDescriptors


AUTO_ROUTE_ID_COLUMNS = ("test_index", "example_index", "route_id", "sample_id", "index")


def safe_fraction(num: int, den: int) -> float | None:
    return float(num) / float(den) if den else None


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_compact(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def as_bool_or_none(x: Any) -> bool | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def parse_reactants_raw(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value)
    try:
        obj = json.loads(text)
    except Exception:
        try:
            obj = ast.literal_eval(text)
        except Exception:
            obj = [text]
    if isinstance(obj, (list, tuple)):
        return [str(x) for x in obj]
    return [str(obj)]


def pick_route_id_column(columns: Sequence[str], explicit: str) -> str:
    if explicit != "auto":
        if explicit not in columns:
            raise KeyError(f"Requested route id column {explicit!r} not found")
        return explicit
    for c in AUTO_ROUTE_ID_COLUMNS:
        if c in columns:
            return c
    raise KeyError(f"Could not auto-detect route ID column from {AUTO_ROUTE_ID_COLUMNS}")


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def canonical_smiles(mol: Chem.Mol, isomeric: bool = True) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def element_counter(mol: Chem.Mol) -> Counter[int]:
    c: Counter[int] = Counter()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            c[atom.GetAtomicNum()] += 1
    return c


def element_counter_symbol(counter: Counter[int]) -> dict[str, int]:
    pt = Chem.GetPeriodicTable()
    return {pt.GetElementSymbol(int(z)): int(n) for z, n in sorted(counter.items())}


def total_formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def heavy_atom_count(mol: Chem.Mol) -> int:
    return int(mol.GetNumHeavyAtoms())


def exact_mass(mol: Chem.Mol) -> float:
    return float(Descriptors.ExactMolWt(mol))


def ring_count(mol: Chem.Mol) -> int:
    try:
        return int(rdMolDescriptors.CalcNumRings(mol))
    except Exception:
        return 0


def fragment_count(mol: Chem.Mol) -> int:
    try:
        return len(Chem.GetMolFrags(mol))
    except Exception:
        return 1


def bond_descriptor(bond: Chem.Bond) -> str:
    if bond.GetIsAromatic():
        return "aromatic"
    return str(bond.GetBondType())


def mcs_mapping_one_reactant_to_product(
    reactant: Chem.Mol,
    product: Chem.Mol,
    *,
    timeout_s: int,
    max_product_matches: int,
) -> dict[str, Any]:
    """
    Obtain a conservative pairwise MCS mapping and candidate product matches.

    The mapping is only a structural surrogate.  It is not an atom mapping and
    must not be interpreted as a reaction-center ground truth.
    """
    try:
        mcs = rdFMCS.FindMCS(
            [reactant, product],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            matchValences=False,
            timeout=int(timeout_s),
        )
    except Exception as exc:
        return {"success": False, "reason": f"mcs_error:{type(exc).__name__}"}

    if mcs.canceled:
        return {"success": False, "reason": "mcs_timeout"}
    if mcs.numAtoms <= 0 or not mcs.smartsString:
        return {"success": False, "reason": "empty_mcs"}

    q = Chem.MolFromSmarts(mcs.smartsString)
    if q is None:
        return {"success": False, "reason": "mcs_query_parse_failed"}

    r_matches = reactant.GetSubstructMatches(q, uniquify=True, maxMatches=8)
    p_matches = product.GetSubstructMatches(q, uniquify=True, maxMatches=max_product_matches)
    if not r_matches or not p_matches:
        return {"success": False, "reason": "mcs_substructure_match_failed"}

    # For the reactant, one representative match is enough for retention/bond
    # accounting.  For the product we retain multiple candidates so the outer
    # greedy assignment can maximize non-overlapping product coverage.
    r_match = tuple(int(x) for x in r_matches[0])
    p_matches = [tuple(int(x) for x in m) for m in p_matches]
    return {
        "success": True,
        "reason": "ok",
        "mcs_atoms": int(mcs.numAtoms),
        "mcs_bonds": int(mcs.numBonds),
        "query_smarts": mcs.smartsString,
        "reactant_match": r_match,
        "product_matches": p_matches,
    }


def choose_nonoverlapping_product_matches(
    pair_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Greedily assign each reactant MCS to a product occurrence maximizing newly
    covered product atoms.  Reactants with larger MCS are assigned first.
    """
    order = sorted(
        range(len(pair_results)),
        key=lambda i: int(pair_results[i].get("mcs_atoms") or 0),
        reverse=True,
    )
    used: set[int] = set()
    selected: list[dict[str, Any]] = [dict() for _ in pair_results]

    for idx in order:
        res = pair_results[idx]
        if not res.get("success"):
            selected[idx] = {"success": False, "reason": res.get("reason")}
            continue
        candidates = res.get("product_matches") or []
        if not candidates:
            selected[idx] = {"success": False, "reason": "no_product_matches"}
            continue
        best = max(candidates, key=lambda m: (len(set(m) - used), -len(set(m) & used)))
        selected[idx] = {
            **res,
            "selected_product_match": tuple(best),
            "new_product_atoms": len(set(best) - used),
            "overlap_product_atoms": len(set(best) & used),
        }
        used.update(best)
    return selected


def approximate_bond_changes(
    reactants: Sequence[Chem.Mol],
    product: Chem.Mol,
    selected_maps: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    # product atom idx -> (reactant_index, reactant_atom_idx), keeping first
    p_to_source: dict[int, tuple[int, int]] = {}
    source_to_p: dict[tuple[int, int], int] = {}

    for ri, res in enumerate(selected_maps):
        if not res.get("success") or "selected_product_match" not in res:
            continue
        r_match = tuple(res["reactant_match"])
        p_match = tuple(res["selected_product_match"])
        for r_idx, p_idx in zip(r_match, p_match, strict=True):
            if p_idx in p_to_source:
                continue
            p_to_source[int(p_idx)] = (ri, int(r_idx))
            source_to_p[(ri, int(r_idx))] = int(p_idx)

    formed = 0
    broken = 0
    order_changed = 0
    product_center_atoms: set[int] = set()
    source_center_atoms: set[tuple[int, int]] = set()

    # Product bonds among mapped atoms: check whether they existed inside the
    # same source reactant with the same bond descriptor.
    for pb in product.GetBonds():
        pa = int(pb.GetBeginAtomIdx())
        pbidx = int(pb.GetEndAtomIdx())
        if pa not in p_to_source or pbidx not in p_to_source:
            continue
        src_a = p_to_source[pa]
        src_b = p_to_source[pbidx]
        if src_a[0] != src_b[0]:
            formed += 1
            product_center_atoms.update([pa, pbidx])
            source_center_atoms.update([src_a, src_b])
            continue
        r = reactants[src_a[0]]
        rb = r.GetBondBetweenAtoms(src_a[1], src_b[1])
        if rb is None:
            formed += 1
            product_center_atoms.update([pa, pbidx])
            source_center_atoms.update([src_a, src_b])
        elif bond_descriptor(rb) != bond_descriptor(pb):
            order_changed += 1
            product_center_atoms.update([pa, pbidx])
            source_center_atoms.update([src_a, src_b])

    # Source bonds among mapped atoms: check if retained in product.
    for ri, r in enumerate(reactants):
        for rb in r.GetBonds():
            a = (ri, int(rb.GetBeginAtomIdx()))
            b = (ri, int(rb.GetEndAtomIdx()))
            if a not in source_to_p or b not in source_to_p:
                continue
            pa = source_to_p[a]
            pbidx = source_to_p[b]
            prod_bond = product.GetBondBetweenAtoms(pa, pbidx)
            if prod_bond is None:
                broken += 1
                product_center_atoms.update([pa, pbidx])
                source_center_atoms.update([a, b])
            # Bond-order changes counted above only once.

    return {
        "estimated_bonds_formed": int(formed),
        "estimated_bonds_broken": int(broken),
        "estimated_bond_order_changes": int(order_changed),
        "estimated_total_bond_changes": int(formed + broken + order_changed),
        "estimated_product_reaction_center_atoms": int(len(product_center_atoms)),
        "estimated_source_reaction_center_atoms": int(len(source_center_atoms)),
    }


def evaluate_step_core(payload: dict[str, Any]) -> dict[str, Any]:
    reactants_smiles = payload["reactants"]
    product_smiles = payload["product"]
    timeout_s = payload["mcs_timeout_s"]
    max_product_matches = payload["max_product_matches"]

    out: dict[str, Any] = {}
    reactants = [mol_from_smiles(s) for s in reactants_smiles]
    product = mol_from_smiles(product_smiles)

    out["qc_reactant_count"] = len(reactants_smiles)
    out["qc_all_reactants_valid"] = all(m is not None for m in reactants)
    out["qc_product_valid"] = product is not None
    if product is None or any(m is None for m in reactants):
        out.update(
            {
                "qc_evaluable": False,
                "qc_failure_reason": "invalid_reactant_or_product",
            }
        )
        return out

    r_mols: list[Chem.Mol] = [m for m in reactants if m is not None]
    p_mol: Chem.Mol = product
    out["qc_evaluable"] = True
    out["qc_failure_reason"] = None

    r_heavy = [heavy_atom_count(m) for m in r_mols]
    p_heavy = heavy_atom_count(p_mol)
    total_r_heavy = sum(r_heavy)
    out["qc_reactant_heavy_atoms_total"] = total_r_heavy
    out["qc_product_heavy_atoms"] = p_heavy
    out["qc_heavy_atom_delta_product_minus_reactants"] = p_heavy - total_r_heavy
    out["qc_product_to_reactant_heavy_atom_ratio"] = safe_fraction(p_heavy, total_r_heavy)

    r_mass = sum(exact_mass(m) for m in r_mols)
    p_mass = exact_mass(p_mol)
    out["qc_reactant_exact_mass_total"] = r_mass
    out["qc_product_exact_mass"] = p_mass
    out["qc_exact_mass_delta_product_minus_reactants"] = p_mass - r_mass
    out["qc_product_to_reactant_exact_mass_ratio"] = (p_mass / r_mass) if r_mass else None

    r_charge = sum(total_formal_charge(m) for m in r_mols)
    p_charge = total_formal_charge(p_mol)
    out["qc_reactant_formal_charge_total"] = r_charge
    out["qc_product_formal_charge"] = p_charge
    out["qc_formal_charge_delta"] = p_charge - r_charge

    r_rings = sum(ring_count(m) for m in r_mols)
    p_rings = ring_count(p_mol)
    out["qc_reactant_ring_count_total"] = r_rings
    out["qc_product_ring_count"] = p_rings
    out["qc_ring_count_delta"] = p_rings - r_rings

    out["qc_product_fragment_count"] = fragment_count(p_mol)
    out["qc_product_is_single_fragment"] = out["qc_product_fragment_count"] == 1

    r_elements: Counter[int] = Counter()
    for m in r_mols:
        r_elements.update(element_counter(m))
    p_elements = element_counter(p_mol)
    positive_delta: Counter[int] = Counter()
    negative_delta: Counter[int] = Counter()
    for z in set(r_elements) | set(p_elements):
        d = p_elements[z] - r_elements[z]
        if d > 0:
            positive_delta[z] = d
        elif d < 0:
            negative_delta[z] = -d
    unaccounted = sum(positive_delta.values())
    out["qc_reactant_element_inventory_json"] = json_compact(element_counter_symbol(r_elements))
    out["qc_product_element_inventory_json"] = json_compact(element_counter_symbol(p_elements))
    out["qc_product_positive_element_delta_json"] = json_compact(element_counter_symbol(positive_delta))
    out["qc_product_negative_element_delta_json"] = json_compact(element_counter_symbol(negative_delta))
    out["qc_unaccounted_product_heavy_atoms_by_element"] = int(unaccounted)
    out["qc_unaccounted_product_heavy_atom_fraction"] = safe_fraction(unaccounted, p_heavy)
    out["qc_product_elements_subset_of_reactants"] = unaccounted == 0

    # Pairwise MCS structural accounting.
    pair = [
        mcs_mapping_one_reactant_to_product(
            r, p_mol, timeout_s=timeout_s, max_product_matches=max_product_matches
        )
        for r in r_mols
    ]
    selected = choose_nonoverlapping_product_matches(pair)
    success_maps = [x for x in selected if x.get("success") and "selected_product_match" in x]
    out["qc_mcs_successful_reactants"] = len(success_maps)
    out["qc_mcs_all_reactants_successful"] = len(success_maps) == len(r_mols)
    out["qc_mcs_timeout_or_failure_count"] = len(r_mols) - len(success_maps)

    covered_product_atoms: set[int] = set()
    retained_source_atoms = 0
    overlap_product_atoms = 0
    mcs_atoms_sum = 0
    for x in success_maps:
        covered_product_atoms.update(int(i) for i in x["selected_product_match"])
        retained_source_atoms += len(x["reactant_match"])
        overlap_product_atoms += int(x.get("overlap_product_atoms") or 0)
        mcs_atoms_sum += int(x.get("mcs_atoms") or 0)
    out["qc_product_atoms_covered_by_reactant_mcs"] = len(covered_product_atoms)
    out["qc_product_atom_coverage_fraction"] = safe_fraction(len(covered_product_atoms), p_heavy)
    out["qc_reactant_atoms_retained_by_mcs"] = retained_source_atoms
    out["qc_reactant_atom_retention_fraction"] = safe_fraction(retained_source_atoms, total_r_heavy)
    out["qc_product_mcs_overlap_atoms_between_reactants"] = overlap_product_atoms
    out["qc_estimated_product_atoms_not_covered_by_reactants"] = max(0, p_heavy - len(covered_product_atoms))
    out["qc_estimated_product_uncovered_atom_fraction"] = safe_fraction(
        max(0, p_heavy - len(covered_product_atoms)), p_heavy
    )

    out.update(approximate_bond_changes(r_mols, p_mol, selected))
    return out


def add_alerts(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not row.get("qc_evaluable"):
        row.update(
            {
                "qc_alert_invalid_structure": True,
                "qc_alert_fragmented_product": None,
                "qc_alert_new_product_atoms": None,
                "qc_alert_low_product_atom_coverage": None,
                "qc_alert_large_formal_charge_change": None,
                "qc_alert_excessive_bond_changes": None,
                "qc_alert_large_reaction_center": None,
                "qc_alert_count": 1,
                "qc_alert_severity": "high",
                "qc_conservative_sanity_pass": False,
            }
        )
        return row

    alerts = {
        "qc_alert_invalid_structure": False,
        "qc_alert_fragmented_product": int(row.get("qc_product_fragment_count") or 0) > args.max_product_fragments,
        "qc_alert_new_product_atoms": float(row.get("qc_unaccounted_product_heavy_atom_fraction") or 0.0)
        > args.max_unaccounted_product_atom_fraction,
        "qc_alert_low_product_atom_coverage": (
            row.get("qc_product_atom_coverage_fraction") is not None
            and float(row["qc_product_atom_coverage_fraction"]) < args.min_product_atom_coverage
        ),
        "qc_alert_large_formal_charge_change": abs(int(row.get("qc_formal_charge_delta") or 0))
        > args.max_abs_charge_delta,
        "qc_alert_excessive_bond_changes": int(row.get("estimated_total_bond_changes") or 0)
        > args.max_total_bond_changes,
        "qc_alert_large_reaction_center": int(row.get("estimated_product_reaction_center_atoms") or 0)
        > args.max_reaction_center_atoms,
    }
    row.update(alerts)
    count = sum(v is True for v in alerts.values())
    row["qc_alert_count"] = int(count)
    row["qc_alert_severity"] = "low" if count == 0 else ("moderate" if count <= 2 else "high")
    row["qc_conservative_sanity_pass"] = count == 0
    return row


def evaluate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    core = evaluate_step_core(payload)
    # Alert parameters are passed explicitly so this function stays picklable.
    class A:
        pass
    a = A()
    for k, v in payload["alert_params"].items():
        setattr(a, k, v)
    return add_alerts(core, a)


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    vals = [float(x) for x in series.tolist() if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    vals_sorted = sorted(vals)
    p95_idx = min(len(vals_sorted) - 1, int(math.ceil(0.95 * len(vals_sorted))) - 1)
    return {
        "count": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p95": float(vals_sorted[p95_idx]),
        "max": float(max(vals)),
    }


def bool_metric(df: pd.DataFrame, col: str, mask: pd.Series | None = None) -> dict[str, Any]:
    sub = df if mask is None else df[mask]
    vals = [v for v in (as_bool_or_none(x) for x in sub[col].tolist()) if v is not None]
    num = sum(v is True for v in vals)
    return {"count": len(vals), "numerator": int(num), "fraction": safe_fraction(num, len(vals))}


def aggregate_routes(step_df: pd.DataFrame, route_id_col: str) -> pd.DataFrame:
    rows = []
    for rid, g in step_df.groupby(route_id_col, dropna=False, sort=False):
        eval_mask = g["qc_evaluable"].map(lambda x: as_bool_or_none(x) is True)
        ev = g[eval_mask]
        alerts = [int(x) for x in ev["qc_alert_count"].tolist() if x is not None and not pd.isna(x)]
        rows.append(
            {
                route_id_col: rid,
                "reaction_steps": len(g),
                "qc_evaluable_steps": len(ev),
                "qc_all_steps_evaluable": len(ev) == len(g) and len(g) > 0,
                "qc_total_alert_count": int(sum(alerts)) if alerts else 0,
                "qc_max_step_alert_count": int(max(alerts)) if alerts else 0,
                "qc_all_steps_conservative_sanity_pass": (
                    len(ev) == len(g)
                    and len(g) > 0
                    and all(as_bool_or_none(x) is True for x in ev["qc_conservative_sanity_pass"].tolist())
                ),
                "qc_any_high_alert_step": any(str(x) == "high" for x in ev["qc_alert_severity"].tolist()),
                "qc_mean_product_atom_coverage": (
                    float(ev["qc_product_atom_coverage_fraction"].dropna().mean())
                    if not ev.empty and ev["qc_product_atom_coverage_fraction"].notna().any()
                    else None
                ),
                "qc_max_estimated_bond_changes": (
                    int(ev["estimated_total_bond_changes"].dropna().max())
                    if not ev.empty and ev["estimated_total_bond_changes"].notna().any()
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2F-2 model-free chemical sanity / reaction-center QC")
    p.add_argument("--f0-step-csv", required=True)
    p.add_argument("--f1-step-csv", default=None, help="Optional Stage 2F-1 step CSV to merge by route_id + step_idx")
    p.add_argument("--route-id-column", default="auto")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--mcs-timeout-s", type=int, default=2)
    p.add_argument("--max-product-matches", type=int, default=32)

    # Alert thresholds.  These are QC heuristics, not feasibility cutoffs.
    p.add_argument("--max-product-fragments", type=int, default=1)
    p.add_argument("--max-unaccounted-product-atom-fraction", type=float, default=0.10)
    p.add_argument("--min-product-atom-coverage", type=float, default=0.75)
    p.add_argument("--max-abs-charge-delta", type=int, default=2)
    p.add_argument("--max-total-bond-changes", type=int, default=6)
    p.add_argument("--max-reaction-center-atoms", type=int, default=8)
    p.add_argument("--quiet-rdkit", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.quiet_rdkit:
        RDLogger.DisableLog("rdApp.*")

    f0_path = Path(args.f0_step_csv)
    df = pd.read_csv(f0_path)
    required = {"reactants_raw", "generated_product_raw", "step_idx"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"F0 step CSV missing required columns: {sorted(missing)}")
    route_id_col = pick_route_id_column(df.columns.tolist(), args.route_id_column)

    if args.f1_step_csv:
        f1_path = Path(args.f1_step_csv)
        f1 = pd.read_csv(f1_path)
        keep_cols = [
            c
            for c in f1.columns
            if c.startswith("f1_") or c in {route_id_col, "step_idx"}
        ]
        f1 = f1[keep_cols].drop_duplicates(subset=[route_id_col, "step_idx"])
        df = df.merge(f1, on=[route_id_col, "step_idx"], how="left", suffixes=("", "_f1"))
    else:
        f1_path = None

    alert_params = {
        "max_product_fragments": args.max_product_fragments,
        "max_unaccounted_product_atom_fraction": args.max_unaccounted_product_atom_fraction,
        "min_product_atom_coverage": args.min_product_atom_coverage,
        "max_abs_charge_delta": args.max_abs_charge_delta,
        "max_total_bond_changes": args.max_total_bond_changes,
        "max_reaction_center_atoms": args.max_reaction_center_atoms,
    }

    payloads = []
    for _, row in df.iterrows():
        payloads.append(
            {
                "reactants": parse_reactants_raw(row["reactants_raw"]),
                "product": str(row["generated_product_raw"]),
                "mcs_timeout_s": args.mcs_timeout_s,
                "max_product_matches": args.max_product_matches,
                "alert_params": alert_params,
            }
        )

    print(f"Evaluating {len(payloads)} generated reaction steps with workers={args.workers} ...")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(evaluate_payload, payloads, chunksize=16))
    else:
        results = []
        for i, payload in enumerate(payloads):
            results.append(evaluate_payload(payload))
            if (i + 1) % 250 == 0:
                print(f"  processed {i + 1}/{len(payloads)}")

    qc_df = pd.DataFrame(results)
    step_df = pd.concat([df.reset_index(drop=True), qc_df.reset_index(drop=True)], axis=1)
    route_df = aggregate_routes(step_df, route_id_col)

    evaluable_mask = step_df["qc_evaluable"].map(lambda x: as_bool_or_none(x) is True)
    summary: dict[str, Any] = {
        "evaluation": {
            "stage": "stage2f2_model_free_chemical_sanity_and_reaction_center_qc",
            "policy": "structural_inventory_pairwise_mcs_and_bond_change_surrogate",
            "interpretation": (
                "QC diagnostics only. Alerts and MCS-derived reaction-center estimates are not proof of chemical "
                "infeasibility or experimental feasibility."
            ),
        },
        "counts": {
            "reaction_steps": len(step_df),
            "qc_evaluable_steps": int(evaluable_mask.sum()),
            "routes": len(route_df),
        },
        "step_level": {
            "all_reactants_valid": bool_metric(step_df, "qc_all_reactants_valid"),
            "product_valid": bool_metric(step_df, "qc_product_valid"),
            "product_single_fragment": bool_metric(step_df, "qc_product_is_single_fragment", evaluable_mask),
            "product_elements_subset_of_reactants": bool_metric(
                step_df, "qc_product_elements_subset_of_reactants", evaluable_mask
            ),
            "mcs_all_reactants_successful": bool_metric(step_df, "qc_mcs_all_reactants_successful", evaluable_mask),
            "conservative_sanity_pass": bool_metric(step_df, "qc_conservative_sanity_pass", evaluable_mask),
            "alert_fragmented_product": bool_metric(step_df, "qc_alert_fragmented_product", evaluable_mask),
            "alert_new_product_atoms": bool_metric(step_df, "qc_alert_new_product_atoms", evaluable_mask),
            "alert_low_product_atom_coverage": bool_metric(
                step_df, "qc_alert_low_product_atom_coverage", evaluable_mask
            ),
            "alert_large_formal_charge_change": bool_metric(
                step_df, "qc_alert_large_formal_charge_change", evaluable_mask
            ),
            "alert_excessive_bond_changes": bool_metric(
                step_df, "qc_alert_excessive_bond_changes", evaluable_mask
            ),
            "alert_large_reaction_center": bool_metric(
                step_df, "qc_alert_large_reaction_center", evaluable_mask
            ),
        },
        "numeric_distributions": {
            "product_atom_coverage_fraction": numeric_summary(step_df.loc[evaluable_mask, "qc_product_atom_coverage_fraction"]),
            "reactant_atom_retention_fraction": numeric_summary(step_df.loc[evaluable_mask, "qc_reactant_atom_retention_fraction"]),
            "unaccounted_product_heavy_atom_fraction": numeric_summary(
                step_df.loc[evaluable_mask, "qc_unaccounted_product_heavy_atom_fraction"]
            ),
            "formal_charge_delta": numeric_summary(step_df.loc[evaluable_mask, "qc_formal_charge_delta"]),
            "estimated_total_bond_changes": numeric_summary(
                step_df.loc[evaluable_mask, "estimated_total_bond_changes"]
            ),
            "estimated_product_reaction_center_atoms": numeric_summary(
                step_df.loc[evaluable_mask, "estimated_product_reaction_center_atoms"]
            ),
        },
        "route_level": {
            "all_steps_evaluable": bool_metric(route_df, "qc_all_steps_evaluable"),
            "all_steps_conservative_sanity_pass": bool_metric(
                route_df, "qc_all_steps_conservative_sanity_pass"
            ),
            "any_high_alert_step": bool_metric(route_df, "qc_any_high_alert_step"),
        },
        "stratified": {},
        "input": {
            "f0_step_csv": str(f0_path),
            "f0_step_csv_sha256": sha256_file(f0_path),
            "f1_step_csv": str(f1_path) if f1_path else None,
            "f1_step_csv_sha256": sha256_file(f1_path) if f1_path else None,
            "route_id_column": route_id_col,
        },
        "parameters": {
            "workers": args.workers,
            "mcs_timeout_s": args.mcs_timeout_s,
            "max_product_matches": args.max_product_matches,
            "alerts": alert_params,
        },
    }

    # Useful stratifications from F0 and F1 if available.
    if "reference_exact_reaction_support" in step_df.columns:
        for label, val in [("reference_exact_supported", True), ("reference_exact_unsupported", False)]:
            mask = step_df["reference_exact_reaction_support"].map(lambda x: as_bool_or_none(x) is val) & evaluable_mask
            summary["stratified"][label] = {
                "reaction_steps": int(mask.sum()),
                "conservative_sanity_pass": bool_metric(step_df, "qc_conservative_sanity_pass", mask),
                "product_atom_coverage_fraction": numeric_summary(step_df.loc[mask, "qc_product_atom_coverage_fraction"]),
                "estimated_total_bond_changes": numeric_summary(step_df.loc[mask, "estimated_total_bond_changes"]),
            }
    if "train_novel_exact_reaction" in step_df.columns:
        for label, val in [("train_seen", False), ("train_novel", True)]:
            mask = step_df["train_novel_exact_reaction"].map(lambda x: as_bool_or_none(x) is val) & evaluable_mask
            summary["stratified"][label] = {
                "reaction_steps": int(mask.sum()),
                "conservative_sanity_pass": bool_metric(step_df, "qc_conservative_sanity_pass", mask),
                "product_atom_coverage_fraction": numeric_summary(step_df.loc[mask, "qc_product_atom_coverage_fraction"]),
                "estimated_total_bond_changes": numeric_summary(step_df.loc[mask, "estimated_total_bond_changes"]),
            }
    if "f1_template_execution_success_exact" in step_df.columns:
        for label, val in [("f1_template_executable", True), ("f1_not_template_executable", False)]:
            mask = step_df["f1_template_execution_success_exact"].map(lambda x: as_bool_or_none(x) is val) & evaluable_mask
            summary["stratified"][label] = {
                "reaction_steps": int(mask.sum()),
                "conservative_sanity_pass": bool_metric(step_df, "qc_conservative_sanity_pass", mask),
                "product_atom_coverage_fraction": numeric_summary(step_df.loc[mask, "qc_product_atom_coverage_fraction"]),
                "estimated_total_bond_changes": numeric_summary(step_df.loc[mask, "estimated_total_bond_changes"]),
            }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_path = out_dir / "reaction_sanity_qc_steps.csv"
    route_path = out_dir / "reaction_sanity_qc_routes.csv"
    summary_path = out_dir / "reaction_sanity_qc_summary.json"
    step_df.to_csv(step_path, index=False)
    route_df.to_csv(route_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done")
    print(f"  steps   : {step_path}")
    print(f"  routes  : {route_path}")
    print(f"  summary : {summary_path}")
    print(json.dumps(summary["step_level"], indent=2))


if __name__ == "__main__":
    main()
