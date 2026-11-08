#!/usr/bin/env python3
"""
Stage 2F-1: known-template support and executability analysis.

Purpose
-------
This is an evaluation-only, model-free *reaction-template* analysis layered on
Stage 2F-0.  It does NOT claim experimental feasibility.

For each generated reaction step that does not have an exact reference-reaction
match in Stage 2F-0, the script asks:

1) Can the generated reaction be atom-mapped and reduced to a reaction template?
2) Does that template occur among templates extracted from the reference corpus?
3) If a matching *reference-derived* forward template is applied to the
   generated reactants, is the generated product among the enumerated outcomes?

The third question is the strongest structural test in this script and is
reported as "known-template executable to generated product".  It remains a
rule-based structural consistency check, not experimental validation.

Important dependency note
-------------------------
The source-level SynLaD route files used by SynDual are not atom-mapped.  A
chemically meaningful reaction template therefore requires atom mapping first.
This script supports RXNMapper as an optional external dependency and caches all
mapping/template work.  If you already have atom-mapped reaction SMILES, they
can be supplied through cache files to avoid rerunning mapping.

Expected inputs
---------------
* Stage 2F-0 step CSV, normally:
    reference_reaction_support_steps.csv
* Reference route-token files, normally Set2 train + val + test JSONL files.

Outputs
-------
* template_executability_steps.csv
* template_executability_routes.csv
* template_executability_summary.json
* atom_mapping_cache.jsonl
* reference_template_library.jsonl

Recommended interpretation
--------------------------
* exact reference support       : strongest database precedent from Stage 2F-0
* known template support        : generated bond-change template occurs in the
                                  reference corpus
* template executable to product: a matched reference-derived template can be
                                  applied to the generated reactants and yields
                                  the generated product

None of these proves that the reaction will succeed experimentally under a
specific set of conditions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions


# -----------------------------------------------------------------------------
# SynLaD route grammar
# -----------------------------------------------------------------------------
BB = "<BB>"
FWD_RXN = "<FWD_RXN>"
FWD_RXN_RUN = "<FWD_RXN_RUN>"
REV_RXN = "<REV_RXN>"
REV_RXN_RUN = "<REV_RXN_RUN>"
EOS = "<EOS>"
PREFIX_TOKENS = {"<RETRO>", "<OPTIMIZE>", "<PROJECT>"}
SPECIAL_TOKENS = {
    BB,
    FWD_RXN,
    FWD_RXN_RUN,
    REV_RXN,
    REV_RXN_RUN,
    EOS,
    *PREFIX_TOKENS,
}

AUTO_ROUTE_COLUMNS = (
    "serialized_route_tokens_json",
    "serialized_route_tokens",
    "route_tokens",
    "source_route_tokens",
    "synlad_route_tokens",
    "official_route_tokens",
)
AUTO_ROUTE_ID_COLUMNS = ("test_index", "example_index", "route_id", "sample_id", "index")


class RouteParseError(ValueError):
    pass


@dataclass(frozen=True)
class ReactionStep:
    step_idx: int
    reactants: tuple[str, ...]
    product: str


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
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


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _as_bool_or_none(x: Any) -> bool | None:
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


# -----------------------------------------------------------------------------
# Chemistry canonicalization
# -----------------------------------------------------------------------------
def canonicalize_component(smiles: str, *, isomeric: bool = True) -> str | None:
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def canonicalize_molecule_set(
    smiles_or_components: str | Sequence[str], *, isomeric: bool = True
) -> tuple[str, ...] | None:
    items = [smiles_or_components] if isinstance(smiles_or_components, str) else smiles_or_components
    out: set[str] = set()
    saw = False
    for item in items:
        if item is None:
            return None
        for comp in str(item).split("."):
            comp = comp.strip()
            if not comp:
                continue
            saw = True
            c = canonicalize_component(comp, isomeric=isomeric)
            if c is None:
                return None
            out.add(c)
    if not saw or not out:
        return None
    return tuple(sorted(out))


def canonical_reaction_smiles(reactants: Sequence[str], product: str) -> str | None:
    r = canonicalize_molecule_set(reactants, isomeric=True)
    p = canonicalize_molecule_set(product, isomeric=True)
    if r is None or p is None:
        return None
    return f"{'.'.join(r)}>>{'.'.join(p)}"


def canonical_product_key(smiles_or_components: str | Sequence[str], *, isomeric: bool) -> str | None:
    x = canonicalize_molecule_set(smiles_or_components, isomeric=isomeric)
    return None if x is None else ".".join(x)


# -----------------------------------------------------------------------------
# Route parsing
# -----------------------------------------------------------------------------
def decode_route_token_cell(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise RouteParseError("Empty route-token cell")
    text = str(value).strip()
    if not text:
        raise RouteParseError("Empty route-token string")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(text)
        except Exception:
            # Last-resort whitespace tokenization is intentionally conservative.
            obj = text.split()
    if isinstance(obj, dict):
        for key in ("tokens", "route_tokens", "serialized_route_tokens"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, (list, tuple)):
        raise RouteParseError(f"Route token cell did not decode to a sequence: {type(obj)}")
    return [str(x) for x in obj]


def parse_forward_route_tokens(tokens: Sequence[str]) -> list[ReactionStep]:
    steps: list[ReactionStep] = []
    i = 0
    step_idx = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == REV_RXN:
            raise RouteParseError("Reverse reaction token encountered in forward-only route")
        if tok != FWD_RXN:
            i += 1
            continue

        # Reactants until first RUN delimiter.
        j = i + 1
        reactants: list[str] = []
        while j < n and tokens[j] != FWD_RXN_RUN:
            t = tokens[j]
            if t in SPECIAL_TOKENS:
                raise RouteParseError(f"Unexpected special token {t!r} inside forward reactant list")
            reactants.append(t)
            j += 1
        if j >= n:
            raise RouteParseError("Forward reaction missing first <FWD_RXN_RUN>")
        if not reactants:
            raise RouteParseError("Forward reaction has zero reactants")

        # Products until second RUN delimiter.  SynDual currently writes one
        # molecular product token/span at source level; accept multiple and join.
        k = j + 1
        products: list[str] = []
        while k < n and tokens[k] != FWD_RXN_RUN:
            t = tokens[k]
            if t in SPECIAL_TOKENS:
                raise RouteParseError(f"Unexpected special token {t!r} inside forward product list")
            products.append(t)
            k += 1
        if k >= n:
            raise RouteParseError("Forward reaction missing second <FWD_RXN_RUN>")
        if not products:
            raise RouteParseError("Forward reaction has zero products")

        product = ".".join(products)
        steps.append(ReactionStep(step_idx=step_idx, reactants=tuple(reactants), product=product))
        step_idx += 1
        i = k + 1
    return steps


def pick_column(columns: Sequence[str], explicit: str, candidates: Sequence[str]) -> str:
    if explicit != "auto":
        if explicit not in columns:
            raise KeyError(f"Requested column {explicit!r} not found. Available: {list(columns)}")
        return explicit
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"Could not auto-detect column. Tried: {list(candidates)}")


def iter_route_records(path: Path, route_column: str) -> Iterator[tuple[Any, list[str]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            first_obj = None
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if first_obj is None:
                    first_obj = obj
                    used_col = pick_column(list(obj.keys()), route_column, AUTO_ROUTE_COLUMNS)
                yield line_idx, decode_route_token_cell(obj[used_col])
        return
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        used_col = pick_column(df.columns.tolist(), route_column, AUTO_ROUTE_COLUMNS)
        for idx, value in zip(df.index, df[used_col], strict=False):
            yield idx, decode_route_token_cell(value)
        return
    raise ValueError(f"Unsupported route file type: {path}")


def collect_unique_reference_reactions(
    route_files: Sequence[Path], route_column: str
) -> tuple[list[str], dict[str, Any]]:
    unique: set[str] = set()
    provenance = []
    parse_failures = 0
    total_steps = 0
    total_records = 0

    for path in route_files:
        source_records = 0
        source_steps = 0
        source_fail = 0
        for _, tokens in iter_route_records(path, route_column):
            source_records += 1
            total_records += 1
            try:
                steps = parse_forward_route_tokens(tokens)
            except Exception:
                source_fail += 1
                parse_failures += 1
                continue
            source_steps += len(steps)
            total_steps += len(steps)
            for st in steps:
                rxn = canonical_reaction_smiles(st.reactants, st.product)
                if rxn is not None:
                    unique.add(rxn)
        provenance.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "records": source_records,
                "parsed_steps": source_steps,
                "route_parse_failures": source_fail,
            }
        )

    return sorted(unique), {
        "sources": provenance,
        "records": total_records,
        "parsed_steps": total_steps,
        "route_parse_failures": parse_failures,
        "unique_canonical_reactions": len(unique),
    }


# -----------------------------------------------------------------------------
# Atom-mapping cache and RXNMapper backend
# -----------------------------------------------------------------------------
def load_mapping_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rxn = rec.get("reaction_smiles")
                if rxn:
                    cache[rxn] = rec
            except Exception:
                continue
    return cache


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_rxnmapper():
    try:
        from rxnmapper import RXNMapper  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "RXNMapper is required to atom-map the unmapped SynLaD route reactions. "
            "Install/configure rxnmapper in this environment, or reuse a populated "
            "--mapping-cache created in another environment. Original import error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return RXNMapper()


def map_reactions_with_cache(
    reactions: Sequence[str],
    *,
    cache_path: Path,
    batch_size: int,
    mapper_backend: str,
    mapper_batch_size: int | None = None,
) -> dict[str, dict[str, Any]]:
    cache = load_mapping_cache(cache_path)
    missing = [r for r in reactions if r not in cache]
    if not missing:
        return cache
    if mapper_backend == "none":
        raise RuntimeError(
            f"Mapping cache misses for {len(missing)} reactions and --mapper=none. "
            "Populate the cache first or use --mapper=rxnmapper."
        )
    if mapper_backend != "rxnmapper":
        raise ValueError(f"Unknown mapper backend: {mapper_backend}")

    mapper = get_rxnmapper()
    actual_batch = int(mapper_batch_size or batch_size)
    for start in range(0, len(missing), actual_batch):
        batch = missing[start : start + actual_batch]
        out_records = []
        try:
            results = mapper.get_attention_guided_atom_maps(batch)
        except TypeError:
            # Compatibility fallback for versions requiring named argument.
            results = mapper.get_attention_guided_atom_maps(reactions=batch)
        if len(results) != len(batch):
            raise RuntimeError(
                f"RXNMapper returned {len(results)} results for batch of {len(batch)} reactions"
            )
        for rxn, result in zip(batch, results, strict=True):
            mapped = result.get("mapped_rxn") if isinstance(result, dict) else None
            confidence = result.get("confidence") if isinstance(result, dict) else None
            rec = {
                "reaction_smiles": rxn,
                "mapped_reaction_smiles": mapped,
                "mapping_confidence": confidence,
                "mapping_success": bool(mapped),
            }
            cache[rxn] = rec
            out_records.append(rec)
        append_jsonl(cache_path, out_records)
        done = min(start + actual_batch, len(missing))
        print(f"[atom-map] {done}/{len(missing)} newly mapped; cache={len(cache)}")
    return cache


# -----------------------------------------------------------------------------
# Template extraction
# -----------------------------------------------------------------------------
def get_template_extractor():
    try:
        from rdchiral.template_extractor import extract_from_reaction  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "rdchiral is required for atom-mapped reaction-template extraction. "
            "Install rdchiral in this environment. Original import error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return extract_from_reaction


def split_reaction_smiles(rxn: str) -> tuple[str, str, str]:
    parts = str(rxn).split(">")
    if len(parts) != 3:
        raise ValueError(f"Reaction SMILES must have reactants>reagents>products form: {rxn!r}")
    return parts[0], parts[1], parts[2]


def template_key(smarts: str) -> str:
    """
    Return a stable comparison key for an rdchiral-extracted template.

    rdchiral's extractor canonicalizes transforms in normal use.  We therefore
    preserve atom mapping/correspondence and only normalize whitespace and
    component order on each side.  A second map-stripped key is emitted only as
    a diagnostic; it is deliberately not used as the primary match criterion.
    """
    text = re.sub(r"\s+", "", str(smarts))
    if ">>" not in text:
        return text
    left, right = text.split(">>", 1)
    left = ".".join(sorted([x for x in left.split(".") if x]))
    right = ".".join(sorted([x for x in right.split(".") if x]))
    return f"{left}>>{right}"


def template_key_map_stripped(smarts: str) -> str:
    text = re.sub(r":\d+(?=\])", "", str(smarts))
    text = re.sub(r"\s+", "", text)
    if ">>" not in text:
        return text
    left, right = text.split(">>", 1)
    return f"{'.'.join(sorted(left.split('.')))}>>{'.'.join(sorted(right.split('.')))}"


def extract_rdchiral_template(mapped_rxn: str, rxn_id: str) -> dict[str, Any]:
    extract_from_reaction = get_template_extractor()
    reactants, reagents, products = split_reaction_smiles(mapped_rxn)
    record = {
        "_id": str(rxn_id),
        "reactants": reactants,
        "products": products,
        "reagents": reagents,
    }
    result = extract_from_reaction(record)
    if not isinstance(result, dict):
        raise RuntimeError("rdchiral template extractor did not return a dict")
    smarts = result.get("reaction_smarts") or result.get("retro_template") or result.get("template")
    if not smarts:
        raise RuntimeError(f"No reaction template found in extractor result keys={list(result.keys())}")
    # Standard rdchiral extractor emits retrosynthetic product>>reactants SMARTS.
    retro = str(smarts)
    if ">>" not in retro:
        raise RuntimeError(f"Unexpected template SMARTS without >>: {retro}")
    p_side, r_side = retro.split(">>", 1)
    forward = f"{r_side}>>{p_side}"
    return {
        "retro_template_smarts": retro,
        "forward_template_smarts": forward,
        "template_key": template_key(retro),
        "template_key_map_stripped": template_key_map_stripped(retro),
        "extractor_result": result,
    }


def load_template_library(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stripped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return exact, stripped
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not rec.get("template_extraction_success"):
                continue
            key = rec.get("template_key")
            skey = rec.get("template_key_map_stripped")
            if key:
                exact[key].append(rec)
            if skey:
                stripped[skey].append(rec)
    return exact, stripped


def build_reference_template_library(
    reference_reactions: Sequence[str],
    *,
    mapping_cache: dict[str, dict[str, Any]],
    template_cache_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    exact, stripped = load_template_library(template_cache_path)
    existing_rxns: set[str] = set()
    if template_cache_path.exists():
        with template_cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("reaction_smiles"):
                        existing_rxns.add(rec["reaction_smiles"])
                except Exception:
                    pass

    new_records: list[dict[str, Any]] = []
    success = 0
    failures = 0
    for idx, rxn in enumerate(reference_reactions):
        if rxn in existing_rxns:
            continue
        mrec = mapping_cache.get(rxn, {})
        mapped = mrec.get("mapped_reaction_smiles")
        rec: dict[str, Any] = {
            "reaction_smiles": rxn,
            "mapped_reaction_smiles": mapped,
            "mapping_confidence": mrec.get("mapping_confidence"),
            "mapping_success": bool(mapped),
        }
        if not mapped:
            rec.update(template_extraction_success=False, template_extraction_error="mapping_failed")
            failures += 1
        else:
            try:
                t = extract_rdchiral_template(mapped, rxn_id=f"ref_{idx}")
                rec.update(t)
                rec["template_extraction_success"] = True
                success += 1
            except Exception as exc:
                rec["template_extraction_success"] = False
                rec["template_extraction_error"] = f"{type(exc).__name__}: {exc}"
                failures += 1
        new_records.append(rec)
        if len(new_records) >= 256:
            append_jsonl(template_cache_path, new_records)
            new_records.clear()
            print(f"[template-extract] processed {idx + 1}/{len(reference_reactions)} reference reactions")
    if new_records:
        append_jsonl(template_cache_path, new_records)

    exact, stripped = load_template_library(template_cache_path)
    total_cache_records = 0
    cache_success = 0
    with template_cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_cache_records += 1
                try:
                    if json.loads(line).get("template_extraction_success"):
                        cache_success += 1
                except Exception:
                    pass

    return exact, stripped, {
        "template_cache_path": str(template_cache_path),
        "template_cache_sha256": sha256_file(template_cache_path) if template_cache_path.exists() else None,
        "reference_reactions_requested": len(reference_reactions),
        "template_cache_records": total_cache_records,
        "template_extraction_success_records": cache_success,
        "unique_template_keys": len(exact),
        "unique_map_stripped_template_keys": len(stripped),
    }


# -----------------------------------------------------------------------------
# Forward template execution
# -----------------------------------------------------------------------------
def _sanitize_outcome_mol(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def _compatible_permutations(
    rxn: rdChemReactions.ChemicalReaction,
    reactant_mols: Sequence[Chem.Mol],
    max_permutations: int,
) -> list[tuple[Chem.Mol, ...]]:
    n_templates = rxn.GetNumReactantTemplates()
    if n_templates != len(reactant_mols):
        return []
    if len(reactant_mols) <= 1:
        return [tuple(reactant_mols)]

    queries = [rxn.GetReactantTemplate(i) for i in range(n_templates)]
    perms: list[tuple[Chem.Mol, ...]] = []
    for perm_idx, perm in enumerate(itertools.permutations(reactant_mols, len(reactant_mols))):
        if perm_idx >= max_permutations:
            break
        ok = True
        for q, m in zip(queries, perm, strict=True):
            try:
                if not m.HasSubstructMatch(q):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            perms.append(tuple(perm))
    return perms


def execute_forward_template(
    forward_smarts: str,
    reactants: Sequence[str],
    generated_product: str,
    *,
    max_permutations: int,
    max_outcomes: int,
) -> dict[str, Any]:
    target_iso = canonical_product_key(generated_product, isomeric=True)
    target_rel = canonical_product_key(generated_product, isomeric=False)
    if target_iso is None or target_rel is None:
        return {
            "attempted": False,
            "reason": "invalid_generated_product",
            "outcome_count": 0,
            "exact_match": False,
            "stereo_relaxed_match": False,
        }

    mols: list[Chem.Mol] = []
    for s in reactants:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return {
                "attempted": False,
                "reason": "invalid_reactant",
                "outcome_count": 0,
                "exact_match": False,
                "stereo_relaxed_match": False,
            }
        mols.append(m)

    try:
        rxn = rdChemReactions.ReactionFromSmarts(forward_smarts)
        if rxn is None:
            raise ValueError("ReactionFromSmarts returned None")
        rxn.Initialize()
    except Exception as exc:
        return {
            "attempted": False,
            "reason": f"template_compile_error:{type(exc).__name__}",
            "outcome_count": 0,
            "exact_match": False,
            "stereo_relaxed_match": False,
        }

    perms = _compatible_permutations(rxn, mols, max_permutations=max_permutations)
    if not perms:
        return {
            "attempted": True,
            "reason": "no_compatible_reactant_assignment",
            "outcome_count": 0,
            "exact_match": False,
            "stereo_relaxed_match": False,
        }

    outcomes_iso: set[str] = set()
    outcomes_rel: set[str] = set()
    run_errors = 0
    for perm in perms:
        try:
            outcomes = rxn.RunReactants(perm)
        except Exception:
            run_errors += 1
            continue
        for outcome in outcomes:
            if len(outcomes_iso) >= max_outcomes:
                break
            clean = []
            for mol in outcome:
                sm = _sanitize_outcome_mol(mol)
                if sm is None:
                    clean = []
                    break
                clean.append(sm)
            if not clean:
                continue
            iso_parts = sorted(
                set(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in clean)
            )
            rel_parts = sorted(
                set(Chem.MolToSmiles(m, canonical=True, isomericSmiles=False) for m in clean)
            )
            outcomes_iso.add(".".join(iso_parts))
            outcomes_rel.add(".".join(rel_parts))
        if len(outcomes_iso) >= max_outcomes:
            break

    return {
        "attempted": True,
        "reason": "ok" if outcomes_iso else ("run_error" if run_errors else "no_products"),
        "compatible_assignment_count": len(perms),
        "run_error_count": run_errors,
        "outcome_count": len(outcomes_iso),
        "exact_match": target_iso in outcomes_iso,
        "stereo_relaxed_match": target_rel in outcomes_rel,
        "outcomes_isomeric_preview": sorted(outcomes_iso)[:20],
    }


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def parse_reactants_raw(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value)
    try:
        x = json.loads(text)
    except Exception:
        try:
            x = ast.literal_eval(text)
        except Exception:
            x = [text]
    if isinstance(x, (list, tuple)):
        return [str(y) for y in x]
    return [str(x)]


def _bool_metric(df: pd.DataFrame, col: str, mask: pd.Series | None = None) -> dict[str, Any]:
    sub = df if mask is None else df[mask]
    if col not in sub.columns:
        return {"count": 0, "numerator": 0, "fraction": None}
    vals = [v for v in (_as_bool_or_none(x) for x in sub[col].tolist()) if v is not None]
    num = sum(bool(v) for v in vals)
    return {"count": len(vals), "numerator": int(num), "fraction": safe_fraction(num, len(vals))}


def evaluate_generated_steps(
    f0_steps: pd.DataFrame,
    *,
    mapping_cache: dict[str, dict[str, Any]],
    reference_templates: dict[str, list[dict[str, Any]]],
    reference_templates_stripped: dict[str, list[dict[str, Any]]],
    only_reference_unsupported: bool,
    allow_map_stripped_fallback: bool,
    max_permutations: int,
    max_outcomes: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row_idx, row in f0_steps.iterrows():
        ref_exact = _as_bool_or_none(row.get("reference_exact_reaction_support"))
        evaluable = _as_bool_or_none(row.get("reaction_evaluable"))
        reactants = parse_reactants_raw(row.get("reactants_raw"))
        product = str(row.get("generated_product_raw"))
        canonical_rxn = row.get("reaction_canonical_isomeric")
        if canonical_rxn is None or (isinstance(canonical_rxn, float) and math.isnan(canonical_rxn)):
            canonical_rxn = canonical_reaction_smiles(reactants, product)
        else:
            canonical_rxn = str(canonical_rxn)

        targeted = bool(evaluable) and (not only_reference_unsupported or ref_exact is False)
        out = dict(row)
        out.update(
            {
                "f1_targeted": targeted,
                "f1_mapping_success": None,
                "f1_mapping_confidence": None,
                "f1_template_extraction_success": None,
                "f1_template_key": None,
                "f1_template_key_map_stripped": None,
                "f1_known_template_support": None,
                "f1_known_template_support_map_stripped": None,
                "f1_matched_reference_template_count": None,
                "f1_template_execution_attempted": None,
                "f1_template_execution_success_exact": None,
                "f1_template_execution_success_stereo_relaxed": None,
                "f1_execution_outcome_count": None,
                "f1_execution_reason": None,
                "f1_combined_exact_or_template_support": bool(ref_exact) if ref_exact is not None else None,
            }
        )

        if not targeted or not canonical_rxn:
            rows.append(out)
            continue

        mrec = mapping_cache.get(canonical_rxn, {})
        mapped = mrec.get("mapped_reaction_smiles")
        out["f1_mapping_success"] = bool(mapped)
        out["f1_mapping_confidence"] = mrec.get("mapping_confidence")
        if not mapped:
            out["f1_template_extraction_success"] = False
            out["f1_known_template_support"] = False
            out["f1_template_execution_attempted"] = False
            out["f1_template_execution_success_exact"] = False
            out["f1_template_execution_success_stereo_relaxed"] = False
            out["f1_execution_reason"] = "mapping_failed"
            out["f1_combined_exact_or_template_support"] = bool(ref_exact)
            rows.append(out)
            continue

        try:
            t = extract_rdchiral_template(mapped, rxn_id=f"gen_{row_idx}")
            out["f1_template_extraction_success"] = True
            out["f1_template_key"] = t["template_key"]
            out["f1_template_key_map_stripped"] = t["template_key_map_stripped"]
        except Exception as exc:
            out["f1_template_extraction_success"] = False
            out["f1_known_template_support"] = False
            out["f1_template_execution_attempted"] = False
            out["f1_template_execution_success_exact"] = False
            out["f1_template_execution_success_stereo_relaxed"] = False
            out["f1_execution_reason"] = f"template_extract_error:{type(exc).__name__}"
            out["f1_combined_exact_or_template_support"] = bool(ref_exact)
            rows.append(out)
            continue

        matches = reference_templates.get(t["template_key"], [])
        stripped_matches = reference_templates_stripped.get(t["template_key_map_stripped"], [])
        out["f1_known_template_support"] = bool(matches)
        out["f1_known_template_support_map_stripped"] = bool(stripped_matches)

        chosen_matches = matches
        if not chosen_matches and allow_map_stripped_fallback:
            chosen_matches = stripped_matches
        out["f1_matched_reference_template_count"] = len(chosen_matches)

        if not chosen_matches:
            out["f1_template_execution_attempted"] = False
            out["f1_template_execution_success_exact"] = False
            out["f1_template_execution_success_stereo_relaxed"] = False
            out["f1_execution_outcome_count"] = 0
            out["f1_execution_reason"] = "no_reference_template_match"
            out["f1_combined_exact_or_template_support"] = bool(ref_exact)
            rows.append(out)
            continue

        execution_exact = False
        execution_relaxed = False
        attempted_any = False
        total_outcomes = 0
        reasons = []
        # Templates sharing a key are usually equivalent; try a few distinct SMARTS
        # representatives defensively.
        seen_smarts: set[str] = set()
        for m in chosen_matches[:8]:
            fwd = m.get("forward_template_smarts")
            if not fwd or fwd in seen_smarts:
                continue
            seen_smarts.add(fwd)
            er = execute_forward_template(
                fwd,
                reactants,
                product,
                max_permutations=max_permutations,
                max_outcomes=max_outcomes,
            )
            attempted_any = attempted_any or bool(er.get("attempted"))
            total_outcomes += int(er.get("outcome_count") or 0)
            reasons.append(str(er.get("reason")))
            execution_exact = execution_exact or bool(er.get("exact_match"))
            execution_relaxed = execution_relaxed or bool(er.get("stereo_relaxed_match"))
            if execution_exact:
                break

        out["f1_template_execution_attempted"] = attempted_any
        out["f1_template_execution_success_exact"] = execution_exact
        out["f1_template_execution_success_stereo_relaxed"] = execution_relaxed
        out["f1_execution_outcome_count"] = total_outcomes
        out["f1_execution_reason"] = ";".join(sorted(set(reasons))) if reasons else "not_attempted"
        out["f1_combined_exact_or_template_support"] = bool(ref_exact) or execution_exact
        rows.append(out)

        if (row_idx + 1) % 250 == 0:
            print(f"[generated-template-eval] processed {row_idx + 1}/{len(f0_steps)} steps")

    return pd.DataFrame(rows)


def aggregate_routes(step_df: pd.DataFrame, route_id_col: str) -> pd.DataFrame:
    rows = []
    for route_id, g in step_df.groupby(route_id_col, dropna=False, sort=False):
        total_steps = len(g)
        targeted = g[g["f1_targeted"].map(lambda x: _as_bool_or_none(x) is True)]

        def all_bool(col: str) -> bool:
            vals = [_as_bool_or_none(x) for x in g[col].tolist()]
            return bool(vals) and all(v is True for v in vals)

        def any_bool(col: str) -> bool:
            vals = [_as_bool_or_none(x) for x in g[col].tolist()]
            return any(v is True for v in vals)

        exact_supported = sum(_as_bool_or_none(x) is True for x in g["reference_exact_reaction_support"].tolist())
        template_exec = sum(_as_bool_or_none(x) is True for x in g["f1_template_execution_success_exact"].tolist())
        combined = sum(_as_bool_or_none(x) is True for x in g["f1_combined_exact_or_template_support"].tolist())

        rows.append(
            {
                route_id_col: route_id,
                "reaction_steps": total_steps,
                "f1_targeted_steps": len(targeted),
                "exact_reference_supported_steps": exact_supported,
                "template_executable_steps": template_exec,
                "combined_supported_steps": combined,
                "combined_supported_step_fraction": safe_fraction(combined, total_steps),
                "all_steps_exact_reference_supported": all_bool("reference_exact_reaction_support"),
                "any_step_template_executable": any_bool("f1_template_execution_success_exact"),
                "all_steps_combined_exact_or_template_supported": all_bool(
                    "f1_combined_exact_or_template_support"
                ),
                "any_step_unsupported_after_f1": not all_bool("f1_combined_exact_or_template_support"),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2F-1 known-template executability analysis")
    p.add_argument("--f0-step-csv", required=True, help="Stage 2F-0 reference_reaction_support_steps.csv")
    p.add_argument(
        "--reference-route-file",
        action="append",
        default=[],
        help="Reference route-token JSONL/CSV. Repeat for train/val/test.",
    )
    p.add_argument("--reference-route-column", default="serialized_route_tokens_json")
    p.add_argument("--route-id-column", default="auto")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mapping-cache", default=None)
    p.add_argument("--template-cache", default=None)
    p.add_argument("--mapper", choices=["rxnmapper", "none"], default="rxnmapper")
    p.add_argument("--mapping-batch-size", type=int, default=32)
    p.add_argument(
        "--all-generated-steps",
        action="store_true",
        help="Evaluate all generated steps, not only exact-reference-unsupported steps.",
    )
    p.add_argument(
        "--allow-map-stripped-template-fallback",
        action="store_true",
        help=(
            "If exact canonical template keys do not match, allow a map-number-stripped key as fallback. "
            "This is less strict and should be reported separately."
        ),
    )
    p.add_argument("--max-permutations", type=int, default=24)
    p.add_argument("--max-outcomes", type=int, default=256)
    p.add_argument("--quiet-rdkit", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.quiet_rdkit:
        RDLogger.DisableLog("rdApp.*")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_cache_path = Path(args.mapping_cache) if args.mapping_cache else out_dir / "atom_mapping_cache.jsonl"
    template_cache_path = Path(args.template_cache) if args.template_cache else out_dir / "reference_template_library.jsonl"

    f0_path = Path(args.f0_step_csv)
    f0 = pd.read_csv(f0_path)
    required = {
        "reactants_raw",
        "generated_product_raw",
        "reaction_evaluable",
        "reference_exact_reaction_support",
    }
    missing = required - set(f0.columns)
    if missing:
        raise KeyError(f"F0 step CSV missing required columns: {sorted(missing)}")

    route_id_col = pick_column(f0.columns.tolist(), args.route_id_column, AUTO_ROUTE_ID_COLUMNS)

    reference_files = [Path(x) for x in args.reference_route_file]
    if not reference_files:
        raise ValueError("At least one --reference-route-file is required")
    for p in reference_files:
        if not p.exists():
            raise FileNotFoundError(p)

    print("[1/6] Collecting unique reference reactions ...")
    reference_rxns, reference_meta = collect_unique_reference_reactions(
        reference_files, args.reference_route_column
    )
    print(json.dumps(reference_meta, indent=2))

    # Generated reactions needed for F1 mapping.
    target_mask = f0["reaction_evaluable"].map(lambda x: _as_bool_or_none(x) is True)
    if not args.all_generated_steps:
        target_mask &= f0["reference_exact_reaction_support"].map(lambda x: _as_bool_or_none(x) is False)
    generated_rxns: list[str] = []
    for _, row in f0[target_mask].iterrows():
        rxn = row.get("reaction_canonical_isomeric")
        if rxn is None or (isinstance(rxn, float) and math.isnan(rxn)):
            rxn = canonical_reaction_smiles(parse_reactants_raw(row["reactants_raw"]), row["generated_product_raw"])
        if rxn:
            generated_rxns.append(str(rxn))
    generated_rxns = sorted(set(generated_rxns))

    print(f"[2/6] Atom mapping {len(reference_rxns)} reference + {len(generated_rxns)} generated unique reactions ...")
    all_to_map = sorted(set(reference_rxns) | set(generated_rxns))
    mapping_cache = map_reactions_with_cache(
        all_to_map,
        cache_path=mapping_cache_path,
        batch_size=args.mapping_batch_size,
        mapper_backend=args.mapper,
    )

    print("[3/6] Building/loading reference template library ...")
    ref_templates, ref_templates_stripped, template_meta = build_reference_template_library(
        reference_rxns,
        mapping_cache=mapping_cache,
        template_cache_path=template_cache_path,
    )
    print(json.dumps(template_meta, indent=2))

    print("[4/6] Evaluating generated reaction templates ...")
    step_df = evaluate_generated_steps(
        f0,
        mapping_cache=mapping_cache,
        reference_templates=ref_templates,
        reference_templates_stripped=ref_templates_stripped,
        only_reference_unsupported=not args.all_generated_steps,
        allow_map_stripped_fallback=args.allow_map_stripped_template_fallback,
        max_permutations=args.max_permutations,
        max_outcomes=args.max_outcomes,
    )

    print("[5/6] Aggregating route-level evidence ...")
    route_df = aggregate_routes(step_df, route_id_col)

    targeted_mask = step_df["f1_targeted"].map(lambda x: _as_bool_or_none(x) is True)
    summary = {
        "evaluation": {
            "stage": "stage2f1_known_template_support_and_executability",
            "policy": "atom_map_extract_reference_templates_match_and_reapply",
            "interpretation": (
                "Known-template structural support/executability only; template agreement or execution does not prove "
                "experimental feasibility."
            ),
            "only_exact_reference_unsupported_steps": not args.all_generated_steps,
            "map_stripped_template_fallback_enabled": bool(args.allow_map_stripped_template_fallback),
        },
        "counts": {
            "f0_steps": len(step_df),
            "targeted_steps": int(targeted_mask.sum()),
            "routes": len(route_df),
            "unique_reference_reactions": len(reference_rxns),
            "unique_generated_reactions_targeted": len(generated_rxns),
        },
        "step_level": {
            "mapping_success": _bool_metric(step_df, "f1_mapping_success", targeted_mask),
            "template_extraction_success": _bool_metric(step_df, "f1_template_extraction_success", targeted_mask),
            "known_template_support_exact_key": _bool_metric(step_df, "f1_known_template_support", targeted_mask),
            "known_template_support_map_stripped": _bool_metric(
                step_df, "f1_known_template_support_map_stripped", targeted_mask
            ),
            "template_execution_success_exact_product": _bool_metric(
                step_df, "f1_template_execution_success_exact", targeted_mask
            ),
            "template_execution_success_stereo_relaxed": _bool_metric(
                step_df, "f1_template_execution_success_stereo_relaxed", targeted_mask
            ),
            "combined_exact_reference_or_template_executable": _bool_metric(
                step_df, "f1_combined_exact_or_template_support"
            ),
        },
        "route_level": {
            "all_steps_exact_reference_supported": _bool_metric(
                route_df, "all_steps_exact_reference_supported"
            ),
            "any_step_template_executable": _bool_metric(route_df, "any_step_template_executable"),
            "all_steps_combined_exact_or_template_supported": _bool_metric(
                route_df, "all_steps_combined_exact_or_template_supported"
            ),
            "any_step_unsupported_after_f1": _bool_metric(route_df, "any_step_unsupported_after_f1"),
        },
        "input": {
            "f0_step_csv": str(f0_path),
            "f0_step_csv_sha256": sha256_file(f0_path),
            "route_id_column": route_id_col,
        },
        "reference_corpus": reference_meta,
        "atom_mapping": {
            "backend": args.mapper,
            "cache_path": str(mapping_cache_path),
            "cache_sha256": sha256_file(mapping_cache_path) if mapping_cache_path.exists() else None,
        },
        "template_library": template_meta,
        "parameters": {
            "mapping_batch_size": args.mapping_batch_size,
            "max_permutations": args.max_permutations,
            "max_outcomes": args.max_outcomes,
        },
    }

    step_path = out_dir / "template_executability_steps.csv"
    route_path = out_dir / "template_executability_routes.csv"
    summary_path = out_dir / "template_executability_summary.json"
    step_df.to_csv(step_path, index=False)
    route_df.to_csv(route_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[6/6] Done")
    print(f"  steps   : {step_path}")
    print(f"  routes  : {route_path}")
    print(f"  summary : {summary_path}")
    print(json.dumps(summary["step_level"], indent=2))


if __name__ == "__main__":
    main()
