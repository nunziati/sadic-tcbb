"""Helpers for the RCSB Search and Data APIs used to build the dataset.

Dataset policy:
- biological assembly 1 only;
- protein-only entries (polymer_composition homomeric/heteromeric protein);
- 200 <= assembly polymer atom count <= 40000.
"""

import json
import time

import requests

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_GRAPHQL_URL = "https://data.rcsb.org/graphql"

MIN_ATOMS = 200
MAX_ATOMS = 40000
PROTEIN_COMPOSITIONS = ["homomeric protein", "heteromeric protein"]


def _attr(attribute, operator, value):
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def assembly_query(min_atoms, max_atoms, start=0, rows=10000):
    """Search query: assembly-1 of protein-only entries within an atom-count range."""
    query = {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            _attr("rcsb_entry_info.polymer_composition", "in", PROTEIN_COMPOSITIONS),
            _attr("rcsb_assembly_container_identifiers.assembly_id", "exact_match", "1"),
            _attr("rcsb_assembly_info.polymer_atom_count", "greater_or_equal", min_atoms),
            _attr("rcsb_assembly_info.polymer_atom_count", "less_or_equal", max_atoms),
        ],
    }
    return {
        "query": query,
        "return_type": "assembly",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "results_verbosity": "compact",
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}],
        },
    }


def run_search(payload, retries=5):
    for attempt in range(retries):
        resp = requests.get(
            SEARCH_URL, params={"json": json.dumps(payload)}, timeout=120
        )
        if resp.status_code == 204:  # no content = zero results
            return {"total_count": 0, "result_set": []}
        if resp.ok:
            return resp.json()
        time.sleep(2 * (attempt + 1))
    resp.raise_for_status()


def count_assemblies(min_atoms, max_atoms):
    payload = assembly_query(min_atoms, max_atoms, start=0, rows=0)
    payload["request_options"]["return_counts"] = True
    del payload["request_options"]["paginate"]
    result = run_search(payload)
    return result.get("total_count", 0)


def fetch_assembly_ids(min_atoms, max_atoms, limit=None):
    """Fetch all assembly ids (entry ids, since assembly id is fixed to 1) in a range.

    Recursively splits the range whenever it holds more than one page (10000),
    so every id in the range is retrieved (no sort-order bias).
    """
    total = count_assemblies(min_atoms, max_atoms)
    if total == 0:
        return []
    if total > 10000 and min_atoms < max_atoms:
        mid = (min_atoms + max_atoms) // 2
        return fetch_assembly_ids(min_atoms, mid) + fetch_assembly_ids(mid + 1, max_atoms)
    result = run_search(assembly_query(min_atoms, max_atoms))
    ids = [r.split("-")[0].lower() for r in result.get("result_set", [])]
    if limit:
        ids = ids[:limit]
    return ids


ENTRY_ASSEMBLY_GQL = """
query($entry_ids: [String!]!, $assembly_ids: [String!]!) {
  entries(entry_ids: $entry_ids) {
    rcsb_id
    rcsb_entry_info {
      polymer_composition
      deposited_atom_count
      assembly_count
      resolution_combined
    }
    exptl { method }
    struct { title }
  }
  assemblies(assembly_ids: $assembly_ids) {
    rcsb_id
    pdbx_struct_assembly { oligomeric_details oligomeric_count }
    rcsb_assembly_info {
      polymer_atom_count
      atom_count
      polymer_entity_instance_count
      modeled_polymer_monomer_count
    }
  }
}
"""


def fetch_entry_and_assembly_info(pdb_ids, batch_size=200, sleep=0.3):
    """Batched Data API (GraphQL) lookup for entries + their assembly 1."""
    out = {}
    ids = [p.lower() for p in pdb_ids]
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        variables = {
            "entry_ids": [p.upper() for p in batch],
            "assembly_ids": [f"{p.upper()}-1" for p in batch],
        }
        for attempt in range(5):
            resp = requests.post(
                DATA_GRAPHQL_URL,
                json={"query": ENTRY_ASSEMBLY_GQL, "variables": variables},
                timeout=120,
            )
            if resp.ok:
                break
            time.sleep(2 * (attempt + 1))
        resp.raise_for_status()
        data = resp.json()["data"]
        entries = {e["rcsb_id"].lower(): e for e in (data.get("entries") or []) if e}
        assemblies = {
            a["rcsb_id"].split("-")[0].lower(): a
            for a in (data.get("assemblies") or [])
            if a
        }
        for pid in batch:
            entry = entries.get(pid)
            asm = assemblies.get(pid)
            info = {"pdb_id": pid, "found": entry is not None}
            if entry:
                ei = entry.get("rcsb_entry_info") or {}
                res = ei.get("resolution_combined") or []
                info.update(
                    polymer_composition=ei.get("polymer_composition"),
                    deposited_atom_count=ei.get("deposited_atom_count"),
                    assembly_count=ei.get("assembly_count"),
                    resolution=res[0] if res else None,
                    method=(entry.get("exptl") or [{}])[0].get("method"),
                    title=(entry.get("struct") or {}).get("title", ""),
                )
            if asm:
                ai = asm.get("rcsb_assembly_info") or {}
                pa = asm.get("pdbx_struct_assembly") or {}
                info.update(
                    assembly_polymer_atom_count=ai.get("polymer_atom_count"),
                    assembly_atom_count=ai.get("atom_count"),
                    assembly_chains=ai.get("polymer_entity_instance_count"),
                    oligomeric_details=pa.get("oligomeric_details"),
                    oligomeric_count=pa.get("oligomeric_count"),
                )
            out[pid] = info
        time.sleep(sleep)
    return out


def passes_filters(info):
    """Apply the dataset policy (module docstring) to a metadata record. Returns (bool, reason)."""
    if not info.get("found"):
        return False, "entry not found (obsolete?)"
    if info.get("polymer_composition") not in PROTEIN_COMPOSITIONS:
        return False, f"composition: {info.get('polymer_composition')}"
    n = info.get("assembly_polymer_atom_count")
    if n is None:
        return False, "no assembly 1 info"
    if n < MIN_ATOMS:
        return False, f"too small ({n} atoms)"
    if n > MAX_ATOMS:
        return False, f"too large ({n} atoms)"
    return True, "ok"
