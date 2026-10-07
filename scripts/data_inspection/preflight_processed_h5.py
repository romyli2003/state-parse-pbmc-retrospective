#!/usr/bin/env python3
"""Read-only structural preflight for processed HDF5 and H5AD files.

The tool inventories HDF5 objects and samples only bounded annotation-like values.
It never samples candidate matrix payloads and never loads an AnnData object.
Biological roles inferred from field names are reported only as candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


VERSION = "0.1.0"
SCHEMA_VERSION = "1.0"
SUPPORTED_SUFFIXES = {".h5", ".hdf5", ".h5ad"}
DEFAULT_OUTPUT_DIR = Path("processed_file_preflight_report")

ROLE_ALIASES = {
    "sample": ("sample", "sample_id", "sampleid", "library", "library_id"),
    "donor": ("donor", "donor_id", "subject", "participant", "patient", "individual"),
    "perturbation_or_condition": (
        "perturbation", "condition", "treatment", "stim", "stimulation", "stimulus",
        "cytokine", "treat", "group",
    ),
    "control": ("control", "is_control", "control_status", "vehicle", "baseline"),
    "cell_type": (
        "cell_type", "celltype", "cell_label", "celllabel", "annotation", "cell_annotation",
        "cell_ontology", "cluster_label",
    ),
    "time_point": ("time", "timepoint", "time_point", "duration", "hour", "hours", "day"),
    "batch": ("batch", "batch_id", "lane", "plate", "run", "orig_ident", "orig.ident"),
    "platform": ("platform", "technology", "chemistry", "assay", "sequencing_platform"),
}
CONTROL_VALUE_TOKENS = {
    "control", "ctrl", "untreated", "unstimulated", "baseline", "vehicle", "mock", "pbs",
}
FEATURE_ID_NAMES = {
    "_index", "feature_id", "feature_ids", "gene_id", "gene_ids", "gene_name", "gene_names",
    "symbol", "symbols", "name", "names", "id", "ids",
}
OBS_INDEX_NAMES = {"_index", "barcode", "barcodes", "cell_id", "cell_ids", "obs_names"}
MATRIX_NAMES = {"x", "matrix", "counts", "count", "expression", "data"}
SPARSE_ENCODINGS = {"csr_matrix", "csc_matrix", "csr", "csc"}


class PreflightError(Exception):
    """Expected input or inspection error suitable for a concise CLI message."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="H5/HDF5/H5AD files or directories")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-sample-values", type=positive_int, default=8,
        help="maximum values sampled from any eligible annotation dataset (default: 8)",
    )
    parser.add_argument(
        "--max-string-chars", type=positive_int, default=160,
        help="maximum characters retained for an individual sampled value (default: 160)",
    )
    parser.add_argument(
        "--max-markdown-items", type=positive_int, default=30,
        help="maximum paths or fields shown per Markdown section (default: 30)",
    )
    return parser.parse_args(argv)


def utc_iso(timestamp: float | None = None) -> str:
    moment = datetime.now(timezone.utc) if timestamp is None else datetime.fromtimestamp(timestamp, timezone.utc)
    return moment.isoformat()


def normalized_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def json_value(value: Any, max_string_chars: int) -> Any:
    """Convert a small scalar/sample value to bounded JSON-compatible data."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value if len(value) <= max_string_chars else value[:max_string_chars] + "…"
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return json_value(str(value), max_string_chars)


def shape_list(shape: Iterable[int] | None) -> list[int] | None:
    return None if shape is None else [int(item) for item in shape]


def logical_size_bytes(dataset: h5py.Dataset) -> int | None:
    if dataset.size is None or h5py.check_dtype(vlen=dataset.dtype) is not None or dataset.dtype.hasobject:
        return None
    return int(dataset.size * dataset.dtype.itemsize)


def attribute_inventory(
    obj: h5py.Group | h5py.Dataset,
    max_values: int,
    max_string_chars: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in sorted(obj.attrs.keys()):
        record: dict[str, Any] = {"name": str(name)}
        try:
            attr_id = obj.attrs.get_id(name)
            record["shape"] = shape_list(attr_id.shape)
            record["dtype"] = str(attr_id.dtype)
            item_count = int(np.prod(attr_id.shape, dtype=np.int64)) if attr_id.shape else 1
            if item_count <= max_values:
                raw = obj.attrs[name]
                values = np.asarray(raw).reshape(-1).tolist()
                record["values"] = [json_value(value, max_string_chars) for value in values]
                record["values_truncated"] = False
            else:
                record["values"] = []
                record["values_truncated"] = True
        except Exception as exc:  # An individual malformed attribute should not abort the file inventory.
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return records


def scalar_attribute(obj: h5py.Group | h5py.Dataset, name: str) -> Any | None:
    if name not in obj.attrs:
        return None
    try:
        attr_id = obj.attrs.get_id(name)
        item_count = int(np.prod(attr_id.shape, dtype=np.int64)) if attr_id.shape else 1
        if item_count > 4:
            return None
        value = np.asarray(obj.attrs[name]).reshape(-1).tolist()
        decoded = [json_value(item, 160) for item in value]
        return decoded[0] if len(decoded) == 1 else decoded
    except Exception:
        return None


def dataset_is_matrix_payload(path: str) -> bool:
    parts = [part.lower() for part in Path(path).parts if part != "/"]
    if not parts:
        return False
    if parts[-1] in {"data", "indices", "indptr"} and any(
        part in {"x", "raw", "layers", "matrix", "counts"} for part in parts[:-1]
    ):
        return True
    if parts[0] == "x" or parts[0] == "layers":
        return True
    return parts[0] == "raw" and (len(parts) == 1 or parts[1] == "x")


def sample_eligible(path: str, dataset: h5py.Dataset) -> tuple[bool, str]:
    """Allow bounded samples only for one-dimensional annotation/identifier datasets."""
    if dataset.ndim > 1 or dataset_is_matrix_payload(path):
        return False, "matrix payload or multidimensional dataset"
    parts = [part.lower() for part in Path(path).parts if part != "/"]
    leaf = parts[-1] if parts else ""
    if not parts:
        return False, "root dataset is not annotation-like"
    if parts[0] in {"obs", "var"}:
        if leaf == "codes":
            return False, "categorical codes are not sampled; categories are sampled instead"
        return True, "embedded obs/var annotation or index"
    if "features" in parts or leaf in FEATURE_ID_NAMES or leaf in OBS_INDEX_NAMES or leaf == "shape":
        return True, "identifier or small structural dataset"
    return False, "not recognized as annotation, identifier, or small structural metadata"


def bounded_dataset_sample(
    dataset: h5py.Dataset,
    limit: int,
    max_string_chars: int,
) -> tuple[list[Any], bool]:
    if dataset.ndim == 0:
        values = [dataset[()]]
        truncated = False
    else:
        count = min(int(dataset.shape[0]), limit)
        values = np.asarray(dataset[:count]).reshape(-1).tolist()
        values = values[:limit]
        truncated = int(dataset.shape[0]) > count
    return [json_value(value, max_string_chars) for value in values], truncated


def dataset_record(
    path: str,
    dataset: h5py.Dataset,
    max_values: int,
    max_string_chars: int,
) -> dict[str, Any]:
    eligible, reason = sample_eligible(path, dataset)
    record: dict[str, Any] = {
        "path": path,
        "kind": "dataset",
        "shape": shape_list(dataset.shape),
        "dtype": str(dataset.dtype),
        "chunks": shape_list(dataset.chunks) if dataset.chunks is not None else None,
        "compression": dataset.compression,
        "compression_options": json_value(dataset.compression_opts, max_string_chars),
        "shuffle": bool(dataset.shuffle),
        "fletcher32": bool(dataset.fletcher32),
        "logical_size_bytes": logical_size_bytes(dataset),
        "storage_size_bytes": int(dataset.id.get_storage_size()),
        "attributes": attribute_inventory(dataset, max_values, max_string_chars),
        "sample_policy": {"eligible": eligible, "reason": reason},
        "sample_values": [],
        "sample_truncated": False,
    }
    if eligible:
        try:
            record["sample_values"], record["sample_truncated"] = bounded_dataset_sample(
                dataset, max_values, max_string_chars,
            )
        except Exception as exc:
            record["sample_error"] = f"{type(exc).__name__}: {exc}"
    return record


def group_record(
    path: str,
    group: h5py.Group,
    max_values: int,
    max_string_chars: int,
) -> dict[str, Any]:
    return {
        "path": path,
        "kind": "group",
        "child_count": len(group),
        "attributes": attribute_inventory(group, max_values, max_string_chars),
    }


def small_shape(group: h5py.Group) -> list[int] | None:
    if "shape" in group.attrs:
        try:
            raw = np.asarray(group.attrs["shape"]).reshape(-1)
            if 1 <= raw.size <= 4:
                return [int(value) for value in raw]
        except (TypeError, ValueError, OSError):
            pass
    if "shape" in group and isinstance(group["shape"], h5py.Dataset):
        shape_dataset = group["shape"]
        if shape_dataset.ndim <= 1 and shape_dataset.size is not None and shape_dataset.size <= 4:
            try:
                return [int(value) for value in np.asarray(shape_dataset[...]).reshape(-1)]
            except (TypeError, ValueError, OSError):
                pass
    return None


def candidate_matrices(handle: h5py.File) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def inspect(path_without_slash: str, obj: h5py.Group | h5py.Dataset) -> None:
        path = "/" + path_without_slash
        leaf = Path(path).name.lower()
        if isinstance(obj, h5py.Group):
            children = set(obj.keys())
            encoding = scalar_attribute(obj, "encoding-type")
            sparse_components = {"data", "indices", "indptr"}.issubset(children)
            if encoding in SPARSE_ENCODINGS or sparse_components:
                evidence = []
                if encoding in SPARSE_ENCODINGS:
                    evidence.append(f"encoding-type={encoding}")
                if sparse_components:
                    evidence.append("contains data, indices, and indptr datasets")
                candidates.append({
                    "classification": "structural_candidate",
                    "path": path,
                    "organization": "sparse_group",
                    "sparse_encoding": encoding,
                    "dimensions": small_shape(obj),
                    "dimension_order": "unknown until the application schema and feature/observation alignment are verified",
                    "count_status": "candidate_only",
                    "count_evidence": "path name suggests counts" if "count" in leaf else "no count semantics confirmed",
                    "evidence": evidence,
                })
        elif obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
            parts = [part.lower() for part in Path(path).parts if part != "/"]
            matrix_named = leaf in MATRIX_NAMES or (parts and parts[0] in {"x", "layers", "raw", "matrix"})
            if matrix_named:
                candidates.append({
                    "classification": "structural_candidate",
                    "path": path,
                    "organization": "dense_dataset",
                    "sparse_encoding": None,
                    "dimensions": shape_list(obj.shape),
                    "dimension_order": "unknown until the application schema and feature/observation alignment are verified",
                    "count_status": "candidate_only",
                    "count_evidence": "path name suggests counts" if "count" in leaf else "no count semantics confirmed",
                    "evidence": ["two-dimensional numeric dataset with a matrix-like path"],
                })

    handle.visititems(inspect)
    return sorted(candidates, key=lambda item: item["path"])


def annotation_group(handle: h5py.File, kind: str) -> h5py.Group | None:
    if kind in handle and isinstance(handle[kind], h5py.Group):
        return handle[kind]
    matrix_features = f"matrix/{'features' if kind == 'var' else 'barcodes'}"
    if matrix_features in handle and isinstance(handle[matrix_features], h5py.Group):
        return handle[matrix_features]
    return None


def categorical_details(
    group: h5py.Group,
    max_values: int,
    max_string_chars: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "encoding_type": scalar_attribute(group, "encoding-type"),
        "codes_path": None,
        "categories_path": None,
        "category_count": None,
        "category_examples": [],
        "examples_truncated": False,
    }
    if "codes" in group and isinstance(group["codes"], h5py.Dataset):
        result["codes_path"] = group["codes"].name
    if "categories" in group and isinstance(group["categories"], h5py.Dataset):
        categories = group["categories"]
        result["categories_path"] = categories.name
        result["category_count"] = int(categories.size)
        try:
            values, truncated = bounded_dataset_sample(categories, max_values, max_string_chars)
            result["category_examples"] = values
            result["examples_truncated"] = truncated
        except Exception as exc:
            result["sample_error"] = f"{type(exc).__name__}: {exc}"
    return result


def metadata_fields(
    handle: h5py.File,
    kind: str,
    max_values: int,
    max_string_chars: int,
) -> list[dict[str, Any]]:
    group = annotation_group(handle, kind)
    if group is None:
        return []
    fields: list[dict[str, Any]] = []
    for name in sorted(group.keys()):
        obj = group[name]
        field: dict[str, Any] = {
            "classification": "confirmed_fact",
            "name": name,
            "path": obj.name,
            "storage_kind": "group" if isinstance(obj, h5py.Group) else "dataset",
        }
        if isinstance(obj, h5py.Dataset):
            field.update({"shape": shape_list(obj.shape), "dtype": str(obj.dtype), "examples": []})
            eligible, _ = sample_eligible(obj.name, obj)
            if eligible:
                try:
                    field["examples"], field["examples_truncated"] = bounded_dataset_sample(
                        obj, max_values, max_string_chars,
                    )
                except Exception as exc:
                    field["sample_error"] = f"{type(exc).__name__}: {exc}"
        else:
            field["categorical_encoding"] = categorical_details(obj, max_values, max_string_chars)
        fields.append(field)
    return fields


def semantic_candidates(obs_fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for field in obs_fields:
        normalized = normalized_name(field["name"])
        examples = field.get("examples", [])
        categorical = field.get("categorical_encoding", {})
        if categorical:
            examples = categorical.get("category_examples", [])
        for role, aliases in ROLE_ALIASES.items():
            normalized_aliases = {normalized_name(alias) for alias in aliases}
            exact = normalized in normalized_aliases
            name_part = any(
                alias and (normalized.startswith(alias + "_") or normalized.endswith("_" + alias))
                for alias in normalized_aliases
            )
            if exact or name_part:
                candidates.append({
                    "classification": "structural_candidate",
                    "role": role,
                    "field": field["name"],
                    "path": field["path"],
                    "evidence": "field-name exact match" if exact else "field-name token match",
                    "biological_meaning_confirmed": False,
                })
        normalized_examples = {normalized_name(str(value)) for value in examples}
        if normalized_examples & CONTROL_VALUE_TOKENS and not any(
            candidate["field"] == field["name"] and candidate["role"] == "control" for candidate in candidates
        ):
            candidates.append({
                "classification": "structural_candidate",
                "role": "control",
                "field": field["name"],
                "path": field["path"],
                "evidence": "bounded examples contain a common control-like token",
                "biological_meaning_confirmed": False,
            })
    return sorted(candidates, key=lambda item: (item["role"], item["field"]))


def identifier_summary(
    fields: list[dict[str, Any]],
    kind: str,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = FEATURE_ID_NAMES if kind == "feature" else OBS_INDEX_NAMES
    normalized_accepted = {normalized_name(name) for name in accepted}
    paths = {
        field["path"]
        for field in fields
        if normalized_name(field["name"]) in normalized_accepted
    }
    for obj in objects:
        if obj["kind"] != "dataset":
            continue
        parts = [part for part in Path(obj["path"]).parts if part != "/"]
        if not parts:
            continue
        leaf = normalized_name(parts[-1])
        context = {normalized_name(part) for part in parts[:-1]}
        if leaf not in normalized_accepted:
            continue
        if kind == "feature" and not ({"var", "features"} & context):
            continue
        if kind == "observation" and "obs" not in context and leaf not in {
            normalized_name("barcode"), normalized_name("barcodes"), normalized_name("obs_names"),
        }:
            continue
        paths.add(obj["path"])
    candidate_paths = sorted(paths)
    return {
        "classification": "structural_candidate" if candidate_paths else "unknown",
        "available": bool(candidate_paths),
        "candidate_paths": candidate_paths,
        "semantics_confirmed": False,
    }


def unknowns_for_file(
    matrices: list[dict[str, Any]],
    semantics: list[dict[str, Any]],
    feature_ids: dict[str, Any],
    obs_index: dict[str, Any],
) -> list[str]:
    unknowns = [
        "Whether any candidate matrix contains raw integer counts, normalized values, or another representation.",
        "The biological meanings and provenance of name-matched metadata fields.",
        "Sample-to-donor-to-condition mappings and whether controls are matched.",
        "Whether cells and feature annotations are aligned to every candidate matrix.",
    ]
    present_roles = {item["role"] for item in semantics}
    for role in ROLE_ALIASES:
        if role not in present_roles:
            unknowns.append(f"No structural candidate was found for the {role!r} field.")
    if not matrices:
        unknowns.append("No matrix-shaped dense dataset or sparse group was structurally identified.")
    if not feature_ids["available"]:
        unknowns.append("No candidate feature-identifier field was structurally identified.")
    if not obs_index["available"]:
        unknowns.append("No candidate barcode/observation-index field was structurally identified.")
    return unknowns


def inspect_file(path: Path, max_values: int, max_string_chars: int) -> dict[str, Any]:
    try:
        before = path.stat()
    except OSError as exc:
        raise PreflightError(f"cannot stat input: {exc}") from exc
    if not os.access(path, os.R_OK):
        raise PreflightError("input is not readable")
    if not h5py.is_hdf5(path):
        raise PreflightError("file does not have a readable HDF5 signature")

    try:
        with h5py.File(path, "r") as handle:
            objects: list[dict[str, Any]] = [
                group_record("/", handle, max_values, max_string_chars),
            ]

            def collect(path_without_slash: str, obj: h5py.Group | h5py.Dataset) -> None:
                full_path = "/" + path_without_slash
                if isinstance(obj, h5py.Group):
                    objects.append(group_record(full_path, obj, max_values, max_string_chars))
                else:
                    objects.append(dataset_record(full_path, obj, max_values, max_string_chars))

            handle.visititems(collect)
            matrices = candidate_matrices(handle)
            obs_fields = metadata_fields(handle, "obs", max_values, max_string_chars)
            var_fields = metadata_fields(handle, "var", max_values, max_string_chars)
            semantics = semantic_candidates(obs_fields)
            feature_ids = identifier_summary(var_fields, "feature", objects)
            observation_index = identifier_summary(obs_fields, "observation", objects)
            root_attrs = attribute_inventory(handle, max_values, max_string_chars)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreflightError(f"HDF5 inspection failed: {type(exc).__name__}: {exc}") from exc

    try:
        after = path.stat()
    except OSError as exc:
        raise PreflightError(f"input became inaccessible after inspection: {exc}") from exc
    objects.sort(key=lambda item: (item["path"], item["kind"]))
    groups = sum(item["kind"] == "group" for item in objects)
    datasets = sum(item["kind"] == "dataset" for item in objects)
    return {
        "status": "ok",
        "path": str(path),
        "confirmed_facts": {
            "format": "HDF5 container; application-level H5AD semantics are only structural unless stated",
            "suffix": path.suffix.lower(),
            "size_bytes": int(before.st_size),
            "mtime_utc": utc_iso(before.st_mtime),
            "mode": "read-only",
            "group_count_including_root": groups,
            "dataset_count": datasets,
            "root_attributes": root_attrs,
            "unchanged_during_inspection": (
                before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
            ),
        },
        "objects": objects,
        "structural_candidates": {
            "matrices": matrices,
            "feature_identifiers": feature_ids,
            "feature_annotation_fields": var_fields,
            "barcode_or_observation_index": observation_index,
            "observation_metadata_fields": obs_fields,
            "semantic_fields": semantics,
        },
        "unknowns": unknowns_for_file(matrices, semantics, feature_ids, observation_index),
        "safety": {
            "matrix_values_sampled": False,
            "full_matrix_loaded": False,
            "anndata_loaded": False,
            "input_open_mode": "r",
        },
    }


def resolve_inputs(inputs: list[Path]) -> tuple[list[Path], list[str]]:
    resolved: list[Path] = []
    errors: list[str] = []
    for supplied in inputs:
        path = supplied.expanduser()
        if not path.exists():
            errors.append(f"{supplied}: input does not exist")
            continue
        if path.is_dir():
            try:
                matches = sorted(
                    candidate.resolve()
                    for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
                )
            except OSError as exc:
                errors.append(f"{supplied}: cannot traverse directory: {exc}")
                continue
            if not matches:
                errors.append(f"{supplied}: directory contains no supported H5/HDF5/H5AD files")
            resolved.extend(matches)
        elif path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                errors.append(f"{supplied}: unsupported suffix {path.suffix!r}")
            else:
                resolved.append(path.resolve())
        else:
            errors.append(f"{supplied}: input is neither a regular file nor a directory")
    deduplicated = sorted(set(resolved), key=str)
    return deduplicated, errors


def report_basename(paths: list[Path]) -> str:
    digest = hashlib.sha256("\0".join(str(path) for path in paths).encode("utf-8")).hexdigest()[:12]
    if len(paths) == 1:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", paths[0].stem).strip("._-") or "input"
        return f"{safe_stem}.{digest}.preflight"
    return f"processed_files_{len(paths)}.{digest}.preflight"


def markdown_report(report: dict[str, Any], max_items: int) -> str:
    lines = [
        "# Processed HDF5/H5AD structural preflight",
        "",
        f"- Tool: `processed_h5_preflight` {report['tool']['version']}",
        f"- Created (UTC): {report['provenance']['created_utc']}",
        f"- Files resolved: {len(report['files'])}",
        f"- Sampling bound: {report['sampling']['max_values_per_dataset']} values per eligible dataset",
        "- Safety: read-only HDF5 access; no AnnData load; no candidate matrix values sampled",
        "- Interpretation: field-name matches and matrix detections are structural candidates, not confirmed biological semantics",
        "",
    ]
    if report["input_resolution_errors"]:
        lines.extend(["## Input resolution errors", ""])
        lines.extend(f"- {error}" for error in report["input_resolution_errors"])
        lines.append("")
    for file_report in report["files"]:
        lines.extend([f"## `{file_report['path']}`", ""])
        if file_report["status"] != "ok":
            lines.extend([f"**Error:** {file_report['error']}", ""])
            continue
        facts = file_report["confirmed_facts"]
        candidates = file_report["structural_candidates"]
        lines.extend([
            "### Confirmed structural facts",
            "",
            f"- File size: {facts['size_bytes']:,} bytes; mtime: {facts['mtime_utc']}",
            f"- Objects: {facts['group_count_including_root']} groups (including root), {facts['dataset_count']} datasets",
            f"- Unchanged during inspection: {facts['unchanged_during_inspection']}",
            "",
            "### Matrix candidates",
            "",
        ])
        matrices = candidates["matrices"]
        if matrices:
            lines.extend(["| Path | Organization | Dimensions | Count semantics |", "|---|---|---:|---|"])
            for matrix in matrices[:max_items]:
                dims = " × ".join(map(str, matrix["dimensions"] or [])) or "unknown"
                lines.append(
                    f"| `{matrix['path']}` | {matrix['organization']} | {dims} | {matrix['count_evidence']} |"
                )
            if len(matrices) > max_items:
                lines.append(f"\n_{len(matrices) - max_items} additional matrix candidates are present in JSON._")
        else:
            lines.append("- None structurally identified.")
        lines.extend(["", "### Embedded metadata", ""])
        obs = candidates["observation_metadata_fields"]
        var = candidates["feature_annotation_fields"]
        lines.extend([
            f"- Observation fields ({len(obs)}): " + (", ".join(f"`{item['name']}`" for item in obs[:max_items]) or "none"),
            f"- Feature fields ({len(var)}): " + (", ".join(f"`{item['name']}`" for item in var[:max_items]) or "none"),
            "- Candidate barcode/index paths: " + (", ".join(
                f"`{path}`" for path in candidates["barcode_or_observation_index"]["candidate_paths"]
            ) or "none"),
            "- Candidate feature-ID paths: " + (", ".join(
                f"`{path}`" for path in candidates["feature_identifiers"]["candidate_paths"]
            ) or "none"),
            "",
            "### Candidate semantic fields (unconfirmed)",
            "",
        ])
        semantic_fields = candidates["semantic_fields"]
        if semantic_fields:
            lines.extend(["| Candidate role | Field | Evidence |", "|---|---|---|"])
            for item in semantic_fields[:max_items]:
                lines.append(f"| {item['role']} | `{item['field']}` | {item['evidence']} |")
            if len(semantic_fields) > max_items:
                lines.append(f"\n_{len(semantic_fields) - max_items} additional candidates are present in JSON._")
        else:
            lines.append("- None identified from field names or bounded examples.")
        lines.extend(["", "### Unknowns requiring source-level verification", ""])
        lines.extend(f"- {unknown}" for unknown in file_report["unknowns"])
        lines.extend(["", "### Object inventory preview", ""])
        for obj in file_report["objects"][:max_items]:
            if obj["kind"] == "dataset":
                lines.append(f"- `{obj['path']}` — dataset, shape={obj['shape']}, dtype={obj['dtype']}")
            else:
                lines.append(f"- `{obj['path']}` — group, children={obj['child_count']}")
        if len(file_report["objects"]) > max_items:
            lines.append(f"- … {len(file_report['objects']) - max_items} additional objects are fully inventoried in JSON.")
        lines.append("")
    lines.extend([
        "## Scope boundary",
        "",
        "This report performs structural preflight only. It does not establish biological field meanings, apply QC, filter data, normalize expression, determine evaluation eligibility, reconstruct sample mappings, or score a model.",
        "",
    ])
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    paths, resolution_errors = resolve_inputs(args.inputs)
    if not paths:
        raise PreflightError("no supported, accessible input files were resolved")
    file_reports: list[dict[str, Any]] = []
    had_error = bool(resolution_errors)
    for path in paths:
        try:
            file_reports.append(inspect_file(path, args.max_sample_values, args.max_string_chars))
        except PreflightError as exc:
            had_error = True
            file_reports.append({"status": "error", "path": str(path), "error": str(exc)})
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "processed_h5_preflight", "version": VERSION},
        "provenance": {
            "created_utc": utc_iso(),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "h5py_version": h5py.__version__,
            "command": shlex.join([sys.executable, *sys.argv]),
            "working_directory": str(Path.cwd()),
            "requested_inputs": [str(path) for path in args.inputs],
            "resolved_inputs": [str(path) for path in paths],
        },
        "sampling": {
            "max_values_per_dataset": args.max_sample_values,
            "max_characters_per_value": args.max_string_chars,
            "eligible_datasets_only": True,
            "candidate_matrix_values_sampled": False,
        },
        "scientific_scope": {
            "unit_of_analysis": "donor × cytokine × cell type",
            "cells_are_independent_biological_replicates": False,
            "biological_semantics_inferred": False,
        },
        "input_resolution_errors": resolution_errors,
        "files": file_reports,
        "global_safety": {
            "source_files_modified": False,
            "full_matrix_loaded": False,
            "qc_or_filtering_performed": False,
            "normalization_performed": False,
            "eligibility_analysis_performed": False,
            "model_scoring_performed": False,
        },
    }
    return report, 2 if had_error else 0


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, exit_code = build_report(args)
        resolved = [Path(path) for path in report["provenance"]["resolved_inputs"]]
        basename = report_basename(resolved)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / f"{basename}.json"
        markdown_path = args.output_dir / f"{basename}.md"
        atomic_write(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        atomic_write(markdown_path, markdown_report(report, args.max_markdown_items))
    except (PreflightError, OSError) as exc:
        print(f"processed_h5_preflight: error: {exc}", file=sys.stderr)
        return 2
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
