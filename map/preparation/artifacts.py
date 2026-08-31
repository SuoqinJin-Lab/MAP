from __future__ import annotations

import json
import mmap
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .._common.feedback import Feedback
from .._common.paths import DatasetPaths
from .networks import (
    UNIMOL_CHECKPOINT,
    UNIMOL_DICTIONARY,
    _drug_vocabulary,
    _prepare_expression_inputs,
    _prepare_graph_assets,
    build_unimol_tokens,
)
from .shared import (
    SHARED_ARTIFACTS,
    create_project,
    artifact_files,
    prepare_cell_metadata,
    prepare_hvg_expression,
    prepare_state_inputs,
    validate_shared_artifacts,
)


def _artifact_root(paths: DatasetPaths, artifact: str) -> Path:
    return paths.material_dir(artifact)


def _artifact_manifest(paths: DatasetPaths, artifact: str, directory: Path | None = None) -> tuple[Path, dict]:
    root = directory or _artifact_root(paths, artifact)
    manifest_file = root / "manifest.json"
    payload = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.is_file()
        else {"format": "map_artifact_v1", "artifact": str(artifact), "smiles": []}
    )
    if payload.get("artifact") not in {None, str(artifact)}:
        raise ValueError(f"Artifact directory belongs to {payload.get('artifact')!r}: {root}")
    return manifest_file, payload


def _write_artifact_manifest(
    paths: DatasetPaths, artifact: str, payload: dict, smiles: list[str], directory: Path | None = None
) -> Path:
    root = directory or _artifact_root(paths, artifact)
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    # Keep any algorithm-specific format marker while the path itself remains
    # an artifact directory rather than a method directory.
    payload_format = payload.get("format", "map_artifact_v1")
    payload.update({
        "format": payload_format,
        "artifact": str(artifact),
        "smiles": list(smiles),
        "consumers": list(payload.get("consumers", ())),
    })
    # Artifact manifests are intentionally owner-neutral.  Older caches may
    # still carry a model/directory marker; drop those markers when refreshed.
    payload.pop("model", None)
    payload.pop("directory", None)
    manifest_file = root / "manifest.json"
    manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_file


def _smiles(paths: DatasetPaths) -> list[str]:
    table = pq.read_table(
        paths.prepared / "conditions.parquet", columns=["canonical_smiles"]
    )
    return sorted({str(value) for value in table.column(0).to_pylist()})


def _fingerprints(smiles: list[str], *, features: bool) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=1024,
        atomInvariantsGenerator=(
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if features else None
        ),
    )
    output = np.zeros((len(smiles), 1024), dtype=np.float32)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid canonical SMILES: {value}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), output[index])
    return output


def _rdkit2d(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    try:
        from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
    except ImportError as error:
        raise ImportError(
            "RDKit2D artifact preparation requires descriptastorus"
        ) from error
    generator = MakeGenerator(("RDKit2D",))
    values = np.asarray([generator.process(value) for value in smiles], dtype=np.float64)
    values = values[:, 1:]
    values[~np.isfinite(values)] = 0.0
    standard_deviation = values.std(axis=0, ddof=1)
    keep = np.isfinite(standard_deviation) & (standard_deviation > 0.01)
    if not keep.any():
        raise RuntimeError("RDKit2D preprocessing removed every descriptor")
    values = values[:, keep]
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0, ddof=1)
    values = (values - mean) / standard_deviation
    columns = list(generator.GetColumns())[1:]
    names = [str(value) for value, selected in zip(columns, keep) if selected]
    return values.astype(np.float32), names


def _deg_masks(
    paths: DatasetPaths,
    directory: Path,
    control_group_means: dict[str, dict] | None = None,
    *,
    top_k: int,
    mask_mode: str = "official",
    overwrite: bool,
) -> tuple[dict[str, dict], list[Path]]:
    """Prepare CRISP's optional ``rank_genes_groups_by_cov`` DEG masks.

    This implements the selection semantics of
    ``ml4bio/CRISP/CRISP/utils.py`` on MAP's materialized HVG matrix: for each
    population/covariate, compare every perturbation group with that
    population's control cells using a Welch t-test and retain the genes with
    the largest absolute statistics.  The upstream default is ``n_genes=50``
    and ``rankby_abs=True``.  It is an approximation of the original AnnData
    path because MAP does not materialize the upstream ``adata.uns`` gene-name
    dictionary and restricts the candidate universe to its prepared HVGs.
    """
    del control_group_means
    mask_mode = str(mask_mode).casefold()
    if mask_mode not in {"official", "validation"}:
        raise ValueError("CRISP deg mask mode must be official or validation")
    if int(top_k) <= 0:
        raise ValueError("CRISP DEG mask top_k must be positive")
    shapes_file = paths.prepared / "materialized_shapes.json"
    if not shapes_file.is_file():
        raise FileNotFoundError("CRISP DEG masks require materialized_shapes.json")
    shapes = json.loads(shapes_file.read_text(encoding="utf-8"))
    payload: dict[str, dict] = {}
    outputs: list[Path] = []
    root = directory / "deg_masks"
    # Scan each population sequentially.  Random advanced indexing into the
    # multi-gigabyte HVG mmap is prohibitively slow on shared filesystems;
    # bounded chunks preserve the same sampled-condition semantics while
    # turning the read into a streaming pass.
    for population, shape in shapes.items():
        population = str(population)
        n_cells = int(shape["n_cells"])
        hvg_dim = int(shape["hvg_dim"])
        source = paths.prepared / population
        required = {
            "condition_ids": source / "condition_ids.npy",
            "condition_offsets": source / "condition_offsets.npy",
            "condition_rows": source / "condition_rows.npy",
            "control_rows": source / ("control_group_rows.npy" if (source / "control_group_rows.npy").is_file() else "control_plate_rows.npy"),
            "hvg": source / "hvg.float16.dat",
        }
        required_files = required.values() if mask_mode == "official" else (
            required["condition_ids"],
            required["condition_offsets"],
            required["condition_rows"],
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            # DEG autofocus is optional.  Keep legacy/compact preparations
            # usable for the other CRISP inputs; enabling the mask later will
            # fail with a targeted message in CRISPDataset.
            continue
        control_rows = (
            np.asarray(np.load(required["control_rows"]), dtype=np.int64)
            if mask_mode == "official"
            else None
        )
        condition_ids = np.asarray(
            np.load(required["condition_ids"]), dtype=np.int64
        )
        condition_offsets = np.asarray(
            np.load(required["condition_offsets"]), dtype=np.int64
        )
        condition_rows = np.load(required["condition_rows"], mmap_mode="r")
        hvg = (
            np.memmap(
                required["hvg"], dtype=np.float16, mode="r", shape=(n_cells, hvg_dim)
            )
            if mask_mode == "official"
            else None
        )
        if len(condition_offsets) != len(condition_ids) + 1:
            raise ValueError(f"Invalid condition offsets for CRISP {population}")
        target = root / population
        target.mkdir(parents=True, exist_ok=True)
        ids_target = target / "condition_ids.int64.npy"
        mask_target = target / "mask.bool.npy"
        metadata_target = target / "metadata.json"
        reusable = (
            not overwrite
            and ids_target.is_file()
            and mask_target.is_file()
            and metadata_target.is_file()
        )
        if reusable:
            metadata = json.loads(metadata_target.read_text(encoding="utf-8"))
            saved_ids = np.load(ids_target, mmap_mode="r")
            saved_mask = np.load(mask_target, mmap_mode="r")
            reusable = (
                metadata.get("format") == (
                    "crisp_rank_genes_groups_cov_mask_v2"
                    if mask_mode == "official"
                    else "crisp_validation_zero_mask_v1"
                )
                and metadata.get("top_k") == int(top_k)
                and metadata.get("hvg_dim") == hvg_dim
                and np.array_equal(saved_ids, condition_ids)
                and saved_mask.shape == (len(condition_ids), hvg_dim)
            )
        if not reusable:
            masks = np.zeros((len(condition_ids), hvg_dim), dtype=np.bool_)
            control_mean = control_var = None
            control_n = 0.0
            if mask_mode == "official":
                control_mean, control_var = _row_moments(hvg, control_rows)
                control_n = float(len(control_rows))
            for index in range(len(condition_ids)):
                rows = np.asarray(
                    condition_rows[
                        int(condition_offsets[index]): int(condition_offsets[index + 1])
                    ],
                    dtype=np.int64,
                )
                if not len(rows):
                    raise ValueError(
                        f"CRISP condition {condition_ids[index]} in {population} is empty"
                    )
                if mask_mode == "validation":
                    continue
                condition_mean, condition_var = _row_moments(hvg, rows)
                condition_n = float(len(rows))
                delta = condition_mean - control_mean
                denominator = np.sqrt(
                    condition_var / max(condition_n, 1.0)
                    + control_var / max(control_n, 1.0)
                )
                score = np.divide(
                    delta,
                    denominator,
                    out=np.zeros_like(delta, dtype=np.float64),
                    where=denominator > 0,
                )
                zero_denominator = (denominator == 0) & (delta != 0)
                score[zero_denominator] = np.sign(delta[zero_denominator]) * np.inf
                count = min(int(top_k), hvg_dim)
                selected = np.argsort(-np.abs(score), kind="mergesort")[:count]
                masks[index, selected] = True
            np.save(ids_target, condition_ids)
            np.save(mask_target, masks)
            metadata_target.write_text(
                json.dumps(
                    {
                        "format": (
                            "crisp_rank_genes_groups_cov_mask_v2"
                            if mask_mode == "official"
                            else "crisp_validation_zero_mask_v1"
                        ),
                        "top_k": int(top_k),
                        "n_genes": int(top_k),
                        "hvg_dim": hvg_dim,
                        "condition_count": len(condition_ids),
                        "population": population,
                        "selection": (
                            "Welch t-test against all population controls; "
                            "rank by absolute statistic"
                            if mask_mode == "official"
                            else "all zeros, matching MAP-validation collators"
                        ),
                        "gene_universe": "prepared_hvg",
                        "scope": (
                            "official_semantics_approximation"
                            if mask_mode == "official"
                            else "validation_contract"
                        ),
                        "upstream": "ml4bio/CRISP:rank_genes_groups_by_cov",
                        "upstream_method": (
                            "scanpy.rank_genes_groups(default method='t-test')"
                        ),
                        "upstream_rankby_abs": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        relative = target.relative_to(directory)
        payload[population] = {
            "directory": str(relative),
            "condition_count": len(condition_ids),
            "hvg_dim": hvg_dim,
            "top_k": int(top_k),
            "n_genes": int(top_k),
            "condition_ids_file": ids_target.name,
            "mask_file": mask_target.name,
            "metadata_file": metadata_target.name,
            "dtype": "bool",
        }
        outputs.extend((ids_target, mask_target, metadata_target))
    return payload, outputs


def _row_moments(hvg, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-gene mean and unbiased variance without materializing all rows."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("CRISP DEG masks require non-empty cell groups")
    # float32 is sufficient for ranking top-k Welch statistics and avoids
    # quadrupling the 16-bit memmap footprint during preparation.  Larger
    # chunks reduce Python/IO overhead for the mostly contiguous condition
    # row ranges while keeping peak temporary memory bounded.
    total = np.zeros(hvg.shape[1], dtype=np.float64)
    squared = np.zeros(hvg.shape[1], dtype=np.float64)
    chunk_size = 65536
    for start in range(0, len(rows), chunk_size):
        values = np.asarray(hvg[rows[start : start + chunk_size]], dtype=np.float32)
        total += values.sum(axis=0, dtype=np.float64)
        squared += np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float64)
    count = float(len(rows))
    mean = total / count
    if len(rows) > 1:
        variance = np.maximum(
            (squared - np.square(total) / count) / (count - 1.0), 0.0
        )
    else:
        variance = np.zeros_like(mean)
    return mean, variance


def _control_group_means(
    paths: DatasetPaths,
    directory: Path,
    *,
    overwrite: bool,
) -> tuple[dict[str, dict], list[Path]]:
    shapes_file = paths.prepared / "materialized_shapes.json"
    if not shapes_file.is_file():
        raise FileNotFoundError(
            "CRISP control means require materialized_shapes.json"
        )
    shapes = json.loads(shapes_file.read_text(encoding="utf-8"))
    payload: dict[str, dict] = {}
    outputs: list[Path] = []
    for population, shape in shapes.items():
        n_cells = int(shape["n_cells"])
        hvg_dim = int(shape["hvg_dim"])
        source = paths.prepared / str(population)
        required = {
            "group_ids": source / ("control_group_ids.npy" if (source / "control_group_ids.npy").is_file() else "control_plate_ids.npy"),
            "offsets": source / ("control_group_offsets.npy" if (source / "control_group_offsets.npy").is_file() else "control_plate_offsets.npy"),
            "rows": source / ("control_group_rows.npy" if (source / "control_group_rows.npy").is_file() else "control_plate_rows.npy"),
            "embeddings": source / "state_embeddings.float16.dat",
            "hvg": source / "hvg.float16.dat",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "CRISP control means require completed materialization and STATE "
                "embeddings; missing: " + ", ".join(missing)
            )
        output = directory / "control_group_means" / str(population)
        output.mkdir(parents=True, exist_ok=True)
        targets = {
            "group_ids": output / "group_ids.int64.npy",
            "embeddings": output / "embeddings.float16.npy",
            "hvg": output / "hvg.float16.npy",
        }
        group_ids = np.asarray(np.load(required["group_ids"]), dtype=np.int64)
        reusable = not overwrite and all(path.is_file() for path in targets.values())
        if reusable:
            saved_ids = np.load(targets["group_ids"], mmap_mode="r")
            saved_embeddings = np.load(targets["embeddings"], mmap_mode="r")
            saved_hvg = np.load(targets["hvg"], mmap_mode="r")
            reusable = (
                np.array_equal(saved_ids, group_ids)
                and saved_embeddings.shape == (len(group_ids), 2048)
                and saved_hvg.shape == (len(group_ids), hvg_dim)
            )
        if not reusable:
            offsets = np.asarray(np.load(required["offsets"]), dtype=np.int64)
            rows = np.load(required["rows"], mmap_mode="r")
            if len(offsets) != len(group_ids) + 1:
                raise ValueError(
                    f"Invalid CRISP control-group offsets for {population}"
                )
            embeddings = np.memmap(
                required["embeddings"],
                dtype=np.float16,
                mode="r",
                shape=(n_cells, 2048),
            )
            hvg = np.memmap(
                required["hvg"],
                dtype=np.float16,
                mode="r",
                shape=(n_cells, hvg_dim),
            )
            embedding_means = np.empty((len(group_ids), 2048), dtype=np.float16)
            hvg_means = np.empty((len(group_ids), hvg_dim), dtype=np.float16)
            for index in range(len(group_ids)):
                selected_rows = np.asarray(
                    rows[int(offsets[index]): int(offsets[index + 1])],
                    dtype=np.int64,
                ).copy()
                if not len(selected_rows):
                    raise ValueError(
                        f"CRISP control group {group_ids[index]} in {population} is empty"
                    )
                # Control rows are stored as mostly contiguous plate ranges.
                # Coalesce runs before reading so preparation does not issue
                # thousands of random 4-KiB mmap requests on Lustre.
                selected_rows.sort()
                boundaries = np.flatnonzero(np.diff(selected_rows) != 1) + 1
                runs = np.split(selected_rows, boundaries)
                total_embedding = np.zeros(embeddings.shape[1], dtype=np.float64)
                total_hvg = np.zeros(hvg.shape[1], dtype=np.float64)
                for run in runs:
                    if run.size == 0:
                        continue
                    values_embedding = np.asarray(embeddings[run[0] : run[-1] + 1], dtype=np.float32)
                    values_hvg = np.asarray(hvg[run[0] : run[-1] + 1], dtype=np.float32)
                    total_embedding += values_embedding.sum(axis=0, dtype=np.float64)
                    total_hvg += values_hvg.sum(axis=0, dtype=np.float64)
                embedding_means[index] = total_embedding / float(len(selected_rows))
                hvg_means[index] = total_hvg / float(len(selected_rows))
            np.save(targets["group_ids"], group_ids)
            np.save(targets["embeddings"], embedding_means)
            np.save(targets["hvg"], hvg_means)
        relative = output.relative_to(directory)
        payload[str(population)] = {
            "directory": str(relative),
            "groups": len(group_ids),
            "embedding_dim": 2048,
            "hvg_dim": hvg_dim,
            "dtype": "float16",
        }
        outputs.extend(targets.values())
    return payload, outputs


def _moa_embedding(
    paths: DatasetPaths,
    smiles: list[str],
    *,
    split_file: str | Path | None = None,
    max_cells_per_condition: int = 32,
    n_components: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Build the paper-style CMonge MoA context from training responses.

    The upstream implementation embeds perturbation conditions by applying
    metric MDS to distances between their expression distributions.  MAP's
    file-backed contract stores millions of cells, so we use the equivalent
    stable approximation: a per-drug mean response (sampled per condition)
    followed by a centered SVD.  Only rows in the requested training split
    contribute, preventing unseen cell-line/drug responses from leaking into
    the context.  The resulting matrix is one vector per canonical SMILES.
    """
    if int(max_cells_per_condition) <= 0 or int(n_components) <= 0:
        raise ValueError("max_cells_per_condition and n_components must be positive")
    table = pq.read_table(paths.prepared / "conditions.parquet").to_pandas()
    table = table.set_index("condition_id")
    if "population" not in table.columns:
        table["population"] = table["cell_line"].astype(str)
    split_payload = None
    if split_file is not None:
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = paths.prepared / split_path
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    selected = set(int(v) for v in (split_payload or {}).get("train", table.index.tolist()))
    smile_to_index = {str(value): i for i, value in enumerate(smiles)}
    sums = np.zeros((len(smiles), 2000), dtype=np.float64)
    counts = np.zeros(len(smiles), dtype=np.int64)
    shapes = json.loads((paths.prepared / "materialized_shapes.json").read_text(encoding="utf-8"))
    for population, shape in shapes.items():
        population = str(population)
        base = paths.prepared / population
        n_cells = int(shape["n_cells"])
        hvg_dim = int(shape.get("hvg_dim", 2000))
        if hvg_dim != 2000:
            raise ValueError(f"CMonge MoA requires 2000 HVGs; {population} has {hvg_dim}")
        hvg = np.memmap(base / "hvg.float16.dat", dtype=np.float16, mode="r", shape=(n_cells, hvg_dim))
        row_condition = np.memmap(
            base / "row_condition.int32.dat", dtype=np.int32, mode="r", shape=(n_cells,)
        )
        expected_ids = np.asarray(
            table.index[table["population"].astype(str) == population],
            dtype=np.int64,
        )
        population_condition_ids = np.asarray(np.load(base / "condition_ids.npy"), dtype=np.int64)
        global_fraction = np.isin(population_condition_ids, expected_ids).mean() if len(population_condition_ids) else 1.0
        local_to_global = None if global_fraction >= 0.95 else expected_ids
        train_rows = (split_payload or {}).get("train_rows", {}).get(population)
        allowed_rows = None
        if train_rows is not None:
            allowed_rows = np.zeros(n_cells, dtype=np.bool_)
            allowed_rows[np.asarray(train_rows, dtype=np.int64)] = True
        selected_population = {
            int(condition_id): smile_to_index.get(str(table.loc[int(condition_id)]["canonical_smiles"]))
            for condition_id in selected
            if int(condition_id) in table.index and str(table.loc[int(condition_id)]["population"]) == population
        }
        selected_population = {k: v for k, v in selected_population.items() if v is not None}
        seen = {condition_id: 0 for condition_id in selected_population}
        # Keep the temporary dense block small (HVG matrices are 2,000 genes
        # wide; 4,096 rows is ~32 MiB in float16) for login-node probes and
        # constrained batch jobs.
        chunk_size = 512
        for start in range(0, n_cells, chunk_size):
            stop = min(start + chunk_size, n_cells)
            labels = np.asarray(row_condition[start:stop], dtype=np.int64)
            if local_to_global is not None:
                valid = (labels >= 0) & (labels < len(local_to_global))
                labels = labels.copy()
                labels[valid] = local_to_global[labels[valid]]
            mask = np.isin(labels, np.fromiter(selected_population, dtype=np.int64))
            if allowed_rows is not None:
                mask &= allowed_rows[start:stop]
            if not mask.any():
                continue
            local = np.flatnonzero(mask)
            values = np.asarray(hvg[start:stop], dtype=np.float32)
            for condition_id in np.unique(labels[local]):
                if seen[int(condition_id)] >= int(max_cells_per_condition):
                    continue
                chosen = local[labels[local] == int(condition_id)]
                remaining = int(max_cells_per_condition) - seen[int(condition_id)]
                chosen = chosen[:remaining]
                if len(chosen):
                    sums[selected_population[int(condition_id)]] += values[chosen].sum(axis=0, dtype=np.float64)
                    counts[selected_population[int(condition_id)]] += len(chosen)
                    seen[int(condition_id)] += len(chosen)
            # Drop pages already consumed so a sequential scan of a 10+ GiB
            # mmap does not accumulate in the process RSS on shared nodes.
            try:
                hvg._mmap.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            if seen and all(value >= int(max_cells_per_condition) for value in seen.values()):
                break
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        raise ValueError(f"No training response available for {len(missing)} drugs; first indices={missing[:8].tolist()}")
    response = sums / counts[:, None]
    response -= response.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(response, full_matrices=False)
    components = min(int(n_components), right.shape[0])
    embedding = response @ right[:components].T
    scale = embedding.std(axis=0, keepdims=True)
    embedding = np.divide(embedding, scale, out=np.zeros_like(embedding), where=scale > 1e-8)
    if components < int(n_components):
        embedding = np.pad(embedding, ((0, 0), (0, int(n_components) - components)))
    metadata = {
        "method": "training-response SVD approximation of CMonge ModeOfActionEmbedding",
        "n_components": int(n_components),
        "max_cells_per_condition": int(max_cells_per_condition),
        "training_conditions": int(len(selected)),
        "drugs": int(len(smiles)),
        "seed": int(seed),
        "split_file": str(split_file) if split_file is not None else None,
    }
    return embedding.astype(np.float32), metadata


def prepare_moa_features(
    paths: DatasetPaths,
    *,
    split_file: str | Path | None = None,
    max_cells_per_condition: int = 32,
    n_components: int = 10,
    seed: int = 42,
    overwrite: bool = False,
):
    """Prepare the split-dependent MoA drug representation."""
    smiles = _smiles(paths)
    from ..train.splits import resolve_split

    split_path, split = resolve_split(paths, "unseen_combination", split_file)
    directory = paths.split_material_dir(split["split_id"], "drug_moa")
    target = directory / "moa.float32.npy"
    metadata = directory / "moa_manifest.json"
    if overwrite or not target.is_file() or not metadata.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        matrix, info = _moa_embedding(
            paths, smiles, split_file=split_path,
            max_cells_per_condition=max_cells_per_condition,
            n_components=n_components, seed=seed,
        )
        np.save(target, matrix)
        metadata.write_text(json.dumps({
            "artifact": "drug_moa",
            "representation": "MoA response embedding",
            "consumers": ["cmonge"], "smiles": smiles,
            "shape": list(matrix.shape), **info,
        }, indent=2), encoding="utf-8")
    payload = {
        "artifact": "drug_moa", "hvg_dim": 2000,
        "drug_representation": "moa", "representation": "CMonge MoA response embedding",
        "matrix": target.name, "smiles": smiles,
        "shape": list(np.load(target, mmap_mode="r").shape),
        "source": "training perturbation responses (paper ModeOfActionEmbedding contract)",
    }
    return _register_artifact(
        paths, "drug_moa", payload, outputs=[target, metadata], smiles=smiles,
        directory=directory,
    )


def _register_artifact(
    paths: DatasetPaths,
    artifact: str,
    payload: dict,
    *,
    outputs: list[Path],
    smiles: list[str] | None = None,
    stage: str | None = None,
    directory: Path | None = None,
):
    root = directory or _artifact_root(paths, artifact)
    _, existing = _artifact_manifest(paths, artifact, root)
    current_smiles = list(smiles or _smiles(paths))
    existing_smiles = existing.get("smiles") or []
    if existing_smiles and existing_smiles != current_smiles:
        raise RuntimeError("Existing artifact uses a different drug vocabulary")
    manifest_file = _write_artifact_manifest(
        paths, artifact, {**existing, **payload}, current_smiles, root
    )
    result_outputs = [*outputs, manifest_file]
    return Feedback(root, stage or f"prepare_{artifact}").finish(
        {
            "artifact": str(artifact),
            "consumers": list(payload.get("consumers", ())),
            "output": str(root),
        },
        result_outputs,
    )


def prepare_ecfp4_features(
    paths: DatasetPaths, *, overwrite: bool = False
):
    """Materialize ECFP4-1024 features consumed by chemCPA."""
    smiles = _smiles(paths)
    directory = _artifact_root(paths, "drug_ecfp4")
    target = directory / "ecfp4_1024.float32.npy"
    if overwrite or not target.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        np.save(target, _fingerprints(smiles, features=False))
    payload = {
        "artifact": "drug_ecfp4",
        "representation": "ECFP4-1024", "matrix": target.name,
        "smiles": smiles, "shape": list(np.load(target, mmap_mode="r").shape),
        "consumers": ["chemcpa"],
    }
    return _register_artifact(paths, "drug_ecfp4", payload, outputs=[target], smiles=smiles)


def prepare_fcfp4_features(
    paths: DatasetPaths, *, overwrite: bool = False
):
    """Materialize FCFP4-1024 features consumed by PRnet."""
    smiles = _smiles(paths)
    directory = _artifact_root(paths, "drug_fcfp4")
    target = directory / "fcfp4_1024.float32.npy"
    if overwrite or not target.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        np.save(target, _fingerprints(smiles, features=True))
    payload = {
        "artifact": "drug_fcfp4",
        "representation": "FCFP4-1024", "matrix": target.name,
        "smiles": smiles, "shape": list(np.load(target, mmap_mode="r").shape),
        "consumers": ["prnet"],
    }
    return _register_artifact(paths, "drug_fcfp4", payload, outputs=[target], smiles=smiles)


def prepare_control_means(
    paths: DatasetPaths, *, overwrite: bool = False
):
    """Prepare control-group pseudobulk means."""
    directory = _artifact_root(paths, "control_means")
    means, outputs = _control_group_means(paths, directory, overwrite=overwrite)
    _, payload = _artifact_manifest(paths, "control_means")
    payload["control_group_means"] = means
    payload["consumers"] = ["crisp"]
    return _register_artifact(paths, "control_means", payload, outputs=outputs)


def prepare_deg_masks(
    paths: DatasetPaths, *, top_k: int = 50, mask_mode: str = "official",
    overwrite: bool = False,
):
    """Prepare condition-level differential-expression masks."""
    directory = _artifact_root(paths, "deg_masks")
    masks, outputs = _deg_masks(
        paths, directory, top_k=int(top_k), mask_mode=mask_mode, overwrite=overwrite
    )
    _, payload = _artifact_manifest(paths, "deg_masks")
    payload.update({
        "deg_masks": masks, "deg_mask_populations": sorted(masks),
        "deg_mask_mode": str(mask_mode).casefold(), "deg_top_k": int(top_k),
    })
    payload["consumers"] = ["crisp"]
    return _register_artifact(paths, "deg_masks", payload, outputs=outputs)


def prepare_molecular_descriptors(
    paths: DatasetPaths, *, overwrite: bool = False
):
    """Prepare standardized RDKit2D molecular descriptors."""
    smiles = _smiles(paths)
    directory = _artifact_root(paths, "drug_rdkit2d")
    target = directory / "rdkit2d.float32.npy"
    metadata = directory / "rdkit2d_manifest.json"
    if overwrite or not target.is_file() or not metadata.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        matrix, descriptors = _rdkit2d(smiles)
        np.save(target, matrix)
        metadata.write_text(json.dumps({
            "artifact": "drug_rdkit2d",
            "representation": "standardized RDKit2D descriptors",
            "consumers": ["cmonge"], "smiles": smiles,
            "shape": list(matrix.shape), "descriptors": descriptors,
        }, indent=2), encoding="utf-8")
    payload = {
        "artifact": "drug_rdkit2d",
        "hvg_dim": 2000,
        "representation": "standardized RDKit molecular descriptors",
        "matrix": target.name, "smiles": smiles,
        "shape": list(np.load(target, mmap_mode="r").shape),
        "consumers": ["crisp", "cmonge"],
    }
    return _register_artifact(paths, "drug_rdkit2d", payload, outputs=[target, metadata], smiles=smiles)


def prepare_unimol_tokens(
    paths: DatasetPaths, *, overwrite: bool = False,
    reference_dir: str | Path | None = None, unimol_dir: str | Path | None = None,
    max_atoms: int = 122, batch_size: int = 32,
):
    """Prepare UniMol drug representations as an independent artifact."""
    reference_dir = Path(reference_dir or paths.frozen_models.parent / "reference_data" / "xpert")
    unimol_dir = Path(unimol_dir or paths.frozen_models / "unimol")
    checkpoint = unimol_dir / UNIMOL_CHECKPOINT
    dictionary = unimol_dir / UNIMOL_DICTIONARY
    if not checkpoint.is_file() or not dictionary.is_file():
        raise FileNotFoundError("UniMol assets are missing")
    smiles, drug_names = _drug_vocabulary(paths)
    directory = _artifact_root(paths, "drug_unimol")
    target = directory / "drug_unimol.float16.npy"
    if overwrite or not target.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        tokens = build_unimol_tokens(smiles, checkpoint, dictionary, max_atoms=max_atoms, batch_size=batch_size)
        np.save(target, tokens.astype(np.float16))
    vocabulary = directory / "drug_vocabulary.json"
    vocabulary.write_text(json.dumps({"smiles": smiles, "names": drug_names}, indent=2), encoding="utf-8")
    _, payload = _artifact_manifest(paths, "drug_unimol")
    payload.update({"drug_unimol_file": target.name, "drug_vocabulary_file": vocabulary.name})
    payload["consumers"] = ["xpert"]
    return _register_artifact(paths, "drug_unimol", payload, outputs=[target, vocabulary], smiles=smiles)


def prepare_expression_bins(
    paths: DatasetPaths, *, input_formats: Iterable[str] = ("official",),
    expression_bins: int = 128, expression_min: float | None = None,
    expression_max: float | None = None, expression_sample_cells: int = 1024,
    overwrite: bool = False,
):
    """Prepare the binned-expression artifact independently."""
    directory = _artifact_root(paths, "expression_bins")
    _, existing = _artifact_manifest(paths, "expression_bins")
    hvg_count = int(existing.get("gene_count", 0))
    if not hvg_count:
        hvg_count = len(json.loads((paths.prepared / "hvg.json").read_text(encoding="utf-8"))["gene_symbols"])
    payload, outputs = _prepare_expression_inputs(
        paths, directory, hvg_count, input_modes=tuple(input_formats),
        expression_bins=int(expression_bins), expression_min=expression_min,
        expression_max=expression_max, sample_cells_per_population=int(expression_sample_cells),
        overwrite=overwrite,
    )
    model_payload = dict(existing or {"artifact": "expression_bins"})
    model_payload["expression"] = payload
    model_payload["expression_bins"] = int(expression_bins)
    model_payload["expression_bin_boundaries_file"] = payload["bin_boundaries_file"]
    model_payload["consumers"] = ["xpert"]
    return _register_artifact(paths, "expression_bins", model_payload, outputs=outputs)


def prepare_graph_assets(
    paths: DatasetPaths, *, overwrite: bool = False, **kwargs
):
    """Prepare graph assets and embeddings.

    The graph bundle includes the UniMol-derived drug node initialization it
    requires. Expression bins are prepared independently by
    :func:`prepare_expression_bins`.
    """
    payload, outputs = _prepare_graph_assets(
        paths, _artifact_root(paths, "graph_assets"),
        overwrite=overwrite, input_formats=("official",),
        include_expression=False, **kwargs
    )
    payload["consumers"] = ["xpert"]
    return _register_artifact(
        paths,
        "graph_assets",
        payload,
        outputs=outputs,
        smiles=list(payload.get("smiles", ())),
    )


__all__ = [
    "SHARED_ARTIFACTS",
    "create_project",
    "artifact_files",
    "prepare_cell_metadata",
    "prepare_state_inputs",
    "prepare_hvg_expression",
    "validate_shared_artifacts",
    "prepare_ecfp4_features",
    "prepare_fcfp4_features",
    "prepare_control_means",
    "prepare_deg_masks",
    "prepare_molecular_descriptors",
    "prepare_moa_features",
    "prepare_unimol_tokens",
    "prepare_graph_assets",
    "prepare_expression_bins",
]
