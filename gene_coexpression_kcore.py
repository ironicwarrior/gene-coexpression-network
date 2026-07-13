#!/usr/bin/env python3
"""Paper-style gene co-expression network with K-core decomposition.

This script reproduces the *type* of network shown in Zhang et al. (2011):
Pearson gene-gene correlations are thresholded to create an undirected graph,
then node degree and K-core values are calculated and visualized.

It is deliberately reusable. The default run downloads the public GEO series
GSE28195. A custom CSV/TSV expression matrix can be supplied instead.

Expected custom input (default orientation):
    gene,sample_1,sample_2,...
    GeneA,8.2,8.7,...
    GeneB,4.1,5.0,...

The values should already be normalized. Raw RNA-seq counts should be transformed
with a variance-stabilizing method, log2(CPM + 1), or log2(TPM + 1) before a
correlation network is interpreted.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


@dataclass
class Config:
    # Input
    data_source: Literal["geo", "csv"] = "geo"
    geo_accession: str = "GSE28195"
    input_path: Optional[str] = None
    input_orientation: Literal["genes_by_samples", "samples_by_genes"] = "genes_by_samples"
    gene_column: Optional[str] = None
    delimiter: Optional[str] = None
    output_dir: str = "coexpression_results"

    # Preprocessing
    log2_transform: Literal["auto", "yes", "no"] = "auto"
    aggregate_duplicate_genes: Literal["median", "mean", "max_variance"] = "median"
    min_nonmissing_samples: int = 4
    min_gene_variance: float = 1e-12

    # Gene selection
    gene_selection: Literal["paper_s8_or_top_variance", "top_variance", "all", "gene_list"] = (
        "paper_s8_or_top_variance"
    )
    top_n_genes: int = 265
    gene_list_path: Optional[str] = None
    paper_s8_url: str = (
        "https://journals.plos.org/plosone/article/file?"
        "type=supplementary&id=info:doi/10.1371/journal.pone.0024680.s009"
    )

    # Correlation network
    correlation_method: Literal["pearson", "spearman"] = "pearson"
    correlation_mode: Literal["absolute", "positive", "negative"] = "absolute"
    threshold_mode: Literal["study_like", "fixed"] = "study_like"
    correlation_threshold: float = 0.90
    threshold_min: float = 0.75
    threshold_max: float = 0.995
    threshold_step: float = 0.005
    target_nodes: int = 156
    target_components: int = 3
    target_max_core: int = 9
    min_degree: int = 1
    keep_largest_components: int = 3

    # Plot/output
    label_mode: Literal["all", "top_degree", "none"] = "all"
    top_labels: int = 60
    highlight_genes: tuple[str, ...] = ()
    random_seed: int = 42
    figure_width: float = 22.0
    figure_height: float = 11.0
    figure_dpi: int = 300


INVALID_GENE_VALUES = {
    "",
    "NA",
    "N/A",
    "NAN",
    "NULL",
    "NONE",
    "---",
    "--",
}


KCORE_COLORS = {
    0: "#eeeeee",
    1: "#f4cccc",  # pale pink
    2: "#b7b7b7",  # gray
    3: "#29b6d8",  # cyan
    4: "#9b36d0",  # purple
    5: "#35d34a",  # bright green
    6: "#f50057",  # magenta
    7: "#006400",  # dark green
    8: "#a6a600",  # olive
    9: "#1039d8",  # dark blue
}


def _print(message: str) -> None:
    print(message, flush=True)


def _sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "sample"


def _normalize_gene_symbol(value: object) -> str:
    if pd.isna(value):
        return ""
    symbol = str(value).strip()
    if symbol.upper() in INVALID_GENE_VALUES:
        return ""

    # Many platform annotations put multiple mappings in one cell.
    symbol = re.split(r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*", symbol)[0].strip()
    if symbol.upper() in INVALID_GENE_VALUES:
        return ""
    return symbol


def _find_column(columns: Iterable[str], preferred: Iterable[str], contains: Iterable[str] = ()) -> Optional[str]:
    columns = list(columns)
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}
    for candidate in preferred:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    for token in contains:
        token_norm = re.sub(r"[^a-z0-9]", "", token.lower())
        for key, original in normalized.items():
            if token_norm in key:
                return original
    return None


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.apply(pd.to_numeric, errors="coerce")
    converted = converted.replace([np.inf, -np.inf], np.nan)
    return converted


def load_custom_expression(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.input_path:
        raise ValueError("input_path is required when data_source='csv'.")

    path = config.input_path
    if config.delimiter is not None:
        sep = config.delimiter
    elif str(path).lower().endswith((".tsv", ".txt", ".txt.gz", ".tsv.gz")):
        sep = "\t"
    else:
        sep = ","

    raw = pd.read_csv(path, sep=sep, compression="infer")
    if raw.empty:
        raise ValueError("The input table is empty.")

    if config.input_orientation == "samples_by_genes":
        sample_id_col = config.gene_column or raw.columns[0]
        if sample_id_col not in raw.columns:
            raise ValueError(f"Sample ID column '{sample_id_col}' is not present.")
        raw = raw.set_index(sample_id_col).T.reset_index().rename(columns={"index": "gene"})
        gene_col = "gene"
    else:
        gene_col = config.gene_column or raw.columns[0]

    if gene_col not in raw.columns:
        raise ValueError(f"Gene column '{gene_col}' is not present in the input table.")

    genes = raw[gene_col].map(_normalize_gene_symbol)
    expr = _numeric_frame(raw.drop(columns=[gene_col]))
    expr.index = genes
    expr = expr.loc[expr.index != ""]
    expr = expr.loc[:, expr.notna().any(axis=0)]

    sample_info = pd.DataFrame(
        {
            "sample_id": expr.columns.astype(str),
            "sample_title": expr.columns.astype(str),
        }
    ).set_index("sample_id")
    return expr, sample_info


def _get_geo_platform(gse: object, geo_dir: Path) -> object:
    import GEOparse  # type: ignore

    if getattr(gse, "gpls", None):
        return next(iter(gse.gpls.values()))

    first_gsm = next(iter(gse.gsms.values()))
    platform_ids = first_gsm.metadata.get("platform_id", [])
    if not platform_ids:
        raise RuntimeError("No platform ID was found in the GEO series.")
    gpl_id = platform_ids[0]
    _print(f"Downloading platform annotation {gpl_id} ...")
    return GEOparse.get_GEO(geo=gpl_id, destdir=str(geo_dir), annotate_gpl=True, silent=False)


def load_geo_expression(config: Config, work_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import GEOparse  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "GEOparse is required for GEO input. Install it with: pip install GEOparse"
        ) from exc

    geo_dir = work_dir / "geo_cache"
    geo_dir.mkdir(parents=True, exist_ok=True)
    _print(f"Downloading/reading GEO series {config.geo_accession} ...")
    gse = GEOparse.get_GEO(
        geo=config.geo_accession,
        destdir=str(geo_dir),
        annotate_gpl=True,
        silent=False,
    )

    expr_probe = gse.pivot_samples("VALUE")
    expr_probe = _numeric_frame(expr_probe)
    if expr_probe.empty:
        raise RuntimeError("GEOparse returned an empty expression matrix.")

    sample_titles: list[str] = []
    sample_rows: list[dict[str, str]] = []
    used_titles: dict[str, int] = {}
    for gsm_id in expr_probe.columns.astype(str):
        gsm = gse.gsms.get(gsm_id)
        title = gsm.metadata.get("title", [gsm_id])[0] if gsm is not None else gsm_id
        title = str(title)
        safe = _sanitize_filename(title)
        count = used_titles.get(safe, 0)
        used_titles[safe] = count + 1
        unique_title = safe if count == 0 else f"{safe}_{count + 1}"
        sample_titles.append(unique_title)
        sample_rows.append(
            {
                "sample_id": gsm_id,
                "sample_title": title,
                "analysis_column": unique_title,
            }
        )
    expr_probe.columns = sample_titles
    sample_info = pd.DataFrame(sample_rows).set_index("sample_id")

    gpl = _get_geo_platform(gse, geo_dir)
    annotation = gpl.table.copy()
    if annotation.empty:
        warnings.warn("The platform annotation table is empty; probe IDs will be used as node labels.")
        expr_probe.index = expr_probe.index.astype(str)
        return expr_probe, sample_info

    probe_col = _find_column(
        annotation.columns,
        preferred=("ID", "ID_REF", "Probe_Id", "Probe ID", "Array_Address_Id"),
        contains=("probeid", "arrayaddress"),
    )
    symbol_col = _find_column(
        annotation.columns,
        preferred=("Symbol", "Gene Symbol", "GENE_SYMBOL", "ILMN_Gene", "Gene"),
        contains=("symbol", "ilmngene"),
    )

    if probe_col is None or symbol_col is None:
        warnings.warn(
            "A probe-to-gene-symbol mapping could not be found automatically. "
            "Probe IDs will be used as node labels."
        )
        expr_probe.index = expr_probe.index.astype(str)
        return expr_probe, sample_info

    mapping = annotation[[probe_col, symbol_col]].copy()
    mapping[probe_col] = mapping[probe_col].astype(str)
    mapping["gene_symbol"] = mapping[symbol_col].map(_normalize_gene_symbol)
    mapping = mapping.loc[mapping["gene_symbol"] != "", [probe_col, "gene_symbol"]]
    mapping = mapping.drop_duplicates(subset=[probe_col], keep="first")

    expr_with_gene = expr_probe.copy()
    expr_with_gene.index = expr_with_gene.index.astype(str)
    expr_with_gene = expr_with_gene.join(mapping.set_index(probe_col), how="left")
    expr_with_gene = expr_with_gene.dropna(subset=["gene_symbol"])

    numeric_cols = [c for c in expr_with_gene.columns if c != "gene_symbol"]
    if config.aggregate_duplicate_genes == "mean":
        expr_gene = expr_with_gene.groupby("gene_symbol", sort=False)[numeric_cols].mean()
    elif config.aggregate_duplicate_genes == "max_variance":
        temp = expr_with_gene.copy()
        temp["_row_variance"] = temp[numeric_cols].var(axis=1, ddof=1)
        selected_idx = temp.groupby("gene_symbol")["_row_variance"].idxmax()
        expr_gene = temp.loc[selected_idx].set_index("gene_symbol")[numeric_cols]
    else:
        expr_gene = expr_with_gene.groupby("gene_symbol", sort=False)[numeric_cols].median()

    return expr_gene, sample_info


def maybe_log2_transform(expr: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, bool]:
    values = expr.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite numeric expression values were found.")

    if mode == "yes":
        do_transform = True
    elif mode == "no":
        do_transform = False
    else:
        q01, q50, q99 = np.nanquantile(finite, [0.01, 0.50, 0.99])
        # Common heuristic used for processed microarray/RNA expression tables.
        do_transform = bool(q99 > 100 or (q99 - q01) > 50 or q50 > 20)

    if not do_transform:
        return expr.copy(), False

    min_value = float(np.nanmin(finite))
    if min_value < 0:
        shifted = expr - min_value
        transformed = np.log2(shifted + 1.0)
    else:
        transformed = np.log2(expr.clip(lower=0) + 1.0)
    return transformed, True


def clean_expression(expr: pd.DataFrame, config: Config) -> pd.DataFrame:
    expr = _numeric_frame(expr)
    expr.index = pd.Index([_normalize_gene_symbol(x) for x in expr.index], name="gene")
    expr = expr.loc[expr.index != ""]
    expr = expr.loc[:, expr.notna().any(axis=0)]

    min_present = min(config.min_nonmissing_samples, expr.shape[1])
    expr = expr.loc[expr.notna().sum(axis=1) >= min_present]
    expr = expr.apply(lambda row: row.fillna(row.median()), axis=1)

    variances = expr.var(axis=1, ddof=1)
    expr = expr.loc[variances > config.min_gene_variance]
    if expr.empty:
        raise ValueError("No genes remained after missing-value and variance filtering.")
    return expr


def _download_paper_s8_gene_list(url: str, available_genes: pd.Index, cache_dir: Path) -> list[str]:
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "paper_table_s8.xls"
    if not target.exists() or target.stat().st_size < 1000:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)

    available_lookup = {str(g).upper(): str(g) for g in available_genes}
    found: set[str] = set()
    workbook = pd.ExcelFile(target)
    token_pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9._-]{1,30}\b")
    for sheet in workbook.sheet_names:
        table = pd.read_excel(target, sheet_name=sheet, header=None)
        for value in table.to_numpy().ravel():
            if pd.isna(value):
                continue
            for token in token_pattern.findall(str(value)):
                match = available_lookup.get(token.upper())
                if match is not None:
                    found.add(match)
    return sorted(found)


def select_genes(expr: pd.DataFrame, config: Config, work_dir: Path) -> tuple[pd.DataFrame, str]:
    mode = config.gene_selection

    if mode == "paper_s8_or_top_variance" and config.geo_accession.upper() == "GSE28195":
        try:
            paper_genes = _download_paper_s8_gene_list(
                config.paper_s8_url, expr.index, work_dir / "paper_supplement"
            )
            if len(paper_genes) >= 50:
                _print(f"Using {len(paper_genes)} genes recovered from the paper's Table S8.")
                return expr.loc[paper_genes], "paper_table_s8"
            warnings.warn(
                f"Only {len(paper_genes)} Table S8 genes matched the expression matrix; "
                "falling back to top-variable genes."
            )
        except Exception as exc:  # network/download/format failures should not stop the run
            warnings.warn(f"Could not use Table S8 ({exc}); falling back to top-variable genes.")
        mode = "top_variance"

    if mode == "gene_list":
        if not config.gene_list_path:
            raise ValueError("gene_list_path is required when gene_selection='gene_list'.")
        gene_table = pd.read_csv(config.gene_list_path, header=None)
        requested = {_normalize_gene_symbol(x).upper() for x in gene_table.iloc[:, 0]}
        selected = [g for g in expr.index if str(g).upper() in requested]
        if len(selected) < 2:
            raise ValueError("Fewer than two requested genes were found in the expression matrix.")
        return expr.loc[selected], "custom_gene_list"

    if mode == "all":
        return expr.copy(), "all_genes"

    n = min(config.top_n_genes, expr.shape[0])
    variances = expr.var(axis=1, ddof=1).sort_values(ascending=False)
    selected = variances.head(n).index
    return expr.loc[selected], f"top_{n}_variable_genes"


def compute_correlation(expr: pd.DataFrame, method: str) -> pd.DataFrame:
    if expr.shape[1] < 3:
        raise ValueError("At least three samples are required to compute gene-gene correlations.")
    if method == "spearman":
        ranked = expr.rank(axis=1, method="average")
        corr_values = np.corrcoef(ranked.to_numpy(dtype=float))
    else:
        corr_values = np.corrcoef(expr.to_numpy(dtype=float))
    corr_values = np.clip(corr_values, -1.0, 1.0)
    return pd.DataFrame(corr_values, index=expr.index, columns=expr.index)


def _edge_mask(values: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    if mode == "positive":
        return values >= threshold
    if mode == "negative":
        return values <= -threshold
    return np.abs(values) >= threshold


def build_graph(
    corr: pd.DataFrame,
    threshold: float,
    mode: str,
    min_degree: int = 1,
    keep_largest_components: int = 0,
) -> nx.Graph:
    genes = corr.index.to_numpy(dtype=str)
    matrix = corr.to_numpy(dtype=float)
    upper_i, upper_j = np.triu_indices_from(matrix, k=1)
    values = matrix[upper_i, upper_j]
    mask = _edge_mask(values, threshold, mode) & np.isfinite(values)

    graph = nx.Graph()
    for i, j, value in zip(upper_i[mask], upper_j[mask], values[mask]):
        graph.add_edge(
            genes[i],
            genes[j],
            correlation=float(value),
            abs_correlation=float(abs(value)),
            weight=float(abs(value)),
            sign="positive" if value >= 0 else "negative",
        )

    if min_degree > 1 and graph.number_of_nodes() > 0:
        changed = True
        while changed and graph.number_of_nodes() > 0:
            remove = [node for node, degree in graph.degree() if degree < min_degree]
            changed = bool(remove)
            graph.remove_nodes_from(remove)

    if keep_largest_components > 0 and graph.number_of_nodes() > 0:
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        keep_nodes = set().union(*components[:keep_largest_components])
        graph = graph.subgraph(keep_nodes).copy()

    return graph


def summarize_graph(graph: nx.Graph) -> dict[str, float | int]:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    if n_nodes == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "components": 0,
            "density": 0.0,
            "max_core": 0,
            "mean_degree": 0.0,
        }
    components = nx.number_connected_components(graph)
    core = nx.core_number(graph) if n_edges else {n: 0 for n in graph.nodes()}
    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "components": components,
        "density": float(nx.density(graph)),
        "max_core": int(max(core.values(), default=0)),
        "mean_degree": float(np.mean([degree for _, degree in graph.degree()])),
    }


def choose_study_like_threshold(corr: pd.DataFrame, config: Config) -> tuple[float, pd.DataFrame]:
    thresholds = np.arange(
        config.threshold_min,
        config.threshold_max + config.threshold_step / 2,
        config.threshold_step,
    )
    rows: list[dict[str, float | int]] = []
    effective_target_nodes = min(config.target_nodes, corr.shape[0])

    for threshold in thresholds:
        graph = build_graph(
            corr,
            threshold=float(threshold),
            mode=config.correlation_mode,
            min_degree=config.min_degree,
            keep_largest_components=config.keep_largest_components,
        )
        summary = summarize_graph(graph)
        nodes = int(summary["nodes"])
        components = int(summary["components"])
        max_core = int(summary["max_core"])

        if nodes == 0:
            score = 1e9
        else:
            node_error = abs(nodes - effective_target_nodes) / max(effective_target_nodes, 1)
            component_error = abs(components - config.target_components) / max(config.target_components, 1)
            core_error = abs(max_core - config.target_max_core) / max(config.target_max_core, 1)
            # Node count is weighted most heavily; component/core values shape the paper-like topology.
            score = 2.5 * node_error + 1.0 * component_error + 0.8 * core_error
            if nodes < 20:
                score += 5.0

        rows.append(
            {
                "threshold": float(threshold),
                **summary,
                "study_like_score": float(score),
            }
        )

    scan = pd.DataFrame(rows).sort_values(["study_like_score", "threshold"], ascending=[True, False])
    chosen = float(scan.iloc[0]["threshold"])
    return chosen, scan.sort_values("threshold").reset_index(drop=True)


def correlation_p_values(edge_table: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    result = edge_table.copy()
    if result.empty:
        result["p_value"] = pd.Series(dtype=float)
        result["fdr_bh"] = pd.Series(dtype=float)
        return result

    r = np.clip(result["correlation"].to_numpy(dtype=float), -0.999999999, 0.999999999)
    degrees_freedom = n_samples - 2
    t_stat = r * np.sqrt(degrees_freedom / (1.0 - r**2))
    p_values = 2.0 * stats.t.sf(np.abs(t_stat), df=degrees_freedom)
    result["p_value"] = p_values

    order = np.argsort(p_values)
    ranked = p_values[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    fdr = np.empty_like(adjusted)
    fdr[order] = adjusted
    result["fdr_bh"] = fdr
    return result


def graph_tables(graph: nx.Graph, n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame()

    core = nx.core_number(graph)
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    clustering = nx.clustering(graph, weight="weight")
    betweenness = nx.betweenness_centrality(graph, weight=None, normalized=True)

    component_id: dict[str, int] = {}
    for idx, component in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True), start=1):
        for node in component:
            component_id[node] = idx

    node_rows = []
    for node in graph.nodes():
        node_rows.append(
            {
                "gene": node,
                "degree": int(degree[node]),
                "weighted_degree": float(weighted_degree[node]),
                "k_core": int(core[node]),
                "component": int(component_id[node]),
                "clustering_coefficient": float(clustering[node]),
                "betweenness_centrality": float(betweenness[node]),
            }
        )
    nodes = pd.DataFrame(node_rows).sort_values(
        ["k_core", "degree", "weighted_degree", "gene"],
        ascending=[False, False, False, True],
    )

    edge_rows = []
    for source, target, attrs in graph.edges(data=True):
        edge_rows.append(
            {
                "source": source,
                "target": target,
                "correlation": float(attrs["correlation"]),
                "abs_correlation": float(attrs["abs_correlation"]),
                "sign": attrs["sign"],
            }
        )
    edges = pd.DataFrame(edge_rows).sort_values("abs_correlation", ascending=False)
    edges = correlation_p_values(edges, n_samples=n_samples)
    return nodes, edges


def _component_layout(graph: nx.Graph, seed: int) -> dict[str, np.ndarray]:
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    positions: dict[str, np.ndarray] = {}
    x_cursor = 0.0

    for component_index, component in enumerate(components):
        subgraph = graph.subgraph(component)
        n = subgraph.number_of_nodes()
        local_seed = seed + component_index * 101
        k_value = max(0.25, 1.4 / math.sqrt(max(n, 2)))
        local = nx.spring_layout(
            subgraph,
            seed=local_seed,
            k=k_value,
            iterations=500,
            weight="weight",
        )

        coords = np.array(list(local.values()), dtype=float)
        if len(coords) == 1:
            coords[:] = 0.0
        else:
            x_span = np.ptp(coords[:, 0]) or 1.0
            y_span = np.ptp(coords[:, 1]) or 1.0
            coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / x_span
            coords[:, 1] = (coords[:, 1] - coords[:, 1].mean()) / y_span

        width = max(1.2, 1.8 * math.sqrt(n / 20.0))
        height = max(0.8, 1.2 * math.sqrt(n / 20.0))
        coords[:, 0] = coords[:, 0] * width + x_cursor
        coords[:, 1] = coords[:, 1] * height
        x_cursor += width + 0.65

        for node, coord in zip(local.keys(), coords):
            positions[str(node)] = coord

    return positions


def _node_color(core_value: int) -> str:
    if core_value in KCORE_COLORS:
        return KCORE_COLORS[core_value]
    # Values above nine are shown as an even darker blue for other datasets.
    return "#061a8c"


def plot_network(graph: nx.Graph, nodes: pd.DataFrame, config: Config, output_dir: Path) -> Path:
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph contains no nodes; no plot can be created.")

    pos = _component_layout(graph, seed=config.random_seed)
    metric = nodes.set_index("gene")
    core_values = metric["k_core"].to_dict()
    degree_values = metric["degree"].to_dict()

    node_order = list(graph.nodes())
    node_colors = [_node_color(int(core_values[n])) for n in node_order]
    node_sizes = [180.0 + 75.0 * math.sqrt(max(int(degree_values[n]), 1)) for n in node_order]

    highlight_upper = {g.upper() for g in config.highlight_genes}
    border_colors = ["black" if n.upper() in highlight_upper else "#333333" for n in node_order]
    border_widths = [2.2 if n.upper() in highlight_upper else 0.8 for n in node_order]

    fig, ax = plt.subplots(figsize=(config.figure_width, config.figure_height))
    ax.set_axis_off()

    edge_colors = ["#555555" if attrs["correlation"] >= 0 else "#9a9a9a" for _, _, attrs in graph.edges(data=True)]
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=edge_colors,
        alpha=0.36,
        width=0.65,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=node_order,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=border_colors,
        linewidths=border_widths,
    )

    labels: dict[str, str]
    if config.label_mode == "none":
        labels = {}
    elif config.label_mode == "top_degree":
        label_nodes = set(metric.sort_values(["k_core", "degree"], ascending=False).head(config.top_labels).index)
        labels = {n: n for n in node_order if n in label_nodes}
    else:
        labels = {n: n for n in node_order}

    if labels:
        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels,
            ax=ax,
            font_size=6.2 if len(labels) > 80 else 7.5,
            font_weight="semibold",
        )

    max_core = int(max(core_values.values()))
    legend_values = sorted(set(core_values.values()))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_node_color(int(k)),
            markeredgecolor="#333333",
            markersize=8,
            label=f"K={int(k)}",
        )
        for k in legend_values
    ]
    ax.legend(
        handles=handles,
        title="K-core",
        loc="upper right",
        frameon=False,
        ncol=1 if len(handles) <= 10 else 2,
    )
    ax.set_title(
        "Gene Co-expression Network: Pearson Correlation and K-core\n"
        f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, maximum K-core={max_core}",
        fontsize=15,
        pad=16,
    )

    output_path = output_dir / "gene_coexpression_kcore_network.png"
    fig.savefig(output_path, dpi=config.figure_dpi, bbox_inches="tight")
    fig.savefig(output_dir / "gene_coexpression_kcore_network.pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_threshold_scan(scan: pd.DataFrame, chosen_threshold: float, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(scan["threshold"], scan["nodes"], marker="o", markersize=3, label="Network nodes")
    ax.plot(scan["threshold"], scan["edges"], marker=".", markersize=3, label="Network edges")
    ax.axvline(chosen_threshold, linestyle="--", linewidth=1.5, label=f"Chosen threshold={chosen_threshold:.3f}")
    ax.set_xlabel("Absolute correlation threshold")
    ax.set_ylabel("Count")
    ax.set_title("Threshold scan")
    ax.legend()
    ax.grid(alpha=0.25)
    path = output_dir / "threshold_scan.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_run_summary(
    config: Config,
    output_dir: Path,
    selection_description: str,
    log2_applied: bool,
    chosen_threshold: float,
    graph: nx.Graph,
    expr_selected: pd.DataFrame,
) -> None:
    summary = {
        "config": asdict(config),
        "selection_description": selection_description,
        "log2_transform_applied": log2_applied,
        "chosen_correlation_threshold": chosen_threshold,
        "selected_genes": int(expr_selected.shape[0]),
        "samples": int(expr_selected.shape[1]),
        "graph": summarize_graph(graph),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_analysis(config: Config) -> dict[str, object]:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config_used.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    if config.data_source == "geo":
        expr_raw, sample_info = load_geo_expression(config, output_dir)
    else:
        expr_raw, sample_info = load_custom_expression(config)

    _print(f"Loaded expression matrix: {expr_raw.shape[0]} genes/probes x {expr_raw.shape[1]} samples")
    expr_transformed, log2_applied = maybe_log2_transform(expr_raw, config.log2_transform)
    expr_clean = clean_expression(expr_transformed, config)
    _print(f"After cleaning: {expr_clean.shape[0]} genes x {expr_clean.shape[1]} samples")

    if expr_clean.shape[1] < 15:
        warnings.warn(
            f"Only {expr_clean.shape[1]} samples are available. Correlation networks with very small "
            "sample counts are exploratory and unstable; the paper dataset has only five time points."
        )

    expr_selected, selection_description = select_genes(expr_clean, config, output_dir)
    _print(f"Gene selection: {selection_description}; retained {expr_selected.shape[0]} genes")
    expr_selected.to_csv(output_dir / "selected_expression_matrix.csv", index_label="gene")
    sample_info.to_csv(output_dir / "sample_information.csv")

    _print(f"Computing {config.correlation_method} gene-gene correlations ...")
    corr = compute_correlation(expr_selected, config.correlation_method)
    corr.to_csv(output_dir / "gene_correlation_matrix.csv", index_label="gene")

    if config.threshold_mode == "study_like":
        chosen_threshold, threshold_scan = choose_study_like_threshold(corr, config)
        threshold_scan.to_csv(output_dir / "threshold_scan.csv", index=False)
        plot_threshold_scan(threshold_scan, chosen_threshold, output_dir)
        _print(f"Study-like automatic threshold selected: {chosen_threshold:.3f}")
    else:
        chosen_threshold = float(config.correlation_threshold)
        threshold_scan = pd.DataFrame()
        _print(f"Using fixed correlation threshold: {chosen_threshold:.3f}")

    graph = build_graph(
        corr,
        threshold=chosen_threshold,
        mode=config.correlation_mode,
        min_degree=config.min_degree,
        keep_largest_components=config.keep_largest_components,
    )
    if graph.number_of_nodes() == 0:
        raise RuntimeError(
            "No network edges survived. Lower correlation_threshold or threshold_min, "
            "or select more genes."
        )

    nodes, edges = graph_tables(graph, n_samples=expr_selected.shape[1])
    nodes.to_csv(output_dir / "node_metrics.csv", index=False)
    edges.to_csv(output_dir / "edge_list.csv", index=False)
    nx.write_graphml(graph, output_dir / "gene_coexpression_network.graphml")

    plot_path = plot_network(graph, nodes, config, output_dir)
    save_run_summary(
        config,
        output_dir,
        selection_description,
        log2_applied,
        chosen_threshold,
        graph,
        expr_selected,
    )

    summary = summarize_graph(graph)
    _print("\nNetwork summary")
    _print("-" * 60)
    _print(f"Nodes:       {summary['nodes']}")
    _print(f"Edges:       {summary['edges']}")
    _print(f"Components:  {summary['components']}")
    _print(f"Density:     {summary['density']:.5f}")
    _print(f"Mean degree: {summary['mean_degree']:.2f}")
    _print(f"Maximum K:   {summary['max_core']}")
    _print("\nTop genes by K-core and degree:")
    _print(nodes.head(20).to_string(index=False))
    _print(f"\nMain network image: {plot_path}")
    _print(f"All outputs: {output_dir}")

    return {
        "expression": expr_selected,
        "correlation": corr,
        "graph": graph,
        "nodes": nodes,
        "edges": edges,
        "threshold_scan": threshold_scan,
        "chosen_threshold": chosen_threshold,
        "output_dir": output_dir,
        "plot_path": plot_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a paper-style gene co-expression network and calculate K-core values."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--geo", default=None, help="GEO accession, for example GSE28195.")
    source.add_argument("--input", default=None, help="Custom CSV/TSV expression matrix.")
    parser.add_argument("--output", default="coexpression_results", help="Output directory.")
    parser.add_argument("--gene-column", default=None, help="Gene-symbol column in a custom matrix.")
    parser.add_argument("--orientation", choices=["genes_by_samples", "samples_by_genes"], default="genes_by_samples")
    parser.add_argument("--top-n", type=int, default=265, help="Number of top-variable genes.")
    parser.add_argument("--gene-selection", choices=["paper_s8_or_top_variance", "top_variance", "all", "gene_list"], default="paper_s8_or_top_variance")
    parser.add_argument("--gene-list", default=None, help="One-column file used with --gene-selection gene_list.")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    parser.add_argument("--mode", choices=["absolute", "positive", "negative"], default="absolute")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed correlation threshold. Omitting this activates study-like automatic tuning.")
    parser.add_argument("--keep-components", type=int, default=3)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--labels", choices=["all", "top_degree", "none"], default="all")
    parser.add_argument("--top-labels", type=int, default=60)
    parser.add_argument("--log2", choices=["auto", "yes", "no"], default="auto")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = Config(
        data_source="csv" if args.input else "geo",
        geo_accession=args.geo or "GSE28195",
        input_path=args.input,
        input_orientation=args.orientation,
        gene_column=args.gene_column,
        output_dir=args.output,
        log2_transform=args.log2,
        gene_selection=args.gene_selection,
        top_n_genes=args.top_n,
        gene_list_path=args.gene_list,
        correlation_method=args.method,
        correlation_mode=args.mode,
        threshold_mode="fixed" if args.threshold is not None else "study_like",
        correlation_threshold=args.threshold if args.threshold is not None else 0.90,
        keep_largest_components=args.keep_components,
        min_degree=args.min_degree,
        label_mode=args.labels,
        top_labels=args.top_labels,
    )

    try:
        run_analysis(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
