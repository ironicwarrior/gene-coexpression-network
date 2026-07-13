# Ready-to-run gene co-expression and K-core analysis

This project creates the same **type** of network shown in Figure 4 of:

Zhang J. et al. (2011), *Identification of Hub Genes Related to the Recovery Phase of Irradiation Injury by Microarray and Integrated Gene Network Analysis*, PLOS ONE, DOI: 10.1371/journal.pone.0024680.

The default analysis downloads the public NCBI GEO series **GSE28195** and runs a paper-style workflow:

1. download the processed expression values and platform annotation;
2. map microarray probes to gene symbols;
3. optionally recover genes from the paper's supplementary Table S8, with a top-variable-gene fallback;
4. calculate pairwise Pearson correlations;
5. create a hard-thresholded undirected network;
6. keep the three largest connected regions;
7. calculate degree, K-core, clustering coefficient, and betweenness;
8. draw nodes by K-core and size them by degree;
9. export CSV, GraphML, PNG, PDF, and JSON files.

## Easiest method: Google Colab

1. Upload `Gene_Coexpression_Kcore_ReadyToRun.ipynb` to Google Colab.
2. Choose **Runtime > Run all**.
3. The notebook downloads GSE28195, runs the analysis, displays the network, and creates a ZIP of all outputs.

The pre-filled study-style targets are:

- Pearson correlation;
- unsigned/absolute correlation network;
- 265 candidate genes if the paper supplement cannot be recovered;
- approximately 156 plotted nodes;
- three main components;
- target maximum K-core of 9.

The article does not report the exact correlation cutoff or every preprocessing detail. The notebook therefore scans thresholds and chooses one that approximates the reported network scale. This produces a reproducible paper-style network, but it is not guaranteed to be pixel-identical to the published image.

## Command-line use

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the public dataset:

```bash
python gene_coexpression_kcore.py --geo GSE28195 --output GSE28195_results
```

Run a custom matrix:

```bash
python gene_coexpression_kcore.py \
  --input my_normalized_expression.csv \
  --output my_results \
  --gene-selection top_variance \
  --top-n 500 \
  --threshold 0.85 \
  --labels top_degree
```

## Custom input format

The default orientation is genes in rows and samples in columns:

```text
gene,control_1,control_2,treated_1,treated_2
TP53,7.1,7.4,8.0,8.3
GAPDH,12.2,12.0,11.4,11.6
```

Use normalized expression values. Do not interpret correlations calculated directly from raw RNA-seq count integers. For RNA-seq, use a variance-stabilized matrix or an appropriate normalized/log-transformed expression matrix.

## Outputs

- `gene_coexpression_kcore_network.png`
- `gene_coexpression_kcore_network.pdf`
- `node_metrics.csv`
- `edge_list.csv`
- `gene_coexpression_network.graphml`
- `selected_expression_matrix.csv`
- `gene_correlation_matrix.csv`
- `threshold_scan.csv`
- `threshold_scan.png`
- `sample_information.csv`
- `run_summary.json`
- `config_used.json`

## Important interpretation note

The paper's Figure 4 is a **Pearson-correlation plus K-core graph**, not a conventional WGCNA module analysis. A graph edge means that two expression profiles passed the selected correlation rule. It does not demonstrate physical protein binding, direct regulation, or causality.

The public study has only five time points. That is sufficient to demonstrate the authors' historical procedure, but correlations from so few observations are unstable. The official WGCNA guidance recommends not attempting WGCNA with fewer than 15 samples and suggests at least 20 when possible.

## Primary sources

- Paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0024680
- GEO series: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28195
- NetworkX K-core definition: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.core.k_core.html
- WGCNA FAQ: https://edo98811.github.io/WGCNA_official_documentation/faq.html
