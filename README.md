
# Trajectorama

Trajectorama is an algorithm that implements coexpression-based integration of multi-study single-cell trajectories. Trajectorama is described in the paper ["Coexpression enables multi-study cellular trajectories of development and disease"](https://www.biorxiv.org/content/10.1101/719088v1) by Brian Hie, Hyunghoon Cho, Bryan Bryson, and Bonnie Berger.

## Installation

The most import dependency is on a [custom implementation](https://github.com/brianhie/louvain-igraph) of Louvain clustering, which can be installed with the below commands:
```
git clone https://github.com/brianhie/louvain-igraph
cd louvain-igraph
python setup.py install
```
Installing Trajectorama can then be done by:
```
python -m pip install trajectorama
```

## API and example usage

We provide a basic API around the core algorithm that takes an expression matrix augmented with study information and returns a list of coexpression matrices, with corresponding indices into the original data:
```python
import trajectorama

X = [ ... ] # Sample-by-gene expression matrix.
studies = [ ... ] # Study identifiers, one for each row of `X`.

Xs_coexpr, sample_idxs = trajectorama.transform(
    X, studies,
    corr_cutoff=0.7,
    corr_method='spearman',
    cluster_method='louvain',
    min_cluster_samples=500,
)
```

The coexpression matrix `Xs_coexpr[i]` is defined over the subset of cells `X[sample_idxs[i], :]`. **See the documentation string under the `transform()` function at the top of [trajectorama/trajectorama.py](trajectorama/trajectorama.py) for the full list of parameters and default values.**

This list of coexpression matrices can then be used in further analysis, e.g., you can flatten the matrices and use [Scanpy](https://scanpy.readthedocs.io/) to visualize the matrices as a KNN graph based on distance in coexpression space:
```python
import numpy as np
import scanpy as sc
from anndata import AnnData

# Save upper triangle and flatten.
n_features = X.shape[1]
triu_idx = np.triu_indices(n_features) # Indices of upper triangle.
X_coexpr = np.concatenate([
    X_coexpr[triu_idx].flatten() for X_coexpr in X_coeprs
])

# Plot KNN graph in coexpression space.
adata = AnnData(X_coexpr)
sc.pp.neighbors(adata)
sc.tl.draw_graph(adata)
sc.pl.draw_graph(adata)

```

The example scripts below show more detailed usage of Trajectorama, which was used to generate the paper results.

## Examples

### Trajectorama for mouse neuronal development

Trajectorama analyzes five large-scale studies of mouse neurons over multiple points in development.

Data can be found at http://trajectorama.csail.mit.edu/data.tar.gz and can be downloaded as:
```
wget http://trajectorama.csail.mit.edu/data.tar.gz
tar xvf data.tar.gz
```

To preprocess the data, run the command:
```
python bin/process.py conf/mouse_develop.txt
```
This preprocessing step only needs to be done once. Then, we perform panclustering and coexpression matrix computation using the command:
```
python bin/mouse_develop.py > mouse_develop.log
```
This will save each coexpression matrix as a `.npz` file to a directory under `target/sparse_correlations/`. Computing all coexpression matrices should complete in around an hour when running on a single core.

The downstream analysis can then be performed on these cached matrices using the commands:
```
python bin/mouse_develop_cached.py >> mouse_develop.log
python bin/mouse_develop_dictionary.py >> mouse_develop.log
```
This will log some relevant statistics and save visualizations under the `figures/` directory.

### Trajectorama for human hematopoiesis

We can perform a similar workflow for human hematopoiesis by running the commands:
```bash
# Download (if not done so for mouse data).
wget http://trajectorama.csail.mit.edu/data.tar.gz
tar xvf data.tar.gz

# Preprocess.
python bin/process.py conf/hematopoiesis.txt

# Analyze.
python bin/hematopoiesis.py > hematopoiesis.log
python bin/hematopoiesis_cached.py >> hematopoiesis.log
python bin/hematopoiesis_dictionary.py >> hematopoiesis.log
```

### Trajectorama for microglia

We can perform a similar workflow for mouse and human microglia in various conditions by running the commands:
```bash
# Download (if not done so for mouse data).
wget http://trajectorama.csail.mit.edu/data.tar.gz
tar xvf data.tar.gz

# Preprocess.
python bin/process.py conf/microglia.txt

# Analyze.
python bin/microglia.py > microglia.log
python bin/microglia_cached.py >> microglia.log
```

## Questions

Create an issue in the repository or contact brianhie@mit.edu for any pertinent questions or concerns. We will do our best to answer promptly and feel free to create a pull request and contribute!
