
## Trajectorama

Trajectorama is an algorithm that implements coexpression-based integration of multi-study single-cell trajectories. Trajectorama is described in the paper ["Coexpression enables multi-study cellular trajectories of development and disease"](https://www.biorxiv.org/content/10.1101/719088v1) by Brian Hie, Hyunghoon Cho, Bryan Bryson, and Bonnie Berger.

**Trajectorama is still under active development. Follow the repository for changes and improvements.**

### Dependencies

Trajectorama is tested with Python version 3.7 on Ubuntu 18.04. There are a number of Python package dependencies.

Scientific python packages included in [Anaconda](https://www.anaconda.com/distribution/) are listed below (tested versions listed):
- scikit-learn (0.20.3)
- numpy (1.16.2)
- scipy (1.3.0)
- matplotlib (3.0.3)
- networkx (2.2)

Other packages are listed below (tested versions listed):
- [fa2](https://github.com/bhargavchippada/forceatlas2) (0.3.5)
- [scanpy](https://scanpy.readthedocs.io/en/stable/) (1.4.4)
- [scanorama](https://github.com/brianhie/scanorama) (1.4)
- [geosketch](https://github.com/brianhie/geosketch) (1.0)
- [python-igraph](https://igraph.org/python/) (0.7.1)

The pipeline also requires a [custom implementation](https://github.com/brianhie/louvain-igraph) of Louvain clustering, which can be installed with the below commands:
```
git clone https://github.com/brianhie/louvain-igraph
cd louvain-igraph
python setup.py install
```

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
```
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

### Questions

Create an issue in the repository or contact brianhie@mit.edu for any pertinent questions or concerns. We will do our best to answer promptly and feel free to create a pull request and contribute!
