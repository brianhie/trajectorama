
## Coscape

Coscape is an algorithm that implements coexpression-based integrative analysis of multi-study single cell transcriptomics. Coscape is described in the paper "Coexpression uncovers a unified single-cell transcriptomic landscape" by Brian Hie, Hyunghoon Cho, Bryan Bryson, and Bonnie Berger.

**Coscape is still under active development. Follow the repository for changes and improvements.**

#### Dependencies

Coscape is tested with Python version 3.7 on Ubuntu 18.04.

There are a number of dependencies

Scientific python included in Anaconda (tested versions listed below):
scikit-learn (0.20.3)
numpy (1.16.2)
scipy (1.3.0)
matplotlib (3.0.3)
networkx (2.2)

Other packages (tested with versions listed below):
fa2 (0.3.5)
scanpy (1.4.4)
scanorama (1.4)
geosketch (1.0)
python-igraph (0.7.1)

The pipeline also requires a custom implementation of Louvain clustering, which can be installed with the below commands:
```
git clone https://github.com/brianhie/louvain-igraph
cd louvain-igraph
python setup.py install

```

#### Coscape for mouse neuronal development

Coscape analyzes five large-scale studies of mouse neurons over multiple points in development.

Data can be found at http://cb.csail.mit.edu/cb/coscape/mouse_data.tar.gz and can be downloaded as:
```
wget http://cb.csail.mit.edu/cb/coscape/data_mouse_develop.tar.gz
tar xvf data_mouse_develop.tar.gz
```

To preprocess and load the data, run the command
```
python bin/mouse_develop.py > mouse_develop.log
```

This will save the coexpression matrices to a directory under `target/sparse_correlations/`.

The downstream analysis can then be performed on these cached matrices using the commands
```
python bin/mouse_develop_cached.py >> mouse_develop.log
python bin/mouse_develop_dictionary.py >> mouse_develop.log
```
This will log some relevant statistics and save visualizations under the `figures/` directory.


#### Coscape for human hematopoiesis

We can perform a similar workflow for human hematopoiesis by running the commands:
```
wget http://cb.csail.mit.edu/cb/coscape/data_hematopoiesis.tar.gz
tar xvf data_hematopoiesis.tar.gz
python bin/hematopoiesis.py > hematopoiesis.log
python bin/hematopoiesis_cached.py >> hematopoiesis.log
python bin/hematopoiesis_dictionary.py >> hematopoiesis.log
```

#### Questions

Create an issue in the repository or contact brianhie@mit.edu for any pertinent questions or concerns. We will do our best to answer promptly and feel free to create a pull request and contribute!
