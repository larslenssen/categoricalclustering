# Clustering for Categorical Data with Relative Entropy Dissimilarity in Python

This package provides best practices for clustering of categorical data sets. It implements CatRED: Clustering for Categorical Data with Relative Entropy Dissimilarity.

For further details on the implemented algorithm CatRED and the evaluation measures MREC and ARES, see:

> Lars Lenssen, Philip Stahmann, Christian Janiesch, Erich Schubert  
> **Archetype Discovery from Taxonomies:**  
> **A Method to Cluster Small Datasets of Categorical Data**  
> TBD 
<!-- +> 58th Hawaii International Conference on System Sciences, HICSS 2025.  
> DOI
> Preprint: 
+ -->

If you use this code in scientific work, please cite above paper. Thank you.

## Documentation

## Installation

### Installation with pip

PyPi is pending 

You need to have Python 3 installed.

```sh
pip install git+https://github.com/larslenssen/categoricalclustering
```

or

```sh
# activate your desired virtual environment first, then:
git clone https://github.com/larslenssen/categorical-clustering.git
cd categoricalclustering
# build and install the package:
pip install -q build
python -m build
# change the version
pip install dist/categoricalclustering-X.X.X-py3-none-any.whl
```




## A Comprehensive Guide to Clustering Categorical Data 

The computational steps have been implemented within a Google Colab notebook. Researchers and practitioners can access this notebook to review, replicate, or build upon the procedures outlined.

The Colab notebook is available at the following URL: [clustering-categorical-data](https://colab.research.google.com/drive/1RZG0hyNUr-VKCe8dtDZR32loU9Euryi-?usp=sharing).

To cluster a dataset comprising categorical or binary variables, it is crucial to first comprehend the data's structure. Determine whether the dataset is binary or categorical. If the data is binary, it is advisable to convert it into categorical form. This conversion process can be facilitated using the ```merge_onehot_categories``` method, which is included in our Python package. This method is also employed as a step (**Data Preprocessing**) within the provided Colab Notebook for ease of implementation.

To decide which features are important for the data set, it makes sense to carry out a correlation analysis. Common methods such as Cramér's V, Tschuprow's T, and Pearson correlation coefficient are implemented in the notebook (**Correlation Analysis**). 

Following a thorough analysis or based on expert knowledge, it may be appropriate to assign different weights to the features for the cluster analysis (**Customize Feature Weights**). In cases where there is uncertainty regarding the weighting of features, it is advisable to perform the cluster analysis unweighted.

After selecting an appropriate clustering method (**Select Clustering Method**) — hierarchical methods such as CatRED are particularly recommended — the results should be analyzed to determine the most suitable clustering solution.

![dendogram](https://github.com/larslenssen/categoricalclustering/blob/main/documentation/dendogram.png?raw=true)
![cluster distribution](https://github.com/larslenssen/categoricalclustering/blob/main/documentation/cluster_dist.png?raw=true)

Results for various numbers of clusters should be compared to evaluate their quality. Once a preferred clustering solution is identified, it can be exported (**Export selected clustering**) for further analysis and processing.

## Example

The Colab Notebook is provided in a slightly abbreviated form as a Jupyter Notebook, available within the package under the filename ```example_notebook.ipynb```:

```sh
df = pd.read_excel("elma2019.xlsx", header=0, index_col=[0]).fillna(0.0).astype(np.int64)
df = cc.merge_onehot_categories(df)
weights = pd.Series(np.ones(len(df.columns)), index=df.columns)
catclustres = cc.catred(df, weights=weights)
cc.analyse_linkagematrix(df, catclustres.linkage_matrix, weights, 4, title=f' Choosing 4 clusters, ')
```

## Implemented Algorithms

* **catRED** (Lenssen and Stahmann and Janiesch and Schubert, 2025)
* ARES: Average Relative Entropy Score (Lenssen and Stahmann and Janiesch and Schubert, 2025)
* MREC: Minimum Relative Entropy Contrast (Lenssen and Stahmann and Janiesch and Schubert, 2025)

## Contributing to `categorical-clustering`

Third-party contributions are welcome. Please use [pull requests](https://github.com/larslenssen/categorical-clustering/pulls) to submit patches.

## Reporting issues

Please report errors as an [issue](https://github.com/larslenssen/categorical-clustering/issues) within the repository's issue tracker.

## Support requests

If you need help, please submit an [issue](https://github.com/larslenssen/categorical-clustering/issues) within the repository's issue tracker.

## License: GPL-3 or later

> This program is free software: you can redistribute it and/or modify
> it under the terms of the GNU General Public License as published by
> the Free Software Foundation, either version 3 of the License, or
> (at your option) any later version.
> 
> This program is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
> GNU General Public License for more details.
> 
> You should have received a copy of the GNU General Public License
> along with this program.  If not, see <https://www.gnu.org/licenses/>.
