# PAKDD-Class-Ratio

## Initialize the Python Environment

The python requirements are stored in [requirements.txt](requirements.txt). Initialize the python environment with:

```bash
pip install -r requirements.txt
```

## Train the Random Forest Classifiers

The notebook [random_forest.ipynb](random_forest.ipynb) generates the random forest models and caches the results.

## Train the Support Vector Classifiers

The notebook [svm_rbf.ipynb](svm_rbf.ipynb) generates support vector classifiers with a radial basis function kernel and caches the results. These are the SVM results published in the paper.

The notebook [svm_linear.ipynb](svm_linear.ipynb) generates support vector classifiers with a linear kernel and caches the results.

## Train the Entity Matching Transformers

To train the entity matching transformer (EMT) models, first generate the formatted data with the script [prepare_emt_data.py](prepare_emt_data.py):

```bash
python prepare_emt_data.py
```

You can then train each EMT model with the following command. This should be run within the [entity-matching-transformer](entity-matching-transformer) directory, in a separate python environment with python 3.8.10 and the requirements in [entity-matching-transformer/requirements.txt](entity-matching-transformer/requirements.txt).

```bash
./train [data]-[ratio]-[fold_index]
```

The three parameters in the command are restricted to:

- data $\in \\{\text{abt-buy}, \text{amazon-google}, \text{walmart-amazon}, \text{wdc\\_xlarge\\_computers}, \text{wdc\\_xlarge\\_shoes}, \text{wdc\\_xlarge\\_watches}\\}$
- ratio $\in \\{1,2,3,4,5\\}$
- fold_index $\in \\{0,1,2,3,4,5,6,7,8,9\\}$

For example, to train the first of the 10 fold configurations on a 1:2 ratio for the abt-buy data:

```bash
./train abt-buy-2-0
```

## Display the Results

To render the results shown in the paper (Figure 2, Figure 3, Figure 4, and Table 2), see [create_figures.ipynb](create_figures.ipynb). This also contains the precision and recall results for the datasets not shown in Figure 3. **Note**: Figure 3 incorrectly is labeled as the results for the amazon-google dataset, but actually shows the results for the wdc_xlarge_watches dataset.

<!-- The full path for fixed_threshold_results.ipynb is required because of https://github.com/orgs/community/discussions/61164 -->
To view the $F_1$ measure results for when the classification threshold is fixed to 0.5, see [fixed_threshold_results.ipynb](/foxcroftjn/PAKDD-Class-Ratio/blob/main/fixed_threshold_results.ipynb). This notebook also contains results for the [Matthews correlation coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) (MCC) and [accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). We only compute these two additional measures for the fixed classification threshold of 0.5.

Reported accuracy improves as the class ratio grows, contrary to $F_1$ measure and MCC. This is because accuracy is computed on both classes (not just the matches) and the increase in the number of correctly identified non-matches is greater than the decrease in the number of correctly identified matches.
