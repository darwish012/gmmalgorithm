# Gaussian Mixture Model Clustering for Clothing Sizes (Python)

A from-scratch implementation of the Gaussian Mixture Model (GMM) trained with the Expectation-Maximization (EM) algorithm, used to cluster a dataset into clothing-size groups. The script also measures how the running time of the algorithm grows with the number of samples and the number of clusters, and plots the clusters in a 2-D PCA projection.

## What is implemented

- Data normalization: z-score normalization of every column (`normalize_data`).
- `initialize_parameters`: random samples as initial means, the overall data covariance for every component, equal weights.
- `expectation`: responsibilities of each sample for each component, using `scipy.stats.multivariate_normal`.
- `maximization`: updates of weights, means and covariance matrices from the responsibilities.
- `gmm`: normalizes the data and runs 25 EM iterations; each sample is assigned to the component with the highest responsibility.
- `gmm_without_normalising`: the same loop on already-normalized data, up to 100 iterations, stopping when the change in log-likelihood is below `1e-15`.
- Runtime experiment: for K = 3 and K = 5, `gmm` is run on random subsamples containing 1% to 100% of the data (in steps of 1%), and the run times are saved to `runtimes.csv`. Three plots follow: raw run time against N, a spline-smoothed version, and a linear-regression fit (`scikit-learn` `LinearRegression`).
- Clustering view: the normalized data is reduced to 2 components with PCA, clustered with K = 3 and K = 5, and drawn as a scatter plot per K. Each plot lists the percentage of samples in each cluster, labelled S, M, L, XL and XXL in cluster order.

## Files

| File | Description |
| --- | --- |
| `anaylsiscode.py` | The whole implementation, experiment and plotting code. |

## How to run

- Python 3 with `pandas`, `numpy`, `matplotlib`, `scipy` and `scikit-learn`.
- The script reads a CSV file with `pd.read_csv` from a hard-coded Windows path (`...\dataset.csv`). The dataset is not included in this repository. Change that path to your own file before running. All columns are passed to the GMM and PCA, so the file must contain numeric columns only.
- Run `python anaylsiscode.py`. It writes `runtimes.csv` in the working directory and opens the plot windows one after another. The runtime experiment runs the GMM 200 times, so it can take a while on a large dataset.

## Author

Abdelrahman Darwish (GitHub: [@darwish012](https://github.com/darwish012))
