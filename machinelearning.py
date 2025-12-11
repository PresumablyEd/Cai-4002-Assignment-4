import numpy as np
import pandas as pd


# --------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------
def preprocess_data(df, remove_missing=True, normalize_data=False, remove_outliers=False):
    """
    Basic preprocessing wrapper.

    Parameters
    ----------
    df : pandas.DataFrame
    remove_missing : bool
    normalize_data : bool
    remove_outliers : bool

    Returns
    -------
    df_clean : pandas.DataFrame
        Preprocessed dataframe.

    NOTE: This is a placeholder. Actual implementation required.
    """


    df_clean = df.copy()

    # ----- TODO: HANDLE MISSING VALUES PROPERLY -----
    if remove_missing:
        # TODO: choose a strategy (drop rows, fill with mean, etc.)
        pass

    # ----- TODO: REMOVE OUTLIERS PROPERLY -----
    if remove_outliers:
        # TODO: implement outlier removal rule
        # e.g., z-score, IQR, percentile clipping
        pass

    # ----- TODO: NORMALIZE DATA PROPERLY -----
    if normalize_data:
        # TODO: perform normalization/scaling on numeric columns
        pass

    return df_clean


# --------------------------------------------
# K-MEANS 
# --------------------------------------------
def initialize_centroids(X, k):
    """
    TODO: initialize k centroids.
    Could choose random points from X or random values.
    """
    raise NotImplementedError("initialize_centroids() must be implemented.")


def assign_clusters(X, centroids):
    """
    TODO: compute distances between points and centroids.
    Return an array of cluster assignments.
    """
    raise NotImplementedError("assign_clusters() must be implemented.")


def update_centroids(X, labels, k):
    """
    TODO: compute new centroid positions based on assigned points.
    """
    raise NotImplementedError("update_centroids() must be implemented.")


def kmeans_from_scratch(X, k, max_iters=100, tol=1e-4):
    """
    Perform K-Means clustering from scratch.

    TODO: Steps to implement:
    1. Initialize centroids
    2. Assign each point to a cluster
    3. Update centroids
    4. Repeat until converged or max_iters reached

    Returns
    -------
    labels : numpy.ndarray
    centroids : numpy.ndarray
    """

    raise NotImplementedError("kmeans_from_scratch() must be implemented.")


def elbow_method(X, max_k=10):
    """
    Compute distortions for k = 1..max_k.

    TODO: For each k:
        - run kmeans_from_scratch
        - compute sum of squared distances to centroids

    Returns
    -------
    distortions : list of float
    """

    raise NotImplementedError("elbow_method() must be implemented.")


# --------------------------------------------
# REGRESSION MODELS
# --------------------------------------------
def run_linear_regression(X, y):
    """
    TODO: Implement multiple linear regression.

    Suggested steps:
    - Add a bias column (ones)
    - Use normal equation or np.linalg.lstsq
    - Compute predictions

    Returns
    -------
    model : dict
    y_pred : numpy.ndarray
    """

    raise NotImplementedError("run_linear_regression() must be implemented.")


def run_polynomial_regression(X, y, degree):
    """
    TODO: Implement polynomial regression.

    Suggested steps:
    - Expand features into polynomial powers
    - Then run linear regression on expanded feature matrix

    Returns
    -------
    model : dict
    y_pred : numpy.ndarray
    """

    raise NotImplementedError("run_polynomial_regression() must be implemented.")
