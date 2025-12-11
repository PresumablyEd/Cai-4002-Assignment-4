import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# --------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------
def preprocess_data(df, remove_missing=True, normalize_data=False, remove_outliers=False):
    """
    Comprehensive data preprocessing for machine learning.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to preprocess
    remove_missing : bool
        Whether to handle missing values
    normalize_data : bool
        Whether to normalize numerical features
    remove_outliers : bool
        Whether to detect and remove outliers

    Returns
    -------
    df_clean : pandas.DataFrame
        Preprocessed dataframe
    """
    df_clean = df.copy()
    
    # ----- MISSING VALUE HANDLING -----
    if remove_missing:
        # Check for missing values in each column
        missing_counts = df_clean.isnull().sum()
        total_rows = len(df_clean)
        
        # Drop rows with >50% missing values
        threshold_cols = len(df_clean.columns) // 2
        df_clean = df_clean.dropna(thresh=threshold_cols)
        
        # Handle remaining missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['object', 'category']:
                    # For categorical, use mode
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                    else:
                        df_clean[col] = df_clean[col].fillna('Unknown')
                else:
                    # For numerical, use median (more robust to outliers)
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
    
    # ----- OUTLIER DETECTION AND TREATMENT -----
    if remove_outliers:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Use IQR method for outlier detection
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers at boundaries (winsorization)
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    # ----- FEATURE NORMALIZATION -----
    if normalize_data:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Use Z-score standardization (better for K-means)
            scaler = StandardScaler()
            df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    
    return df_clean


# --------------------------------------------
# K-MEANS 
# --------------------------------------------
def initialize_centroids(X, k):
    """
    Initialize k centroids using random selection from data points.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix (n_samples, n_features)
    k : int
        Number of clusters
        
    Returns
    -------
    centroids : numpy.ndarray
        Initial centroid positions (k, n_features)
    """
    n_samples, n_features = X.shape
    
    # Randomly select k unique points from the dataset
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    
    return centroids


def assign_clusters(X, centroids):
    """
    Assign each data point to the nearest centroid using Euclidean distance.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix (n_samples, n_features)
    centroids : numpy.ndarray
        Current centroid positions (k, n_features)
        
    Returns
    -------
    labels : numpy.ndarray
        Cluster assignment for each data point (n_samples,)
    """
    n_samples = X.shape[0]
    k = centroids.shape[0]
    
    # Calculate distances from each point to each centroid
    distances = np.zeros((n_samples, k))
    
    for i in range(k):
        # Euclidean distance from all points to centroid i
        distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
    
    # Assign each point to the nearest centroid
    labels = np.argmin(distances, axis=1)
    
    return labels


def update_centroids(X, labels, k):
    """
    Update centroid positions based on the mean of assigned points.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix (n_samples, n_features)
    labels : numpy.ndarray
        Current cluster assignments (n_samples,)
    k : int
        Number of clusters
        
    Returns
    -------
    new_centroids : numpy.ndarray
        Updated centroid positions (k, n_features)
    """
    n_features = X.shape[1]
    new_centroids = np.zeros((k, n_features))
    
    for i in range(k):
        # Get all points assigned to cluster i
        cluster_points = X[labels == i]
        
        if len(cluster_points) > 0:
            # Calculate mean of points in cluster
            new_centroids[i] = np.mean(cluster_points, axis=0)
        else:
            # If no points assigned to cluster, reinitialize randomly
            new_centroids[i] = X[np.random.choice(len(X))]
    
    return new_centroids


def kmeans_from_scratch(X, k, max_iters=100, tol=1e-4):
    """
    Perform K-Means clustering from scratch.

    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix (n_samples, n_features)
    k : int
        Number of clusters
    max_iters : int
        Maximum number of iterations
    tol : float
        Convergence tolerance for centroid movement

    Returns
    -------
    labels : numpy.ndarray
        Cluster assignment for each data point (n_samples,)
    centroids : numpy.ndarray
        Final centroid positions (k, n_features)
    """
    # Initialize centroids
    centroids = initialize_centroids(X, k)
    
    for iteration in range(max_iters):
        # Assign points to clusters
        labels = assign_clusters(X, centroids)
        
        # Update centroids
        new_centroids = update_centroids(X, labels, k)
        
        # Check for convergence
        centroid_movement = np.sqrt(np.sum((new_centroids - centroids)**2))
        
        centroids = new_centroids
        
        if centroid_movement < tol:
            print(f"K-means converged after {iteration + 1} iterations")
            break
    
    return labels, centroids


def elbow_method(X, max_k=10):
    """
    Compute within-cluster sum of squares (WCSS) for different values of k.

    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix (n_samples, n_features)
    max_k : int
        Maximum number of clusters to test

    Returns
    -------
    distortions : list of float
        WCSS values for each k (from 1 to max_k)
    """
    distortions = []
    
    # Start from k=1 to max_k
    for k in range(1, max_k + 1):
        labels, centroids = kmeans_from_scratch(X, k, max_iters=50)
        
        # Calculate WCSS (within-cluster sum of squares)
        wcss = 0.0
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # Sum of squared distances from points to their centroid
                wcss += np.sum((cluster_points - centroids[i])**2)
        
        distortions.append(wcss)
        print(f"K={k}, WCSS={wcss:.2f}")
    
    return distortions


# --------------------------------------------
# REGRESSION MODELS
# --------------------------------------------
def run_linear_regression(X, y):
    """
    Implement multiple linear regression using the normal equation.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix (n_samples, n_features)
    y : numpy.ndarray
        Target vector (n_samples,)

    Returns
    -------
    model : dict
        Dictionary containing model parameters and coefficients
    y_pred : numpy.ndarray
        Predicted values
    """
    # Ensure X and y are numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Add bias term (intercept) to X
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    
    # Use least squares method to solve for coefficients
    # This is more numerically stable than the normal equation
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(X_with_bias, y, rcond=None)
    
    # Make predictions
    y_pred = X_with_bias @ coefficients
    
    # Create model dictionary
    model = {
        'coefficients': coefficients,
        'intercept': coefficients[0],
        'slope': coefficients[1:],
        'residuals': residuals,
        'rank': rank,
        'singular_values': singular_values
    }
    
    return model, y_pred


def run_polynomial_regression(X, y, degree):
    """
    Implement polynomial regression by expanding features and using linear regression.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix (n_samples, n_features)
    y : numpy.ndarray
        Target vector (n_samples,)
    degree : int
        Degree of polynomial features

    Returns
    -------
    model : dict
        Dictionary containing model parameters and coefficients
    y_pred : numpy.ndarray
        Predicted values
    """
    # Ensure X and y are numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Generate polynomial features
    n_samples, n_features = X.shape
    
    # Start with bias term
    X_poly = np.ones((n_samples, 1))
    
    # Add polynomial features up to specified degree
    for d in range(1, degree + 1):
        for feature in range(n_features):
            X_poly = np.column_stack([X_poly, X[:, feature] ** d])
    
    # Use least squares method to solve for coefficients
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(X_poly, y, rcond=None)
    
    # Make predictions
    y_pred = X_poly @ coefficients
    
    # Create model dictionary
    model = {
        'coefficients': coefficients,
        'intercept': coefficients[0],
        'polynomial_coefficients': coefficients[1:],
        'degree': degree,
        'residuals': residuals,
        'rank': rank,
        'singular_values': singular_values
    }
    
    return model, y_pred
