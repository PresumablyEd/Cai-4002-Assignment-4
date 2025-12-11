import streamlit as st
import pandas as pd
import numpy as np

from machinelearning import (
    preprocess_data,
    kmeans_from_scratch,
    elbow_method,
    run_linear_regression,
    run_polynomial_regression,
)

st.set_page_config(page_title="Product Performance ML App")

st.title("Product Performance Analysis App")

page = st.sidebar.radio(
    "Go to",
    ["Data Upload", "Preprocessing", "Clustering", "Regression"],
)

# -------------------- DATA UPLOAD --------------------
if page == "Data Upload":
    st.header("Data Upload")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    col1, col2 = st.columns(2)
    with col1:
        load_default = st.button("Load default product_sales.csv")

    # --- User uploads file ---
    if file is not None:
        df = pd.read_csv(file)
        st.session_state["raw_data"] = df
        st.write("Preview:")
        st.dataframe(df)

    # --- Load default CSV from folder ---
    elif load_default:
        try:
            df = pd.read_csv("product_sales.csv")
            st.session_state["raw_data"] = df
            st.success("Loaded product_sales.csv")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Could not load product_sales.csv: {e}")

    else:
        st.info("Upload a CSV file or load the default one.")

# -------------------- PREPROCESSING --------------------
elif page == "Preprocessing":
    st.header("Preprocessing")

    if "raw_data" not in st.session_state:
        st.warning("Upload or load data first.")
    else:
        df = st.session_state["raw_data"]
        st.write("Current data:")
        st.dataframe(df)

        remove_missing = st.checkbox("Remove missing values", value=True)
        normalize = st.checkbox("Normalize numeric columns")
        remove_outliers = st.checkbox("Remove outliers")

        if st.button("Run preprocessing"):
            try:
                df_clean = preprocess_data(
                    df,
                    remove_missing=remove_missing,
                    normalize_data=normalize,
                    remove_outliers=remove_outliers,
                )
                st.session_state["preprocessed_data"] = df_clean
                st.write("Preprocessed data:")
                st.dataframe(df_clean)
            except Exception as e:
                st.error(f"preprocess_data() error (backend not done yet?): {e}")

# -------------------- CLUSTERING --------------------
elif page == "Clustering":
    st.header("K-Means Clustering")

    df = st.session_state.get("preprocessed_data", st.session_state.get("raw_data", None))

    if df is None:
        st.warning("Upload (and maybe preprocess) data first.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            st.error("No numeric columns found.")
        else:
            st.write("Numeric columns:", numeric_cols)

            features = st.multiselect("Select feature columns", numeric_cols)
            k = st.slider("Number of clusters (k)", 2, 10, 3)

            if st.button("Run K-Means"):
                if not features:
                    st.warning("Choose at least one feature.")
                else:
                    X = df[features].values
                    try:
                        labels, centroids = kmeans_from_scratch(X, k)
                        st.write("Cluster labels (first 20):", labels[:20])
                        st.write("Centroids:", centroids)
                    except Exception as e:
                        st.error(f"kmeans_from_scratch() error (expected until backend done): {e}")

            if st.button("Run Elbow Method"):
                if not features:
                    st.warning("Choose at least one feature.")
                else:
                    X = df[features].values
                    try:
                        distortions = elbow_method(X, max_k=10)
                        st.write("Distortions:", distortions)
                    except Exception as e:
                        st.error(f"elbow_method() error (expected until backend done): {e}")

# -------------------- REGRESSION --------------------
elif page == "Regression":
    st.header("Regression")

    df = st.session_state.get("preprocessed_data", st.session_state.get("raw_data", None))

    if df is None:
        st.warning("Upload (and maybe preprocess) data first.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Need at least two numeric columns.")
        else:
            target = st.selectbox("Target (y)", numeric_cols)
            feature_options = [c for c in numeric_cols if c != target]
            features = st.multiselect("Features (X)", feature_options)

            model_type = st.selectbox("Model type", ["Linear", "Polynomial"])

            degree = 2
            if model_type == "Polynomial":
                degree = st.slider("Polynomial degree", 2, 6, 2)

            if st.button("Run regression"):
                if not features:
                    st.warning("Select at least one feature.")
                else:
                    X = df[features].values
                    y = df[target].values

                    try:
                        if model_type == "Linear":
                            model, y_pred = run_linear_regression(X, y)
                        else:
                            model, y_pred = run_polynomial_regression(X, y, degree)

                        st.write("First 20 predictions:")
                        st.write(y_pred[:20])
                    except Exception as e:
                        st.error(f"Regression function error (backend not implemented yet): {e}")
