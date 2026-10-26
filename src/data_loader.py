import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Dict, Any

def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    current_file = os.path.abspath(__file__)
    # current_file is in <root>/src/data_loader.py
    src_dir = os.path.dirname(current_file)
    root_dir = os.path.dirname(src_dir)
    return root_dir

def find_file(relative_subpaths: list) -> Optional[str]:
    """Tries multiple candidate paths for a file and returns the first existing path."""
    root = get_project_root()
    # Candidate search roots
    candidate_roots = [
        root,
        os.path.join(root, "Sales-Forecasting-Datascience-Project-main"),
        os.getcwd(),
        os.path.join(os.getcwd(), "Sales-Forecasting-Datascience-Project-main"),
    ]
    for c_root in candidate_roots:
        for subpath in relative_subpaths:
            full_path = os.path.abspath(os.path.join(c_root, subpath))
            if os.path.isfile(full_path):
                return full_path
    return None

def get_dataset_path() -> str:
    """Finds and returns the absolute path to Train.csv."""
    candidates = [
        os.path.join("Dataset", "Train.csv"),
        "Train.csv",
    ]
    path = find_file(candidates)
    if not path:
        raise FileNotFoundError(
            "Could not find 'Train.csv' in expected locations (Dataset/Train.csv)."
        )
    return path

def get_model_path() -> str:
    """Finds and returns the absolute path to Model.pkl."""
    candidates = [
        os.path.join("Pickle", "Model.pkl"),
        "Model.pkl",
    ]
    path = find_file(candidates)
    if not path:
        raise FileNotFoundError(
            "Could not find 'Model.pkl' in expected locations (Pickle/Model.pkl)."
        )
    return path

@st.cache_data(show_spinner="Loading sales historical dataset...")
def load_dataset() -> pd.DataFrame:
    """
    Loads and returns the original Train.csv dataset.
    Cached with st.cache_data for performance.
    """
    path = get_dataset_path()
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner="Loading trained CatBoost model...")
def load_model():
    """
    Loads and returns the trained CatBoost regression model.
    Cached with st.cache_resource to prevent redundant disk I/O.
    """
    path = get_model_path()
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def compute_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes headline KPI metrics for the overview dashboard."""
    total_sales = float(df["Item_Outlet_Sales"].sum())
    avg_sales = float(df["Item_Outlet_Sales"].mean())
    median_sales = float(df["Item_Outlet_Sales"].median())
    total_records = int(len(df))
    unique_products = int(df["Item_Identifier"].nunique())
    unique_outlets = int(df["Outlet_Identifier"].nunique())
    avg_mrp = float(df["Item_MRP"].mean())
    avg_visibility = float(df["Item_Visibility"].mean())

    return {
        "total_sales": total_sales,
        "avg_sales": avg_sales,
        "median_sales": median_sales,
        "total_records": total_records,
        "unique_products": unique_products,
        "unique_outlets": unique_outlets,
        "avg_mrp": avg_mrp,
        "avg_visibility": avg_visibility,
    }

def get_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Produces a clean dataframe detailing column data types, non-null counts, and missing percentages."""
    summary = []
    total = len(df)
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = (null_count / total) * 100
        dtype = str(df[col].dtype)
        unique_count = int(df[col].nunique())
        summary.append({
            "Column Name": col,
            "Data Type": dtype,
            "Non-Null Count": total - null_count,
            "Missing Count": null_count,
            "Missing (%)": round(null_pct, 2),
            "Unique Values": unique_count,
            "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
        })
    return pd.DataFrame(summary)
