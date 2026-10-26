import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from src.data_loader import (
    load_dataset,
    load_model,
    compute_dataset_summary,
    get_data_quality_report,
)
from src.preprocessing import (
    OUTLET_IDENTIFIERS,
    ITEM_TYPES,
    OUTLET_SIZES,
    OUTLET_LOCATION_TYPES,
    OUTLET_TYPES,
    ITEM_FAT_CONTENTS,
    OUTLET_METADATA,
    GLOBAL_MEAN_WEIGHT,
    GLOBAL_MEAN_VISIBILITY,
    validate_uploaded_csv,
    CATBOOST_FEATURE_NAMES,
)
from src.prediction import (
    predict_single,
    predict_batch,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom CSS: Modern Navy-Blue & White Business Theme
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global layout & typography */
    .main {
        background-color: #F8FAFC;
    }
    h1, h2, h3, h4 {
        color: #0F2042;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    
    /* Header card */
    .header-banner {
        background: linear-gradient(135deg, #0F2042 0%, #1E3A8A 60%, #2563EB 100%);
        color: #FFFFFF;
        padding: 26px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(15, 32, 66, 0.12);
    }
    .header-banner h1 {
        color: #FFFFFF !important;
        margin: 0 0 6px 0;
        font-size: 2.1rem;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: #E2E8F0;
        margin: 0;
        font-size: 1.05rem;
        font-weight: 400;
    }
    
    /* Metric KPI Card */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(15, 32, 66, 0.08);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F2042;
        margin-bottom: 4px;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 500;
    }
    
    /* Prediction Banner */
    .prediction-result-card {
        background: linear-gradient(135deg, #0F2042 0%, #1E3A8A 100%);
        color: #FFFFFF;
        padding: 28px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.2);
        margin: 18px 0;
    }
    .prediction-title {
        font-size: 1.05rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #93C5FD;
        font-weight: 600;
    }
    .prediction-amount {
        font-size: 3.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 10px 0;
    }
    
    /* Clean Cards */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    
    /* Badges */
    .badge-primary {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button[kind="primary"] {
        background-color: #1E3A8A;
        border-color: #1E3A8A;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #0F2042;
        border-color: #0F2042;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data & Model Ingestion (Cached)
# ---------------------------------------------------------
try:
    df_raw = load_dataset()
    catboost_model = load_model()
    dataset_loaded = True
except Exception as e:
    df_raw = pd.DataFrame()
    catboost_model = None
    dataset_loaded = False
    st.error(f"Initialization Error: {e}")

# ---------------------------------------------------------
# Sidebar Navigation & Branding
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">
        <h2 style="color: #0F2042; margin: 0; font-size: 1.35rem; display: flex; align-items: center; gap: 8px;">
            🛒 Retail Analytics
        </h2>
        <span style="font-size: 0.82rem; color: #64748B;">BigMart Sales Forecasting v2.0</span>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation Menu",
        options=[
            "📊 Overview Dashboard",
            "🔮 Sales Prediction",
            "🔍 Dataset Explorer",
            "📈 Forecast Visualization",
            "📖 About the Project",
        ],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Quick System Status")
    if dataset_loaded:
        st.success(f"✅ Dataset Loaded: {len(df_raw):,} records")
        st.success(f"✅ Model: CatBoost (22 features)")
    else:
        st.error("❌ Dataset / Model not found")
        
    st.caption("College Internship Data Science Project Demo")


# ---------------------------------------------------------
# PAGE A: OVERVIEW DASHBOARD
# ---------------------------------------------------------
if page == "📊 Overview Dashboard":
    st.markdown("""
    <div class="header-banner">
        <h1>Sales Forecasting & Analytics Dashboard</h1>
        <p>Machine Learning-Powered Retail Sales Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not dataset_loaded:
        st.warning("Please ensure Train.csv is available in the Dataset folder.")
        st.stop()

    summary = compute_dataset_summary(df_raw)

    # Top KPI Metrics Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Historical Sales</div>
            <div class="kpi-value">${summary['total_sales']/1e6:.2f}M</div>
            <div class="kpi-subtext">Across all 10 outlets</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Sales / Record</div>
            <div class="kpi-value">${summary['avg_sales']:,.2f}</div>
            <div class="kpi-subtext">Median: ${summary['median_sales']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Transactions</div>
            <div class="kpi-value">{summary['total_records']:,}</div>
            <div class="kpi-subtext">Clean historical records</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Unique Products</div>
            <div class="kpi-value">{summary['unique_products']:,}</div>
            <div class="kpi-subtext">16 item categories</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Outlets</div>
            <div class="kpi-value">{summary['unique_outlets']}</div>
            <div class="kpi-subtext">4 distinct store formats</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Filter Controls
    with st.expander("🔍 Interactive Overview Filters", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            loc_filter = st.multiselect(
                "Filter by Outlet Location Type:",
                options=sorted(df_raw["Outlet_Location_Type"].dropna().unique()),
                default=sorted(df_raw["Outlet_Location_Type"].dropna().unique()),
            )
        with f_col2:
            type_filter = st.multiselect(
                "Filter by Outlet Type:",
                options=sorted(df_raw["Outlet_Type"].dropna().unique()),
                default=sorted(df_raw["Outlet_Type"].dropna().unique()),
            )
        with f_col3:
            item_filter = st.multiselect(
                "Filter by Item Type:",
                options=sorted(df_raw["Item_Type"].dropna().unique()),
                default=sorted(df_raw["Item_Type"].dropna().unique()),
            )

    # Apply filters
    filtered_df = df_raw[
        (df_raw["Outlet_Location_Type"].isin(loc_filter)) &
        (df_raw["Outlet_Type"].isin(type_filter)) &
        (df_raw["Item_Type"].isin(item_filter))
    ]

    if filtered_df.empty:
        st.warning("No records match the current filter selection. Please reset your filters.")
        st.stop()

    # Visual Analytics Row 1
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.subheader("📊 Sales Distribution Analysis")
        fig_dist = px.histogram(
            filtered_df,
            x="Item_Outlet_Sales",
            nbins=45,
            marginal="box",
            color_discrete_sequence=["#1E3A8A"],
            title="Distribution of Historical Sales Revenue",
            labels={"Item_Outlet_Sales": "Sales Revenue ($)"},
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with row1_col2:
        st.subheader("🏬 Sales by Outlet Type")
        outlet_sales = (
            filtered_df.groupby("Outlet_Type")["Item_Outlet_Sales"]
            .agg(["sum", "mean"])
            .reset_index()
            .rename(columns={"sum": "Total Sales", "mean": "Average Sales"})
        )
        fig_outlet = px.bar(
            outlet_sales,
            x="Outlet_Type",
            y="Total Sales",
            color="Outlet_Type",
            text_auto=".2s",
            color_discrete_sequence=["#1E3A8A", "#2563EB", "#38BDF8", "#0F2042"],
            title="Total Revenue by Outlet Store Format",
            labels={"Total Sales": "Total Sales ($)", "Outlet_Type": "Outlet Format"},
        )
        fig_outlet.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_outlet, use_container_width=True)

    # Visual Analytics Row 2
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.subheader("🏷️ Sales Performance by Item Category")
        item_sales = (
            filtered_df.groupby("Item_Type")["Item_Outlet_Sales"]
            .sum()
            .reset_index()
            .sort_values(by="Item_Outlet_Sales", ascending=True)
        )
        fig_item = px.bar(
            item_sales,
            x="Item_Outlet_Sales",
            y="Item_Type",
            orientation="h",
            color="Item_Outlet_Sales",
            color_continuous_scale="Blues",
            title="Total Sales by Item Type",
            labels={"Item_Outlet_Sales": "Total Sales ($)", "Item_Type": "Product Category"},
        )
        fig_item.update_layout(
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_item, use_container_width=True)

    with row2_col2:
        st.subheader("📍 Location Tier vs Outlet Size Revenue")
        loc_size_sales = (
            filtered_df.groupby(["Outlet_Location_Type", "Outlet_Size"], dropna=False)["Item_Outlet_Sales"]
            .mean()
            .reset_index()
        )
        loc_size_sales["Outlet_Size"] = loc_size_sales["Outlet_Size"].fillna("Unspecified (Small Mode)")
        fig_loc = px.bar(
            loc_size_sales,
            x="Outlet_Location_Type",
            y="Item_Outlet_Sales",
            color="Outlet_Size",
            barmode="group",
            color_discrete_map={
                "High": "#0F2042",
                "Medium": "#2563EB",
                "Small": "#60A5FA",
                "Unspecified (Small Mode)": "#94A3B8",
            },
            title="Average Sales: Location Tier by Store Size",
            labels={"Item_Outlet_Sales": "Avg Sales ($)", "Outlet_Location_Type": "Location Tier"},
        )
        fig_loc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_loc, use_container_width=True)

    # Correlation Heatmap Row
    st.subheader("🔥 Numerical Correlation Matrix")
    numeric_cols = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year", "Item_Outlet_Sales"]
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="Blues",
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 13, "color": "black"},
        hoverongaps=False,
    ))
    fig_corr.update_layout(
        title="Pearson Correlation Across Numerical Features",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------------------------------
# PAGE B: SALES PREDICTION
# ---------------------------------------------------------
elif page == "🔮 Sales Prediction":
    st.markdown("""
    <div class="header-banner">
        <h1>Interactive Sales Prediction Engine</h1>
        <p>Real-Time Inference Powered by the Pre-Trained CatBoost Regressor</p>
    </div>
    """, unsafe_allow_html=True)

    if catboost_model is None:
        st.error("Error: Trained model `Pickle/Model.pkl` could not be loaded.")
        st.stop()

    st.markdown("""
    Provide product and retail store attributes below to generate an immediate, model-driven sales prediction.
    """)

    # Interactive Form
    with st.form("prediction_form", clear_on_submit=False):
        col_prod, col_outlet = st.columns(2)
        
        with col_prod:
            st.markdown("### 📦 Product Attributes")
            item_id = st.text_input(
                "Item Identifier (Code)",
                value="FDA15",
                help="Example: FDA15, DRC01, NCD19. Prefix indicates Food (FD), Drinks (DR), or Non-Consumable (NC).",
            )
            item_weight = st.number_input(
                "Item Weight (kg)",
                min_value=1.0,
                max_value=30.0,
                value=float(GLOBAL_MEAN_WEIGHT),
                step=0.1,
                help="Net weight in kilograms. Missing values default to global mean (12.86 kg).",
            )
            item_fat = st.selectbox(
                "Item Fat Content",
                options=ITEM_FAT_CONTENTS,
                index=0,
                help="Fat content. Non-consumable items are automatically reclassified as Non-Edible.",
            )
            item_vis = st.slider(
                "Item Visibility (%)",
                min_value=0.0,
                max_value=0.35,
                value=float(GLOBAL_MEAN_VISIBILITY),
                step=0.005,
                format="%.3f",
                help="Percentage of total display area allocated to this product in store aisles.",
            )
            item_type = st.selectbox(
                "Item Category (Type)",
                options=ITEM_TYPES,
                index=ITEM_TYPES.index("Dairy") if "Dairy" in ITEM_TYPES else 0,
            )
            item_mrp = st.number_input(
                "Item MRP ($)",
                min_value=10.0,
                max_value=350.0,
                value=140.0,
                step=1.0,
                help="Maximum Retail Price. Primary driver of individual transaction revenue (~56% importance).",
            )

        with col_outlet:
            st.markdown("### 🏬 Outlet Attributes")
            outlet_id = st.selectbox(
                "Outlet Identifier",
                options=OUTLET_IDENTIFIERS,
                index=OUTLET_IDENTIFIERS.index("OUT049") if "OUT049" in OUTLET_IDENTIFIERS else 0,
                help="Select target store branch (OUT010 through OUT049).",
            )
            
            # Lookup default metadata for selected outlet
            default_meta = OUTLET_METADATA.get(outlet_id, {})
            default_est = default_meta.get("Establishment_Year", 1999)
            default_size = default_meta.get("Size", "Medium")
            default_loc = default_meta.get("Location_Type", "Tier 1")
            default_type = default_meta.get("Type", "Supermarket Type1")

            auto_fill = st.checkbox("Auto-align store specs with selected Outlet ID", value=True)

            if auto_fill:
                est_year = st.number_input(
                    "Establishment Year",
                    min_value=1980,
                    max_value=2024,
                    value=int(default_est),
                    disabled=True,
                )
                outlet_size = st.selectbox(
                    "Outlet Size",
                    options=OUTLET_SIZES,
                    index=OUTLET_SIZES.index(default_size) if default_size in OUTLET_SIZES else 1,
                    disabled=True,
                )
                outlet_loc = st.selectbox(
                    "Outlet Location Type",
                    options=OUTLET_LOCATION_TYPES,
                    index=OUTLET_LOCATION_TYPES.index(default_loc) if default_loc in OUTLET_LOCATION_TYPES else 0,
                    disabled=True,
                )
                outlet_type = st.selectbox(
                    "Outlet Store Type",
                    options=OUTLET_TYPES,
                    index=OUTLET_TYPES.index(default_type) if default_type in OUTLET_TYPES else 1,
                    disabled=True,
                )
            else:
                est_year = st.number_input(
                    "Establishment Year",
                    min_value=1980,
                    max_value=2024,
                    value=int(default_est),
                )
                outlet_size = st.selectbox(
                    "Outlet Size",
                    options=OUTLET_SIZES,
                    index=OUTLET_SIZES.index(default_size) if default_size in OUTLET_SIZES else 1,
                )
                outlet_loc = st.selectbox(
                    "Outlet Location Type",
                    options=OUTLET_LOCATION_TYPES,
                    index=OUTLET_LOCATION_TYPES.index(default_loc) if default_loc in OUTLET_LOCATION_TYPES else 0,
                )
                outlet_type = st.selectbox(
                    "Outlet Store Type",
                    options=OUTLET_TYPES,
                    index=OUTLET_TYPES.index(default_type) if default_type in OUTLET_TYPES else 1,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            submit_btn = st.form_submit_button("🔮 Predict Sales", type="primary", use_container_width=True)

    # Execution upon submission
    if submit_btn:
        with st.spinner("Applying exact data transformations and invoking CatBoost Regressor..."):
            input_dict = {
                "Item_Identifier": item_id,
                "Item_Weight": item_weight,
                "Item_Fat_Content": item_fat,
                "Item_Visibility": item_vis,
                "Item_Type": item_type,
                "Item_MRP": item_mrp,
                "Outlet_Identifier": outlet_id,
                "Outlet_Establishment_Year": est_year,
                "Outlet_Size": outlet_size,
                "Outlet_Location_Type": outlet_loc,
                "Outlet_Type": outlet_type,
            }

            try:
                predicted_val, feature_df, expl = predict_single(catboost_model, input_dict)

                # Prominent Output Card
                st.markdown(f"""
                <div class="prediction-result-card">
                    <div class="prediction-title">Estimated Sales Revenue</div>
                    <div class="prediction-amount">${predicted_val:,.2f}</div>
                    <div style="font-size: 0.95rem; color: #DBEAFE;">
                        Primary Model Drivers: {expl['primary_driver']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Contextual Business Drivers
                st.subheader("💡 Key Prediction Factors & Business Insights")
                for insight in expl["insights"]:
                    st.markdown(f"- {insight}")

                # Model Transparency: Preprocessed 22-dimensional feature vector
                with st.expander("🛠️ Inspect Exact 22-Feature Preprocessed Vector Fed to Model"):
                    st.markdown("Below is the exact encoded vector fed into `catboost.predict()`:")
                    st.dataframe(feature_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Pipeline Error: {e}")


# ---------------------------------------------------------
# PAGE C: DATASET EXPLORER
# ---------------------------------------------------------
elif page == "🔍 Dataset Explorer":
    st.markdown("""
    <div class="header-banner">
        <h1>Dataset Explorer & Quality Audit</h1>
        <p>Interactive Exploration, Schema Inspection, and Custom CSV Validation</p>
    </div>
    """, unsafe_allow_html=True)

    tab_explore, tab_quality, tab_upload = st.tabs([
        "📋 Browse Dataset",
        "🔬 Data Quality & Schema",
        "📤 Upload & Predict Custom CSV",
    ])

    with tab_explore:
        st.subheader("Raw Training Dataset Preview")
        
        # Search & Filter Controls
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            search_query = st.text_input("Search by Item Identifier or Outlet ID:", "")
        with filter_col2:
            sort_by = st.selectbox(
                "Sort by Column:",
                options=list(df_raw.columns),
                index=df_raw.columns.get_loc("Item_Outlet_Sales"),
            )
        with filter_col3:
            sort_order = st.radio("Order:", options=["Descending", "Ascending"], horizontal=True)

        display_df = df_raw.copy()
        if search_query:
            display_df = display_df[
                display_df["Item_Identifier"].astype(str).str.contains(search_query, case=False, na=False) |
                display_df["Outlet_Identifier"].astype(str).str.contains(search_query, case=False, na=False)
            ]

        display_df = display_df.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))

        row_limit = st.slider("Display row limit:", min_value=10, max_value=500, value=50, step=10)
        st.dataframe(display_df.head(row_limit), use_container_width=True)
        st.caption(f"Showing {min(row_limit, len(display_df))} of {len(display_df):,} matching rows.")

        # Download dataset CSV
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="BigMart_Sales_Data.csv",
            mime="text/csv",
        )

    with tab_quality:
        st.subheader("Dataset Structure & Missing Value Analysis")
        quality_df = get_data_quality_report(df_raw)
        st.dataframe(quality_df, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Statistical Summary (Numerical Columns)")
        st.dataframe(df_raw.describe(), use_container_width=True)

    with tab_upload:
        st.subheader("Upload Custom CSV for Analysis & Batch Predictions")
        st.markdown("""
        Upload a retail sales CSV file. The system validates column compatibility, applies preprocessing, 
        and allows generating batch CatBoost model predictions.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Shape: {user_df.shape[0]} rows, {user_df.shape[1]} columns.")
                
                # Schema validation
                is_valid, missing_cols, warnings = validate_uploaded_csv(user_df)
                
                if warnings:
                    for w in warnings:
                        st.info(f"ℹ️ Note: {w}")

                if not is_valid:
                    st.error(f"Uploaded CSV is missing mandatory columns required by model: {missing_cols}")
                    st.stop()

                st.dataframe(user_df.head(20), use_container_width=True)

                if st.button("🚀 Run Batch Model Predictions on Uploaded File", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        preds_df, eval_metrics = predict_batch(catboost_model, user_df)
                        
                        st.success("Batch prediction completed successfully!")
                        if eval_metrics:
                            st.markdown("### 📊 Model Evaluation on Uploaded Ground Truth:")
                            m_col1, m_col2, m_col3 = st.columns(3)
                            m_col1.metric("R² Score", f"{eval_metrics['r2_score']:.4f}")
                            m_col2.metric("RMSE ($)", f"{eval_metrics['rmse']:,.2f}")
                            m_col3.metric("MAE ($)", f"{eval_metrics['mae']:,.2f}")

                        st.subheader("Predicted Output Table Preview")
                        st.dataframe(preds_df.head(50), use_container_width=True)

                        out_csv = preds_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=out_csv,
                            file_name="Model_Sales_Predictions.csv",
                            mime="text/csv",
                        )

            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")


# ---------------------------------------------------------
# PAGE D: FORECAST VISUALIZATION
# ---------------------------------------------------------
elif page == "📈 Forecast Visualization":
    st.markdown("""
    <div class="header-banner">
        <h1>Predictive & Forecast Analytics</h1>
        <p>Model Evaluation, Actual vs. Predicted Variance, and Feature Importance</p>
    </div>
    """, unsafe_allow_html=True)

    # Important Distinction Alert
    st.info("""
    ℹ️ **Methodology Note for Evaluators & Teachers**:
    This project performs **item-level regression prediction** (predicting sales volume based on product attributes, pricing, and retail store formats).
    It does not simulate temporal calendar forecasting (such as day-by-day seasonality or ARIMA time-series) because the historical BigMart dataset records aggregate store sales rather than dated daily transactional time-series.
    """)

    # Cache batch prediction on full dataset for visualization
    @st.cache_data
    def get_full_predictions():
        preds_df, metrics = predict_batch(catboost_model, df_raw)
        return preds_df, metrics

    with st.spinner("Generating full model evaluation predictions..."):
        df_eval, full_metrics = get_full_predictions()

    # Metric Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Model Architecture", "CatBoost Regressor")
    m_col2.metric("Overall R² Score", f"{full_metrics['r2_score']:.4f}")
    m_col3.metric("RMSE ($)", f"{full_metrics['rmse']:,.2f}")
    m_col4.metric("Mean Absolute Error ($)", f"{full_metrics['mae']:,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Scatter Plot: Actual vs Predicted
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.subheader("🎯 Actual vs. Model Predicted Sales")
        # Subsample 1500 points for crisp interactive rendering
        sample_eval = df_eval.sample(n=min(1500, len(df_eval)), random_state=42)
        fig_scatter = px.scatter(
            sample_eval,
            x="Item_Outlet_Sales",
            y="Predicted_Sales",
            color="Outlet_Type",
            opacity=0.65,
            trendline="ols",
            title="Actual Sales vs. Predicted Sales (Sampled)",
            labels={"Item_Outlet_Sales": "Actual Sales ($)", "Predicted_Sales": "Predicted Sales ($)"},
        )
        # Add 45-degree reference line
        max_val = max(sample_eval["Item_Outlet_Sales"].max(), sample_eval["Predicted_Sales"].max())
        fig_scatter.add_shape(
            type="line",
            line=dict(dash="dash", color="black", width=1.5),
            x0=0, x1=max_val, y0=0, y1=max_val
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_vis2:
        st.subheader("📉 Prediction Residuals (Errors) Distribution")
        fig_res = px.histogram(
            df_eval,
            x="Residual",
            nbins=50,
            marginal="box",
            color_discrete_sequence=["#2563EB"],
            title="Distribution of Residuals (Actual - Predicted)",
            labels={"Residual": "Residual Error ($)"},
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="red")
        fig_res.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # Aggregated Comparison: By Outlet Type & Item Type
    st.subheader("📊 Grouped Performance: Actual vs. Predicted Aggregate")
    grp_col1, grp_col2 = st.columns(2)
    
    with grp_col1:
        comp_outlet = df_eval.groupby("Outlet_Type")[["Item_Outlet_Sales", "Predicted_Sales"]].mean().reset_index()
        comp_outlet = pd.melt(
            comp_outlet,
            id_vars=["Outlet_Type"],
            value_vars=["Item_Outlet_Sales", "Predicted_Sales"],
            var_name="Type",
            value_name="Average Sales ($)",
        )
        comp_outlet["Type"] = comp_outlet["Type"].replace({"Item_Outlet_Sales": "Actual Sales", "Predicted_Sales": "Predicted Sales"})
        fig_grp_outlet = px.bar(
            comp_outlet,
            x="Outlet_Type",
            y="Average Sales ($)",
            color="Type",
            barmode="group",
            color_discrete_map={"Actual Sales": "#0F2042", "Predicted Sales": "#38BDF8"},
            title="Average Sales: Actual vs. Predicted by Outlet Type",
        )
        fig_grp_outlet.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_grp_outlet, use_container_width=True)

    with grp_col2:
        # Feature Importance Plot
        st.subheader("🏆 Model Feature Importance")
        fi_series = pd.Series(catboost_model.feature_importances_, index=CATBOOST_FEATURE_NAMES).sort_values(ascending=True)
        # Top 12 features for clean presentation
        top_fi = fi_series.tail(12).reset_index()
        top_fi.columns = ["Feature", "Importance (%)"]
        fig_fi = px.bar(
            top_fi,
            x="Importance (%)",
            y="Feature",
            orientation="h",
            color="Importance (%)",
            color_continuous_scale="Blues",
            title="Top Feature Importance (CatBoost Regressor)",
        )
        fig_fi.update_layout(
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# ---------------------------------------------------------
# PAGE E: ABOUT THE PROJECT
# ---------------------------------------------------------
elif page == "📖 About the Project":
    st.markdown("""
    <div class="header-banner">
        <h1>About the Sales Forecasting Project</h1>
        <p>Comprehensive Academic Documentation & Presentation Guide</p>
    </div>
    """, unsafe_allow_html=True)

    tab_overview, tab_prep, tab_models, tab_qa = st.tabs([
        "📌 Project Background & Objectives",
        "🧹 Data Cleaning & Preprocessing",
        "🤖 Machine Learning Methodology",
        "🎓 Viva & Teacher Q&A Preparation",
    ])

    with tab_overview:
        st.markdown("""
        ### 1. Problem Statement
        Retail organizations handle thousands of diverse inventory items across diverse retail formats ranging from small grocery outlets to large-scale hypermarkets.
        Without an accurate sales forecasting mechanism, retail managers face:
        - **Stockouts & lost revenue** for high-velocity items.
        - **Excess inventory holding costs & spoilage** for slow-moving products.
        - **Inefficient store space allocation**.

        The goal of this project is to develop an end-to-end Machine Learning regression solution that predicts the sales volume of individual products across various retail store formats.

        ### 2. Project Objectives
        1. **Conduct Exploratory Data Analysis (EDA)** to understand retail revenue dynamics.
        2. **Implement disciplined preprocessing** to resolve real-world data flaws (missing weights, zero visibility artifacts, missing outlet sizes).
        3. **Engineer predictive features** such as store maturity vintage (`Outlet_Years`) and item classifications (`New_Item_Type`).
        4. **Benchmark multiple regression algorithms** (Linear, Ridge, Lasso, Decision Trees, Random Forest, Extra Trees, XGBoost, LightGBM, and CatBoost).
        5. **Deploy the champion model** inside an interactive, business-ready Streamlit analytics application.

        ### 3. Dataset Profile
        - **Source**: BigMart Sales Data (`Train.csv`).
        - **Total Records**: 8,523 observations.
        - **Attributes**: 12 columns (11 independent features + 1 target variable `Item_Outlet_Sales`).
        """)

    with tab_prep:
        st.markdown("""
        ### Data Preprocessing Pipeline

        Real-world retail datasets require careful data cleaning prior to model training:

        | Column | Issue Identified | Resolution Applied |
        | :--- | :--- | :--- |
        | **`Item_Weight`** | 1,463 missing values (~17.1%) | Imputed via item identifier mean lookup table; overall mean fallback. |
        | **`Outlet_Size`** | 2,410 missing values (~28.3%) | Imputed using the statistical mode of `Outlet_Size` grouped by `Outlet_Type`. |
        | **`Item_Visibility`** | 526 records with 0.0 visibility | Replaced invalid zeros with the column-wide average visibility (0.0661). |
        | **`Item_Fat_Content`** | Inconsistent labels (`LF`, `low fat`, `reg`) | Standardized to `Low Fat` and `Regular`. Reclassified to `Non-Edible` for non-consumable goods. |
        | **`New_Item_Type`** | Unstructured item identifiers | Derived 2-letter prefix (`FD` = Food, `DR` = Drinks, `NC` = Non-Consumable). |
        | **`Outlet_Years`** | Static establishment year | Transformed into store operating age (`Reference_Year - Establishment_Year`). |
        | **Categorical Features** | Non-numeric data | `Outlet_Identifier` and `Item_Type` label encoded; remaining categories one-hot encoded into 22 model features. |
        """)

    with tab_models:
        st.markdown("""
        ### Model Selection & Evaluation

        The project systematically evaluated 8 distinct regression models using 10-fold cross-validation and R² scoring:

        1. **Baseline Linear Models**:
           - Ordinary Least Squares (OLS) Linear Regression: $R^2 \\approx 0.56$
           - Ridge Regression (L2 Regularization): $R^2 \\approx 0.56$
           - Lasso Regression (L1 Regularization): $R^2 \\approx 0.56$
        2. **Tree-Based Ensembles**:
           - Decision Tree Regressor: prone to overfitting on raw data.
           - Random Forest Regressor ($R^2 \\approx 0.58$ after tuning).
           - Extra Trees Regressor: fast variance reduction.
        3. **Gradient Boosted Decision Trees (GBDT)**:
           - XGBoost Regressor ($R^2 \\approx 0.59$).
           - LightGBM Regressor ($R^2 \\approx 0.59$).
           - **CatBoost Regressor** (👑 Champion Model): achieved highest stability and accuracy with $R^2 \\approx 0.605$ on full data.

        #### Champion Model Hyperparameters (Tuned via RandomizedSearchCV):
        ```python
        CatBoostRegressor(
            learning_rate=0.0894,
            max_depth=2,
            n_estimators=109,
            subsample=0.6676,
            loss_function='RMSE'
        )
        ```
        """)

    with tab_qa:
        st.markdown("""
        ### 🎓 Viva & Teacher Q&A Cheatsheet

        Use these concise, technically accurate answers during your presentation:

        **Q1: Why did you choose CatBoost over Random Forest or Linear Regression?**
        > *Answer*: "CatBoost handles categorical variables and tabular interactions exceptionally well using symmetric decision trees (oblivious trees), which prevents overfitting on retail tabular data. In cross-validation, CatBoost consistently achieved the lowest Root Mean Squared Error (RMSE) and highest $R^2$ score ($0.605$)."

        **Q2: What is the most important feature driving sales predictions?**
        > *Answer*: "`Item_MRP` (Maximum Retail Price) is by far the most influential feature, accounting for approximately 55.8% of the model's split decisions, followed by store format (`Outlet_Type` Grocery Store vs Supermarket Type 3) accounting for ~42.9% combined. This reflects real-world retail economics where unit pricing sets base transaction value and store format determines customer traffic volume."

        **Q3: How did you handle items with 0.0 visibility?**
        > *Answer*: "In a physical supermarket, no stocked item can have zero physical shelf visibility unless it is not displayed at all. Therefore, zero visibility is a data recording artifact. We replaced zero values with the mean visibility across all items."

        **Q4: Is this a time-series forecast or a regression prediction?**
        > *Answer*: "This is an item-level tabular regression model. The BigMart dataset records aggregate cross-sectional store sales without transactional date timestamps. Rather than forecasting sales across time dates, our model predicts sales volume for any product given its physical properties, price point, and the retail environment it is placed in."

        **Q5: How are missing weights and outlet sizes handled in real-time predictions?**
        > *Answer*: "Our preprocessing module implements deterministic fallback rules: missing weights use item-level pivot means or global fallbacks, and missing outlet sizes are dynamically imputed based on store format mode."
        """)
