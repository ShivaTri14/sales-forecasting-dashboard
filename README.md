# 🛒 Sales Forecasting & Analytics Dashboard
### Machine Learning-Powered Retail Sales Prediction System

An interactive, end-to-end data science web application that forecasts product sales across retail store formats using an optimized **CatBoost Regression** machine learning pipeline. Built with Python, Scikit-Learn, and Streamlit.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features & Dashboard Pages](#-key-features--dashboard-pages)
- [Machine Learning Architecture](#-machine-learning-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start Guide (Windows PowerShell / VS Code)](#-quick-start-guide-windows-powershell--vs-code)
- [Cloud Deployment (Streamlit Community Cloud)](#-cloud-deployment-streamlit-community-cloud)
- [Viva & Presentation Guide](#-viva--presentation-guide)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Project Overview

Retail supply chains face a dual challenge: under-stocking high-demand items leads to lost revenue, while over-stocking slow-moving products leads to inventory holding costs and perishability losses.

This project delivers an interactive decision-support system that predicts the sales volume (`Item_Outlet_Sales`) of retail merchandise across different store types (grocery stores, supermarkets, hypermarkets). It transforms raw retail data into actionable pricing and placement strategies.

### Technologies Used
- **Frontend / Dashboard**: Streamlit, Plotly Express & Graph Objects, HTML5/CSS3 custom theme
- **Machine Learning Core**: CatBoost Regressor, Scikit-Learn
- **Data Engineering**: Pandas, NumPy, SciPy
- **Runtime & Deployment**: Python 3.10 - 3.14, Streamlit Community Cloud

---

## 🖥️ Key Features & Dashboard Pages

### 📊 Page 1: Overview Dashboard
- **Executive KPI Cards**: Total Historical Sales ($18.59M), Average Sales per Item ($2,181.29), Total Transactions (8,523), Unique Products (1,559), and Active Outlets (10).
- **Interactive Visualizations**:
  - Historical Sales Distribution with boxplot margin.
  - Revenue Breakdown by Store Format (`Outlet_Type`).
  - Item Category Sales Performance (`Item_Type`).
  - Store Tier vs. Outlet Size Matrix.
  - Correlation Heatmap across numerical features.
- **Dynamic Slicing**: Multiselect filters for location tiers, store types, and product lines.

### 🔮 Page 2: Real-Time Sales Prediction
- **Product Specification Inputs**: Item Identifier code, Item Net Weight, Item Fat Content, Item Shelf Visibility, Item Category, and Maximum Retail Price (MRP).
- **Store Specification Inputs**: Outlet Identifier (with auto-fill option for historical store attributes), Establishment Year, Store Size, and Location Tier.
- **Instant Model Inference**: Applies the exact 22-dimensional feature transformation pipeline and feeds it to the pre-trained `Model.pkl`.
- **Actionable Business Insights**: Automatically explains the key drivers behind each individual prediction (MRP tier impact, store format multiplier, shelf visibility prominence).
- **Feature Vector Inspector**: Transparently displays the exact numeric vector fed into CatBoost.

### 🔍 Page 3: Dataset Explorer & Quality Audit
- **Interactive Data Table**: Search by Item Code or Outlet ID, sort by any column, and adjust display limits.
- **Data Quality Audit**: Non-null counts, exact missing percentages, data types, and unique cardinality.
- **Upload Custom CSV**: Upload new retail batches for automatic schema validation and batch model predictions.
- **Export Capabilities**: One-click download buttons for filtered datasets and generated predictions.

### 📈 Page 4: Predictive & Forecast Analytics
- **Actual vs. Predicted Scatter**: Evaluates model fit ($R^2 \approx 0.605$) with an ideal 45-degree reference line.
- **Residual Error Distribution**: Histogram and boxplot confirming balanced, near-zero errors.
- **Store Format Comparison**: Side-by-side grouped bar charts comparing actual vs predicted sales across retail types.
- **Global Feature Importance**: Interactive bar chart displaying the contribution of all 22 model features.
- **Methodology Note**: Transparently clarifies the distinction between item-level regression prediction and calendar time-series forecasting.

### 📖 Page 5: Project Documentation & Presentation Guide
- Complete academic walkthrough covering problem definition, data cleaning rules, benchmarked regression algorithms, and ready-to-present viva Q&A.

---

## 🤖 Machine Learning Architecture

### Preprocessing & Feature Engineering Pipeline
1. **`Item_Weight` Imputation**: Missing values (~17.1%) are imputed using an item-specific mean lookup table; overall dataset mean ($12.86\text{ kg}$) is used as a fallback.
2. **`Outlet_Size` Imputation**: Missing values (~28.3%) are imputed using the statistical mode of `Outlet_Size` conditioned on `Outlet_Type`.
3. **`Item_Visibility` Correction**: Physically impossible zero-visibility values ($0.0$) are replaced with the global mean visibility ($0.0661$).
4. **`Item_Fat_Content` Normalization**: Cleans mislabeled entries (`LF`, `low fat`, `reg`) and automatically reclassifies non-consumable items as `Non-Edible`.
5. **`New_Item_Type` Derivation**: Extracted from the item identifier prefix (`FD` $\to$ Food, `DR` $\to$ Drinks, `NC` $\to$ Non-Consumables).
6. **`Outlet_Years` Engineering**: Calculates operational store maturity ($\text{Reference Year} - \text{Outlet\_Establishment\_Year}$).
7. **Encoding Pipeline**: Label encoding for `Outlet` and `Item_Type`; One-Hot Encoding for categorical dummy columns producing the exact 22 features expected by the trained model.

### Top Feature Importance Breakdown
- **Item_MRP (55.85%)**: Primary revenue determinant; higher unit prices scale total transaction volume.
- **Outlet_Type_0 - Grocery Store (29.34%)**: Small-format footprint penalty (-75% volume vs hypermarkets).
- **Outlet_Type_3 - Supermarket Type 3 (13.54%)**: High footfall hypermarket volume premium.
- **Outlet_Years & Item_Visibility (1.27%)**: Store maturity and aisle prominence.

---

## 📁 Project Directory Structure

```text
Sales-Forecasting-Datascience-Project-main/
├── app.py                      # Main Streamlit interactive web application
├── requirements.txt            # Production dependencies for local run & cloud deployment
├── README.md                   # Complete documentation, setup, and presentation guide
├── .gitignore                  # Git exclusions (venv, pycache, temporary files)
├── .streamlit/
│   └── config.toml             # Custom white & navy-blue theme configuration
├── src/                        # Modular, reusable production code
│   ├── __init__.py             # Package initializer
│   ├── data_loader.py          # Data ingestion, caching, and descriptive statistics
│   ├── preprocessing.py        # Exact 22-feature transformation pipeline
│   └── prediction.py           # Single & batch inference and explanation generator
├── Dataset/
│   └── Train.csv               # Historical BigMart training dataset (8,523 rows)
├── Pickle/
│   └── Model.pkl               # Pre-trained CatBoost regression model artifact
├── Codes/
│   ├── Sales Forecast.ipynb    # Original Jupyter notebook with model exploration
│   └── Sales Forecast.py       # Standalone training script
└── Graph and Visualization/    # Original exploratory data analysis chart exports
    ├── Correlation Heat Map.png
    ├── Count of Item_Fat_Content.png
    ├── Count of Item_Type.png
    ├── Count of Outlet_Establishment_Year.png
    ├── Count of Outlet_Location_Type.png
    ├── Count of Outlet_Size.png
    ├── Count of Outlet_Type.png
    └── Distribution of Item_Weight.png
```

---

## 🚀 Quick Start Guide (Windows PowerShell / VS Code)

### Prerequisites
- Windows 10/11 with Python 3.10+ installed.
- PowerShell or Windows Terminal.

### Step 1: Open the Project in VS Code
Open VS Code, press `Ctrl + O` or select **File > Open Folder...**, and navigate to:
```text
C:\Users\shiva\Downloads\Sales-Forecasting-Datascience-Project-main
```

### Step 2: Open PowerShell Terminal
In VS Code, press `` Ctrl + ` `` or go to **Terminal > New Terminal**.

### Step 3: Set Up a Virtual Environment (Recommended)
```powershell
# Create a fresh virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

> **Note on Execution Policy:** If PowerShell displays `cannot be loaded because running scripts is disabled on this system`, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Launch the Application
```powershell
streamlit run app.py
```

The application will launch automatically in your default web browser at:
👉 **`http://localhost:8501`**

---

## ☁️ Cloud Deployment (Streamlit Community Cloud)

You can publish this dashboard online for free with a public shareable URL:

1. **Push to GitHub**:
   - Initialize a Git repository in the project root:
     ```powershell
     git init
     git add .
     git commit -m "Complete Sales Forecasting & Analytics Dashboard"
     ```
   - Create a new public repository on [GitHub](https://github.com/new).
   - Link and push your repository:
     ```powershell
     git remote add origin https://github.com/<your-username>/sales-forecasting-dashboard.git
     git branch -M main
     git push -u origin main
     ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
   - Click **New app**.
   - Select your repository (`sales-forecasting-dashboard`), branch (`main`), and set **Main file path** to `app.py`.
   - Click **Deploy!**
   - In 2–3 minutes, your dashboard will be live with a public URL (e.g. `https://sales-forecasting-dashboard.streamlit.app`).

---

## 🎓 Viva & Presentation Guide

### Elevator Pitch (30-Second Summary for Evaluators)
> "Good morning/afternoon, teachers. For my internship project, I developed an interactive Sales Forecasting & Analytics Dashboard for retail organizations. Using historical data from BigMart across 10 diverse retail outlets, I addressed common real-world data issues such as missing item weights and missing store sizes, engineered retail-specific features like store maturity age and item categories, and trained a tuned CatBoost Regressor. Our model achieves an R² score of ~0.605 on full data, where Item MRP and store format together drive over 98% of the sales variation. I encapsulated this entire workflow into a Streamlit business dashboard that enables real-time sales predictions, batch CSV scoring, and transparent factor analysis."

### Key Viva Questions & Clear Answers

| Question | Model Answer |
| :--- | :--- |
| **Why CatBoost over Random Forest?** | CatBoost handles tabular categorical interactions natively using symmetric decision trees (oblivious trees). This minimizes variance and overfitting compared to standard deep decision trees, achieving lower RMSE in cross-validation. |
| **Why not forecast sales by future dates?** | The BigMart dataset records aggregate cross-sectional sales per item and outlet rather than daily chronological time-series records. It is formulated as a tabular regression problem predicting volume based on product and store features. |
| **Why is Item MRP the dominant feature?** | Maximum Retail Price sets the baseline dollar value per transaction. Even if store traffic is equal, a $250 item generates vastly more revenue than a $30 item. |
| **How did you fix 0.0 visibility?** | In a physical store, items displayed on shelves cannot have 0% visibility. Zeroes were treated as data recording errors and imputed using the overall non-zero mean. |
| **What happens if a user enters an unknown outlet?** | The preprocessing pipeline implements safe fallbacks, mapping unseen categories to the modal category to prevent runtime crashes. |

---

## 🔧 Troubleshooting & FAQs

- **Port already in use?**
  Run: `streamlit run app.py --server.port 8502`
- **ModuleNotFoundError?**
  Ensure your virtual environment is active: `.\.venv\Scripts\Activate.ps1`, then run `pip install -r requirements.txt`.
- **Cached data refresh?**
  Press `C` in your browser while viewing the Streamlit app to clear the cache, or click the hamburger menu `⋮` > **Clear cache**.
