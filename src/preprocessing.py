import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

# Exact feature names expected by the trained CatBoost model in Pickle/Model.pkl
CATBOOST_FEATURE_NAMES: List[str] = [
    'Item_Weight',
    'Item_Visibility',
    'Item_Type',
    'Item_MRP',
    'Outlet_Years',
    'Outlet',
    'Item_Fat_Content_0',
    'Item_Fat_Content_1',
    'Item_Fat_Content_2',
    'Outlet_Size_0',
    'Outlet_Size_1',
    'Outlet_Size_2',
    'Outlet_Location_Type_0',
    'Outlet_Location_Type_1',
    'Outlet_Location_Type_2',
    'Outlet_Type_0',
    'Outlet_Type_1',
    'Outlet_Type_2',
    'Outlet_Type_3',
    'New_Item_Type_0',
    'New_Item_Type_1',
    'New_Item_Type_2'
]

# Exact label encoder classes learned from the training dataset
OUTLET_IDENTIFIERS: List[str] = [
    'OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019',
    'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'
]

ITEM_TYPES: List[str] = [
    'Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy',
    'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks',
    'Health and Hygiene', 'Household', 'Meat', 'Others',
    'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'
]

OUTLET_SIZES: List[str] = ['High', 'Medium', 'Small']
OUTLET_LOCATION_TYPES: List[str] = ['Tier 1', 'Tier 2', 'Tier 3']
OUTLET_TYPES: List[str] = [
    'Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'
]
ITEM_FAT_CONTENTS: List[str] = ['Low Fat', 'Regular']

# Global fallback defaults derived from Train.csv statistics
GLOBAL_MEAN_WEIGHT = 12.857645
GLOBAL_MEAN_VISIBILITY = 0.066132

# Mode Outlet_Size per Outlet_Type for imputation (from training notebook)
OUTLET_SIZE_MODE_MAP = {
    'Grocery Store': 'Small',
    'Supermarket Type1': 'Small',
    'Supermarket Type2': 'Medium',
    'Supermarket Type3': 'Medium'
}

# Outlet metadata lookup to auto-populate outlet properties if user selects an outlet
OUTLET_METADATA = {
    'OUT010': {'Establishment_Year': 1998, 'Size': 'Small', 'Location_Type': 'Tier 3', 'Type': 'Grocery Store'},
    'OUT013': {'Establishment_Year': 1987, 'Size': 'High', 'Location_Type': 'Tier 3', 'Type': 'Supermarket Type1'},
    'OUT017': {'Establishment_Year': 2007, 'Size': 'Small', 'Location_Type': 'Tier 2', 'Type': 'Supermarket Type1'},
    'OUT018': {'Establishment_Year': 2009, 'Size': 'Medium', 'Location_Type': 'Tier 3', 'Type': 'Supermarket Type2'},
    'OUT019': {'Establishment_Year': 1985, 'Size': 'Small', 'Location_Type': 'Tier 1', 'Type': 'Grocery Store'},
    'OUT027': {'Establishment_Year': 1985, 'Size': 'Medium', 'Location_Type': 'Tier 3', 'Type': 'Supermarket Type3'},
    'OUT035': {'Establishment_Year': 2004, 'Size': 'Small', 'Location_Type': 'Tier 2', 'Type': 'Supermarket Type1'},
    'OUT045': {'Establishment_Year': 2002, 'Size': 'Small', 'Location_Type': 'Tier 2', 'Type': 'Supermarket Type1'},
    'OUT046': {'Establishment_Year': 1997, 'Size': 'Small', 'Location_Type': 'Tier 1', 'Type': 'Supermarket Type1'},
    'OUT049': {'Establishment_Year': 1999, 'Size': 'Medium', 'Location_Type': 'Tier 1', 'Type': 'Supermarket Type1'}
}

# Item type category mapping
ITEM_TYPE_TO_CATEGORY = {
    'Baking Goods': 'Food',
    'Breads': 'Food',
    'Breakfast': 'Food',
    'Canned': 'Food',
    'Dairy': 'Food',
    'Frozen Foods': 'Food',
    'Fruits and Vegetables': 'Food',
    'Hard Drinks': 'Drinks',
    'Health and Hygiene': 'Non-Consumables',
    'Household': 'Non-Consumables',
    'Meat': 'Food',
    'Others': 'Non-Consumables',
    'Seafood': 'Food',
    'Snack Foods': 'Food',
    'Soft Drinks': 'Drinks',
    'Starchy Foods': 'Food'
}

def derive_new_item_type(item_identifier: str, item_type: str = "") -> str:
    """Derives high-level category (Drinks, Food, Non-Consumables)."""
    if item_identifier and len(str(item_identifier).strip()) >= 2:
        prefix = str(item_identifier).strip()[:2].upper()
        if prefix == 'FD':
            return 'Food'
        elif prefix == 'DR':
            return 'Drinks'
        elif prefix == 'NC':
            return 'Non-Consumables'
    # Fallback to item_type
    return ITEM_TYPE_TO_CATEGORY.get(item_type, 'Food')

def preprocess_single_input(data: Dict[str, Any], reference_year: int = 2024) -> pd.DataFrame:
    """
    Transforms a single user-input dictionary from the prediction form
    into a 1x22 DataFrame matching the exact CatBoost features and order.
    """
    item_id = str(data.get("Item_Identifier", "FDA15")).strip()
    item_type = str(data.get("Item_Type", "Dairy")).strip()
    new_item_type = derive_new_item_type(item_id, item_type)

    # Item Fat Content
    fat_content = str(data.get("Item_Fat_Content", "Low Fat")).strip()
    if fat_content in ['LF', 'low fat']:
        fat_content = 'Low Fat'
    elif fat_content == 'reg':
        fat_content = 'Regular'

    # If Non-Consumables, rule from training: mark as Non-Edible
    if new_item_type == 'Non-Consumables':
        fat_content = 'Non-Edible'

    # Item Weight
    try:
        weight = float(data.get("Item_Weight", GLOBAL_MEAN_WEIGHT))
        if pd.isna(weight) or weight <= 0:
            weight = GLOBAL_MEAN_WEIGHT
    except (ValueError, TypeError):
        weight = GLOBAL_MEAN_WEIGHT

    # Item Visibility
    try:
        visibility = float(data.get("Item_Visibility", GLOBAL_MEAN_VISIBILITY))
        if pd.isna(visibility) or visibility <= 0:
            visibility = GLOBAL_MEAN_VISIBILITY
    except (ValueError, TypeError):
        visibility = GLOBAL_MEAN_VISIBILITY

    # Item MRP
    try:
        mrp = float(data.get("Item_MRP", 140.0))
    except (ValueError, TypeError):
        mrp = 140.0

    # Outlet Identifier
    outlet_id = str(data.get("Outlet_Identifier", "OUT049")).strip()
    if outlet_id in OUTLET_IDENTIFIERS:
        outlet_code = OUTLET_IDENTIFIERS.index(outlet_id)
    else:
        outlet_code = 0

    # Item Type code (LabelEncoder integer)
    if item_type in ITEM_TYPES:
        item_type_code = ITEM_TYPES.index(item_type)
    else:
        item_type_code = 0

    # Outlet Establishment Year & Outlet Years
    try:
        est_year = int(data.get("Outlet_Establishment_Year", 1999))
    except (ValueError, TypeError):
        est_year = 1999
    outlet_years = float(reference_year - est_year)

    # Outlet Size
    outlet_size = str(data.get("Outlet_Size", "")).strip()
    outlet_type = str(data.get("Outlet_Type", "Supermarket Type1")).strip()
    if not outlet_size or outlet_size not in OUTLET_SIZES:
        outlet_size = OUTLET_SIZE_MODE_MAP.get(outlet_type, 'Small')

    # Outlet Location Type
    outlet_loc = str(data.get("Outlet_Location_Type", "Tier 1")).strip()
    if outlet_loc not in OUTLET_LOCATION_TYPES:
        outlet_loc = 'Tier 1'

    # Build dummy variables
    # Item_Fat_Content: ['Low Fat', 'Non-Edible', 'Regular']
    fat_0 = 1.0 if fat_content == 'Low Fat' else 0.0
    fat_1 = 1.0 if fat_content == 'Non-Edible' else 0.0
    fat_2 = 1.0 if fat_content == 'Regular' else 0.0

    # Outlet_Size: ['High', 'Medium', 'Small']
    size_0 = 1.0 if outlet_size == 'High' else 0.0
    size_1 = 1.0 if outlet_size == 'Medium' else 0.0
    size_2 = 1.0 if outlet_size == 'Small' else 0.0

    # Outlet_Location_Type: ['Tier 1', 'Tier 2', 'Tier 3']
    loc_0 = 1.0 if outlet_loc == 'Tier 1' else 0.0
    loc_1 = 1.0 if outlet_loc == 'Tier 2' else 0.0
    loc_2 = 1.0 if outlet_loc == 'Tier 3' else 0.0

    # Outlet_Type: ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']
    type_0 = 1.0 if outlet_type == 'Grocery Store' else 0.0
    type_1 = 1.0 if outlet_type == 'Supermarket Type1' else 0.0
    type_2 = 1.0 if outlet_type == 'Supermarket Type2' else 0.0
    type_3 = 1.0 if outlet_type == 'Supermarket Type3' else 0.0

    # New_Item_Type: ['Drinks', 'Food', 'Non-Consumables']
    new_0 = 1.0 if new_item_type == 'Drinks' else 0.0
    new_1 = 1.0 if new_item_type == 'Food' else 0.0
    new_2 = 1.0 if new_item_type == 'Non-Consumables' else 0.0

    # Assemble dictionary according to exact feature names
    feature_row = {
        'Item_Weight': float(weight),
        'Item_Visibility': float(visibility),
        'Item_Type': float(item_type_code),
        'Item_MRP': float(mrp),
        'Outlet_Years': float(outlet_years),
        'Outlet': float(outlet_code),
        'Item_Fat_Content_0': fat_0,
        'Item_Fat_Content_1': fat_1,
        'Item_Fat_Content_2': fat_2,
        'Outlet_Size_0': size_0,
        'Outlet_Size_1': size_1,
        'Outlet_Size_2': size_2,
        'Outlet_Location_Type_0': loc_0,
        'Outlet_Location_Type_1': loc_1,
        'Outlet_Location_Type_2': loc_2,
        'Outlet_Type_0': type_0,
        'Outlet_Type_1': type_1,
        'Outlet_Type_2': type_2,
        'Outlet_Type_3': type_3,
        'New_Item_Type_0': new_0,
        'New_Item_Type_1': new_1,
        'New_Item_Type_2': new_2
    }

    df_out = pd.DataFrame([feature_row])[CATBOOST_FEATURE_NAMES]
    return df_out

def preprocess_batch_dataframe(df_raw: pd.DataFrame, reference_year: int = 2024) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the exact cleaning, encoding, and feature alignment to a raw DataFrame.
    Returns:
        (df_features, df_cleaned_display)
    """
    df = df_raw.copy()

    # Weight imputation
    if 'Item_Weight' in df.columns:
        if 'Item_Identifier' in df.columns:
            weight_mean_pivot = df.pivot_table(values='Item_Weight', index='Item_Identifier')
            miss_weight = df['Item_Weight'].isnull()
            for i in df[miss_weight].index:
                item_code = df.loc[i, 'Item_Identifier']
                if item_code in weight_mean_pivot.index:
                    df.loc[i, 'Item_Weight'] = weight_mean_pivot.loc[item_code, 'Item_Weight']
                else:
                    df.loc[i, 'Item_Weight'] = GLOBAL_MEAN_WEIGHT
        else:
            df['Item_Weight'] = df['Item_Weight'].fillna(GLOBAL_MEAN_WEIGHT)
    else:
        df['Item_Weight'] = GLOBAL_MEAN_WEIGHT

    # Outlet Size imputation
    if 'Outlet_Size' in df.columns and 'Outlet_Type' in df.columns:
        miss_size = df['Outlet_Size'].isnull()
        df.loc[miss_size, 'Outlet_Size'] = df.loc[miss_size, 'Outlet_Type'].apply(
            lambda x: OUTLET_SIZE_MODE_MAP.get(str(x), 'Small')
        )
    elif 'Outlet_Size' not in df.columns:
        df['Outlet_Size'] = 'Small'

    # Visibility 0 replacement
    if 'Item_Visibility' in df.columns:
        vis_mean = df['Item_Visibility'][df['Item_Visibility'] > 0].mean()
        if pd.isna(vis_mean) or vis_mean == 0:
            vis_mean = GLOBAL_MEAN_VISIBILITY
        df['Item_Visibility'] = df['Item_Visibility'].replace(0, vis_mean).fillna(vis_mean)
    else:
        df['Item_Visibility'] = GLOBAL_MEAN_VISIBILITY

    # Fat Content cleaning
    if 'Item_Fat_Content' in df.columns:
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
            'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
        })
    else:
        df['Item_Fat_Content'] = 'Low Fat'

    # New_Item_Type derivation
    if 'Item_Identifier' in df.columns:
        df['New_Item_Type'] = df['Item_Identifier'].astype(str).str[:2].replace({
            'FD': 'Food', 'NC': 'Non-Consumables', 'DR': 'Drinks'
        })
    elif 'Item_Type' in df.columns:
        df['New_Item_Type'] = df['Item_Type'].map(ITEM_TYPE_TO_CATEGORY).fillna('Food')
    else:
        df['New_Item_Type'] = 'Food'

    # Mark Non-Consumables as Non-Edible fat content
    df.loc[df['New_Item_Type'] == 'Non-Consumables', 'Item_Fat_Content'] = 'Non-Edible'

    # Outlet Years
    if 'Outlet_Establishment_Year' in df.columns:
        df['Outlet_Years'] = reference_year - df['Outlet_Establishment_Year']
    else:
        df['Outlet_Years'] = 15.0

    # Save cleaned copy for display / analysis
    df_cleaned = df.copy()

    # Outlet Label Encoding
    if 'Outlet_Identifier' in df.columns:
        df['Outlet'] = df['Outlet_Identifier'].apply(
            lambda x: OUTLET_IDENTIFIERS.index(x) if x in OUTLET_IDENTIFIERS else 0
        )
    else:
        df['Outlet'] = 0

    # Item Type Label Encoding
    if 'Item_Type' in df.columns:
        df['Item_Type'] = df['Item_Type'].apply(
            lambda x: ITEM_TYPES.index(x) if x in ITEM_TYPES else 0
        )
    else:
        df['Item_Type'] = 0

    # One-hot encoding
    # Item_Fat_Content_0 (Low Fat), 1 (Non-Edible), 2 (Regular)
    df['Item_Fat_Content_0'] = (df['Item_Fat_Content'] == 'Low Fat').astype(float)
    df['Item_Fat_Content_1'] = (df['Item_Fat_Content'] == 'Non-Edible').astype(float)
    df['Item_Fat_Content_2'] = (df['Item_Fat_Content'] == 'Regular').astype(float)

    # Outlet_Size_0 (High), 1 (Medium), 2 (Small)
    df['Outlet_Size_0'] = (df['Outlet_Size'] == 'High').astype(float)
    df['Outlet_Size_1'] = (df['Outlet_Size'] == 'Medium').astype(float)
    df['Outlet_Size_2'] = (df['Outlet_Size'] == 'Small').astype(float)

    # Outlet_Location_Type_0 (Tier 1), 1 (Tier 2), 2 (Tier 3)
    df['Outlet_Location_Type_0'] = (df['Outlet_Location_Type'] == 'Tier 1').astype(float)
    df['Outlet_Location_Type_1'] = (df['Outlet_Location_Type'] == 'Tier 2').astype(float)
    df['Outlet_Location_Type_2'] = (df['Outlet_Location_Type'] == 'Tier 3').astype(float)

    # Outlet_Type_0 (Grocery Store), 1 (Supermarket Type1), 2 (Supermarket Type2), 3 (Supermarket Type3)
    df['Outlet_Type_0'] = (df['Outlet_Type'] == 'Grocery Store').astype(float)
    df['Outlet_Type_1'] = (df['Outlet_Type'] == 'Supermarket Type1').astype(float)
    df['Outlet_Type_2'] = (df['Outlet_Type'] == 'Supermarket Type2').astype(float)
    df['Outlet_Type_3'] = (df['Outlet_Type'] == 'Supermarket Type3').astype(float)

    # New_Item_Type_0 (Drinks), 1 (Food), 2 (Non-Consumables)
    df['New_Item_Type_0'] = (df['New_Item_Type'] == 'Drinks').astype(float)
    df['New_Item_Type_1'] = (df['New_Item_Type'] == 'Food').astype(float)
    df['New_Item_Type_2'] = (df['New_Item_Type'] == 'Non-Consumables').astype(float)

    # Ensure all required columns exist with float types
    for col in CATBOOST_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    features_df = df[CATBOOST_FEATURE_NAMES].astype(float)
    return features_df, df_cleaned

def validate_uploaded_csv(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Validates an uploaded CSV dataframe against the expected columns.
    Returns:
        (is_valid, missing_columns, warnings)
    """
    essential_columns = [
        'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
        'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
    ]
    missing = [c for c in essential_columns if c not in df.columns]
    warnings = []

    if 'Item_Outlet_Sales' not in df.columns:
        warnings.append(
            "Column 'Item_Outlet_Sales' not found. Ground-truth comparison and R² evaluation will be disabled for this upload."
        )

    if 'Item_Identifier' not in df.columns:
        warnings.append(
            "Column 'Item_Identifier' not found. Categories will be inferred from 'Item_Type'."
        )

    is_valid = len(missing) == 0
    return is_valid, missing, warnings
