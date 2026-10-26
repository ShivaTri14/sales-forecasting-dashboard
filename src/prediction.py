import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.preprocessing import preprocess_single_input, preprocess_batch_dataframe, CATBOOST_FEATURE_NAMES

def predict_single(model, input_data: Dict[str, Any], reference_year: int = 2024) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """
    Takes raw user input dictionary, applies exact preprocessing,
    runs the trained CatBoost model, and returns:
    (predicted_sales, feature_df, explanation_dict)
    """
    features_df = preprocess_single_input(input_data, reference_year=reference_year)
    raw_pred = float(model.predict(features_df)[0])

    # Sales cannot be negative in reality
    predicted_sales = max(0.0, raw_pred)

    # Generate business interpretation
    explanation = generate_prediction_explanation(input_data, predicted_sales)

    return predicted_sales, features_df, explanation

def predict_batch(model, df_raw: pd.DataFrame, reference_year: int = 2024) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """
    Applies preprocessing to batch dataset and runs model prediction.
    Returns:
        (df_with_predictions, metrics_dict_or_None)
    """
    features_df, df_cleaned = preprocess_batch_dataframe(df_raw, reference_year=reference_year)
    raw_preds = model.predict(features_df)
    preds = np.clip(raw_preds, a_min=0.0, a_max=None)

    df_result = df_cleaned.copy()
    df_result["Predicted_Sales"] = np.round(preds, 2)

    metrics = None
    if "Item_Outlet_Sales" in df_raw.columns:
        actuals = df_raw["Item_Outlet_Sales"].values
        # Drop rows where actual is NaN if any
        valid_mask = ~np.isnan(actuals)
        if valid_mask.sum() > 0:
            y_act = actuals[valid_mask]
            y_pred = preds[valid_mask]
            r2 = float(r2_score(y_act, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_act, y_pred)))
            mae = float(mean_absolute_error(y_act, y_pred))
            metrics = {
                "r2_score": r2,
                "rmse": rmse,
                "mae": mae,
                "sample_count": int(valid_mask.sum())
            }
            df_result["Residual"] = np.round(df_result["Item_Outlet_Sales"] - df_result["Predicted_Sales"], 2)

    return df_result, metrics

def generate_prediction_explanation(input_data: Dict[str, Any], predicted_sales: float) -> Dict[str, Any]:
    """
    Constructs an intuitive business breakdown of why the model predicted this sales volume,
    highlighting the dominant features learned by CatBoost.
    """
    mrp = float(input_data.get("Item_MRP", 140.0))
    outlet_type = str(input_data.get("Outlet_Type", "Supermarket Type1"))
    outlet_id = str(input_data.get("Outlet_Identifier", "OUT049"))
    vis = float(input_data.get("Item_Visibility", 0.066))
    est_year = int(input_data.get("Outlet_Establishment_Year", 1999))
    store_age = 2024 - est_year

    insights = []

    # 1. MRP Impact (55.8% Feature Importance)
    if mrp >= 200:
        insights.append(f"**High MRP Tier (${mrp:.2f})**: Significantly elevates total sales potential due to higher unit transaction value.")
    elif mrp <= 70:
        insights.append(f"**Budget MRP Tier (${mrp:.2f})**: Lower unit revenue caps the overall revenue generation per item.")
    else:
        insights.append(f"**Mid-Range MRP (${mrp:.2f})**: Provides steady baseline revenue per unit sold.")

    # 2. Outlet Type Impact (42.9% Combined Importance)
    if outlet_type == "Grocery Store":
        insights.append(
            "**Grocery Store Penalty**: Historical data shows grocery stores generate substantially lower volume (-75% vs Supermarket Type3) due to constrained retail footprint and foot traffic."
        )
    elif outlet_type == "Supermarket Type3":
        insights.append(
            "**Supermarket Type 3 Premium**: High-traffic hypermarket format drives peak sales throughput (+80% above average)."
        )
    else:
        insights.append(
            f"**Standard Supermarket Format ({outlet_type})**: Offers strong consistent distribution and customer base."
        )

    # 3. Store Vintage / Maturity
    if store_age >= 25:
        insights.append(f"**Established Outlet Vintage ({store_age} years old)**: Benefits from high customer loyalty and proven local market demand.")
    elif store_age <= 10:
        insights.append(f"**Modern Outlet ({store_age} years old)**: Newer retail branch experiencing growth ramp-up.")

    # 4. Item Visibility
    if vis < 0.02:
        insights.append(f"**Low Shelf Visibility ({vis:.3f})**: Low prominence in retail aisles; improved shelf placement could boost conversions.")
    elif vis > 0.12:
        insights.append(f"**High Display Prominence ({vis:.3f})**: Prime aisle placement attracts higher customer attention.")

    return {
        "predicted_sales": predicted_sales,
        "primary_driver": f"Item MRP (${mrp:.2f}) & Store Format ({outlet_type})",
        "insights": insights
    }
