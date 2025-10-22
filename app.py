# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 13:25:12 2025
@author: alexi (modified)
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# Load model, threshold, and mean/std CSV
# =========================
model = joblib.load("model/best_rf.pkl")
threshold_data = joblib.load("model/threshold.pkl")
best_thresh = threshold_data.get("best_threshold")

# Precomputed means and stds (kept from your original file)
feature_means = {
    "alcohol_x": 5.453754451,
    "cholesterol": 181.5073331,
    "moderate_pa": 59.60793792,
    "vigorous_pa": 49.6231405,
    "height": 66.41341194,
    "weight": 179.0332633,
    "calories": 1930.070135,
    "protein": 71.01640037,
    "carbs": 224.9348537,
    "sugar": 97.23631057,
    "fat": 80.24344171,
    "alcohol_y": 5.453754451,
    "insulin": 87.65179003,
    "sleep": 7.760957892,
}

feature_stds = {
    "alcohol_x": 16.24982132,
    "cholesterol": 39.06219388,
    "moderate_pa": 737.6031039,
    "vigorous_pa": 687.960533,
    "height": 4.168204693,
    "weight": 47.35433188,
    "calories": 776.9697004,
    "protein": 33.45953841,
    "carbs": 99.22542613,
    "sugar": 58.52259581,
    "fat": 39.05626193,
    "alcohol_y": 16.24982132,
    "insulin": 148.704706,
    "sleep": 1.605230055,
}

df = pd.read_csv("health_not_null2.csv")

# add a few derived / dataset-driven defaults as in original
feature_means['age'] = df['age'].mean()
feature_stds['age'] = df['age'].std()
feature_means['avg_drinks_day'] = 1.684867
feature_stds['avg_drinks_day'] = 2.180017
feature_means['physical_activity'] = 195.58988503671603
feature_stds['physical_activity'] = 1938.2701871451966
feature_means['smoking'] = df['smoking'].mode()[0]
feature_means['liver'] = df['liver'].mode()[0]
feature_means['heart'] = df['heart'].mode()[0]
feature_means['income'] = df['income'].mode()[0]

# =========================
# Feature definitions and user-friendly labels
# =========================
features = ['cholesterol', 'income', 'insulin', 'smoking', 'calories', 'protein', 'carbs',
            'sugar', 'fat', 'height', 'weight', 'gender', 'age', 'liver', 'heart',
            'sleep', 'avg_drinks_day', 'physical_activity']

# Human-friendly labels (as requested)
feature_labels = {
    "cholesterol": "Cholesterol (mg/dL)",
    "income": "Do you have +20k in savings?",
    "insulin": "Insulin (μU/mL)",
    "smoking": "Have you smoked more that 100 cigarettes in your life?",
    "calories": "Daily Calorie Intake (kcal)",
    "protein": "Protein (g/day)",
    "carbs": "Carbohydrates (g/day)",
    "sugar": "Sugar (g/day)",
    "fat": "Fat (g/day)",
    "height": "Height (cm)",
    "weight": "Weight (kg)",
    "gender": "Gender (0 = Male, 1 = Female)",
    "age": "Age (years)",
    "liver": "Have you had a liver condition?",
    "heart": "Have you had a heart condition?",
    "sleep": "Sleep Duration (hours/night)",
    "avg_drinks_day": "Average Drinks per Day",
    "physical_activity": "Physical Activity (minutes/week)"
}

# =========================
# Disclaimer on top (user requested)
# =========================
st.markdown(
    "<p style='font-size:14px; color:#7a7a7a; border-left: 4px solid #f0ad4e; padding: 10px;'>"
    "<strong>Disclaimer:</strong> This application is for informational and educational purposes only. "
    "It does not provide medical advice, diagnosis, or treatment. Always consult a healthcare professional "
    "for medical concerns and before making changes to your health routine.</p>",
    unsafe_allow_html=True
)

# App title & intro
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter your values below. If you don’t know a value, choose 'I don't know' (where available) or enter -1 for numeric fields.")

# =========================
# Build the input UI
# =========================
patient = {}
cols = st.columns(3)

for i, feature in enumerate(features):
    col = cols[i % 3]
    label = feature_labels.get(feature, feature)

    # Categorical / special-handling features
    if feature == "gender":
        # keep numeric 0/1 (model expects numeric)
        patient[feature] = col.slider(label, 0, 1, 0)

    elif feature in ["smoking", "liver", "heart", "income"]:
        options = ["I don't know", "No", "Yes"]
        choice = col.selectbox(label, options)
        if choice == "Yes":
            patient[feature] = 1
        elif choice == "No":
            patient[feature] = 0
        else:
            # fallback to dataset mode/mean as before
            patient[feature] = feature_means[feature]

    elif feature == 'weight':
        # Label indicates lbs; feature_means is in lbs, feature_stds in lbs -> no unit conversion
        val = col.number_input(label, value=float(feature_means[feature]))
        if val == -1:
            # preserve original handling: use mean/std ratio when unknown
            patient[feature] = feature_means[feature] / feature_stds[feature]
        else:
            patient[feature] = (val - feature_means[feature]) / feature_stds[feature]

    elif feature == 'height':
        # Label indicates inches; feature_means is in inches, feature_stds in inches -> no conversion
        val = col.number_input(label, value=float(feature_means[feature]))
        if val == -1:
            patient[feature] = feature_means[feature] / feature_stds[feature]
        else:
            patient[feature] = (val - feature_means[feature]) / feature_stds[feature]

    else:
        # generic numeric fields
        default_val = feature_means.get(feature, 0.0)
        val = col.number_input(label, value=float(default_val))
        if val == -1:
            # original app used mean/std here when -1 entered
            patient[feature] = feature_means.get(feature, 0.0) / feature_stds.get(feature, 1.0)
        else:
            patient[feature] = (val - feature_means.get(feature, 0.0)) / feature_stds.get(feature, 1.0)

# =========================
# Prediction & display
# =========================
if st.button("Predict"):
    patient_df = pd.DataFrame([patient])[features]
    y_proba = model.predict_proba(patient_df)[:, 1][0]
    prediction = "⚠️ Likely Diabetic" if y_proba >= best_thresh else "✅ Likely Not Diabetic"

    st.metric("Predicted Probability", f"{y_proba:.2f}")
    st.subheader(prediction)

    # Personalized tips section (keeps your original logic)
    if y_proba >= best_thresh:
        st.markdown("### 💡 Health Tips for You")
        st.write(
            """
            - Reduce refined carbs and sugary drinks.  
            - Include fiber-rich foods like oats, lentils, and veggies.  
            - Exercise for at least **30 minutes daily** (walking, cycling, swimming).  
            - Get your blood sugar checked regularly.  
            - Try to maintain **7–8 hours of sleep** per night.
            """
        )
        # check sleep and physical activity relative to normalized inputs
        if 'sleep' in patient_df.columns:
            if patient_df['sleep'].iloc[0] < (7 - feature_means['sleep']) / feature_stds['sleep']:
                st.warning("🛌 You may not be sleeping enough — try aiming for 7–8 hours.")
        if 'physical_activity' in patient_df.columns:
            if patient_df['physical_activity'].iloc[0] < 0:
                st.info("🏃‍♂️ Increasing your physical activity could help lower your risk.")
    else:
        st.markdown("### 💪 Keep It Up!")
        st.write(
            """
            - Your risk looks low — maintain your healthy habits!  
            - Stay active and keep a balanced diet.  
            - Limit sugary snacks and alcohol.  
            - Continue getting regular checkups.
            """
        )

    # Feature importance breakdown (same as original)
    st.markdown("---")
    st.markdown("### 🔍 What influenced your result most?")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

    with st.expander("Show top 5 most important features"):
        st.table(importance_df.head(5))
