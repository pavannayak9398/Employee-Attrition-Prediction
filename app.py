# Streamlit entry point
import streamlit as st
import pandas as pd
from core.train import train_lightgbm
from utils.constants import TARGET_COL
from utils.registry_utils import list_models, load_model
from core.predict import predict_batch
from core.explain import get_shap_explainer
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Attrition POC", layout="wide")

tab1, tab2 = st.tabs(["Train Model", "Predict"])

with tab1:
    st.header("Train Employee Attrition Model")

    uploaded_file = st.file_uploader(
        "Upload training CSV",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df.head())

        st.subheader("Dataset Overview")
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))

        if TARGET_COL not in df.columns:
            st.error(f"Target column '{TARGET_COL}' not found!")
        else:
            st.subheader("Target Distribution")
            st.bar_chart(df[TARGET_COL].value_counts())
            
            st.subheader("Data Transformation Steps")
            st.markdown("""
                The following preprocessing steps will be applied before model training:

                1. **Target Encoding**
                - `Attrition` column mapped:  
                - Yes → 1  
                - No → 0  

                2. **Feature Separation**
                - Numerical and categorical features are automatically detected.

                3. **Numerical Feature Processing**
                - Missing values (if any) are handled.
                - Features are standardized using `StandardScaler`.

                4. **Categorical Feature Processing**
                - Categorical variables are encoded using `OneHotEncoder`.
                - Unknown categories during inference are safely ignored.

                5. **Train-Test Split**
                - Dataset is split into 80% training and 20% testing.
                - Stratified split to preserve attrition class balance.

                6. **Model Training**
                - LightGBM classifier trained with class imbalance handling.

                7. **Model Registration**
                - Trained model, preprocessor, and metadata are saved to the model registry.
                """)


            if st.button("Start Training"):
                with st.spinner("Training model..."):
                    metrics, model_dir = train_lightgbm(df)

                st.success("Training completed!")

                st.markdown("### Results")
                st.write("Accuracy:", metrics["accuracy"])
                st.write("ROC-AUC:", metrics["roc_auc"])
                st.write("Model saved at:", model_dir)

with tab2:
    st.header("Predict Employee Attrition")

    models = list_models()

    if not models:
        st.warning("No trained models found. Please train a model first.")
        st.stop()

    model_options = {
        f"{m['name']} | {m['created_at']} | ROC-AUC: {m['roc_auc']}": m["id"]
        for m in models
    }

    selected_label = st.selectbox(
        "Select a trained model",
        list(model_options.keys())
    )

    selected_model_id = model_options[selected_label]

    model, preprocessor, metadata = load_model(selected_model_id)

    st.subheader("Model Metadata")
    st.json(metadata)

    test_file = st.file_uploader(
        "Upload test CSV (without Attrition column)",
        type=["csv"],
        key="predict_upload"
    )

    if test_file:
        df_test = pd.read_csv(test_file)

        st.subheader("Test Data Preview")
        st.dataframe(df_test.head())

        if st.button("Run Prediction"):
            with st.spinner("Generating predictions..."):
                # ---------- Prediction ----------
                predictions = predict_batch(
                    df_test,
                    model,
                    preprocessor
                )

            st.success("Prediction completed!")

            st.subheader("Prediction Results")
            st.dataframe(predictions.head(20))

            csv = predictions.to_csv(index=False).encode()
            st.download_button(
                "Download Predictions",
                csv,
                "attrition_predictions.csv",
                "text/csv"
            )

            # ---------- SHAP ----------
            st.subheader("Global Feature Importance (SHAP)")

            # Transform features
            X_trans = preprocessor.transform(df_test)

            # Extract feature names
            feature_names = preprocessor.get_feature_names_out()

            # SHAP explainer (old-version safe)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)

            # Use class 1 = Attrition (Leave)
            shap_matrix = shap_values[1]

            fig = plt.figure()
            shap.summary_plot(
                shap_matrix,
                X_trans,
                feature_names=feature_names,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig)

            # ---------- Local Explanation ----------
            st.subheader("Explain Single Employee")

            idx = st.number_input(
                "Select row index",
                min_value=0,
                max_value=len(df_test) - 1,
                value=0
            )

            fig2 = plt.figure()
            shap.waterfall_plot(
                explainer.expected_value[1],
                shap_values[1][idx],
                feature_names=feature_names,
                show=False
            )
            st.pyplot(fig2)
