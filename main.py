import streamlit as st
import pandas as pd
import joblib
import sklearn.compose._column_transformer as column_transformer


if not hasattr(column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        def __setstate__(self, state):
            self.__dict__.update(state)
            self.extend(state.get("data", []))

    column_transformer._RemainderColsList = _RemainderColsList


model = joblib.load("best_xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
Predict whether a telecom customer is likely to churn using an
**XGBoost Machine Learning Model**.

Fill in the customer details below and click **Predict Churn**.
""")

st.divider()

left, right = st.columns(2)

with left:
    gender = st.selectbox("Gender", ["Male", "Female"])

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"],
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"],
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"],
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12,
    )

    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"],
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
    )

with right:
    security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"],
    )

    backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"],
    )

    protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"],
    )

    support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"],
    )

    tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"],
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"],
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0,
    )


if st.button("🔍 Predict Churn", use_container_width=True):
    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone],
        "MultipleLines": [multiple],
        "InternetService": [internet],
        "OnlineSecurity": [security],
        "OnlineBackup": [backup],
        "DeviceProtection": [protection],
        "TechSupport": [support],
        "StreamingTV": [tv],
        "StreamingMovies": [movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total],
    })

    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    st.divider()

    st.header("📈 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠ Customer is likely to Churn")
        else:
            st.success("✅ Customer is likely to Stay")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%",
        )

    with col2:
        st.progress(float(probability))

        if probability > 0.75:
            st.warning("High Risk Customer")
        elif probability > 0.50:
            st.info("Medium Risk Customer")
        else:
            st.success("Low Risk Customer")

    st.divider()

    st.subheader("💼 Business Recommendation")

    if prediction == 1:
        st.warning("""
• Offer loyalty discounts

• Recommend a long-term contract

• Provide personalized customer support

• Offer bundled services to improve retention
""")
    else:
        st.success("""
• Customer is likely to stay

• Continue providing quality service

• Recommend premium plans

• Maintain regular engagement
""")


st.sidebar.title("📊 Model Information")

st.sidebar.markdown("""
### Model

✅ XGBoost Classifier

### Dataset

IBM Telco Customer Churn

### Features

19 Input Features

### Objective

Predict whether a customer is likely to churn.
""")

st.divider()

st.caption(
    "Developed by **KONDRAPU KARTHIK** | National Institute of Technology Karnataka"
)
