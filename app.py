import streamlit as st
import requests

# Frontend
st.title("Green AutoML: User-Controlled Search Space")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    st.success("Dataset uploaded successfully!")
    # Allow dataset preview
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

# Step 2: Algorithm Selection
st.subheader("Select Algorithms for Search Space")

algorithms = ["Random Forest", "XGBoost", "LightGBM", "SVM", "KNN"]
selected_algorithms = st.multiselect("Choose algorithms:", algorithms, default=algorithms)

# Step 3: Submit
if st.button("Run AutoML"):
    if uploaded_file and selected_algorithms:
        st.info("Running AutoML with selected algorithms...")
        # Send data to backend
        response = requests.post(
            "http://127.0.0.1:5000/run_automl",
            json={"algorithms": selected_algorithms, "data": df.to_json()}
        )
        if response.status_code == 200:
            st.success("AutoML completed successfully!")
            st.write("Results:")
            st.json(response.json())
        else:
            st.error("An error occurred while running AutoML.")
    else:
        st.warning("Please upload a dataset and select at least one algorithm.")
