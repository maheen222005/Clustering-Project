
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="P712 World Development Clustering",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------
# 2. Load Saved Models
# ----------------------------------

@st.cache_resource
def load_models():
    imputer = joblib.load(BASE_DIR / "imputer.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    pca = joblib.load(BASE_DIR / "pca.pkl")
    model = joblib.load(BASE_DIR / "kmeans_model.pkl")
    features = joblib.load(BASE_DIR / "feature_names.pkl")

    return imputer, scaler, pca, model, features

try:
    imputer, scaler, pca, model, features = load_models()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()

# ----------------------------------
# 3. App Header
# ----------------------------------

st.title("🌍 World Development Clustering")
st.subheader("P712 - Unsupervised Machine Learning Project")

st.write(
    "Explore development indicators and predict "
    "the cluster assigned by the trained K-Means model."
)

st.info(
    "Enter values using the same units and feature definitions "
    "as the dataset used to train the model."
)

# ----------------------------------
# 4. Sidebar
# ----------------------------------

st.sidebar.header("Project Information")
st.sidebar.write("**Project:** P712 Clustering")
st.sidebar.write("**Algorithm:** K-Means")
st.sidebar.write("**Learning Type:** Unsupervised Learning")
st.sidebar.write(f"**Features:** {len(features)}")
st.sidebar.write(f"**Clusters:** {model.n_clusters}")

page = st.sidebar.radio(
    "Choose Page",
    ["Cluster Prediction", "Model Information"]
)

# ----------------------------------
# 5. Cluster Prediction Page
# ----------------------------------

if page == "Cluster Prediction":

    st.header("🔍 Predict Development Cluster")

    st.write(
        "Enter the numerical values for the development indicators."
    )

    with st.form("prediction_form"):

        user_values = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):

            with col1 if i % 2 == 0 else col2:

                user_values[feature] = st.number_input(
                    label=str(feature),
                    value=0.0,
                    format="%.4f",
                    key=f"input_{i}"
                )

        submitted = st.form_submit_button(
            "Predict Cluster",
            type="primary"
        )

    if submitted:

        input_df = pd.DataFrame(
            [[user_values[f] for f in features]],
            columns=features
        )

        try:
            # Use the same preprocessing order as training
            imputed = imputer.transform(input_df)
            scaled = scaler.transform(imputed)
            reduced = pca.transform(scaled)

            prediction = model.predict(reduced)

            cluster = int(prediction[0])

            st.success(f"Predicted Cluster: {cluster}")

            st.subheader("Input Summary")
            st.dataframe(input_df, use_container_width=True)

            st.caption(
                "Cluster numbers are identifiers, not development rankings."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------------------------
# 6. Model Information Page
# ----------------------------------

elif page == "Model Information":

    st.header("📊 Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithm", "K-Means")
    col2.metric("Number of Clusters", model.n_clusters)
    col3.metric("Input Features", len(features))

    st.subheader("Features Used")

    st.write(features)

    st.subheader("Preprocessing Pipeline")

    st.markdown(
        """
        1. Missing value handling using the saved imputer
        2. Feature scaling using the saved StandardScaler
        3. Dimensionality reduction using the saved PCA
        4. Cluster prediction using the saved K-Means model
        """
    )

    st.subheader("Cluster Centroids")

    centroids = pd.DataFrame(
        model.cluster_centers_,
        columns=[
            f"PC{i + 1}"
            for i in range(model.cluster_centers_.shape[1])
        ]
    )

    st.dataframe(centroids, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))

    for i, row in centroids.iterrows():
        ax.plot(
            row.index,
            row.values,
            marker="o",
            label=f"Cluster {i}"
        )

    ax.set_title("K-Means Centroids in PCA Space")
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Centroid Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------
# 7. Footer
# ----------------------------------

st.divider()

st.caption(
    "P712 - World Development Clustering | "
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)