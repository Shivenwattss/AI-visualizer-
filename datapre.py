import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Preprocessing Visualizer",
    layout="wide"
)

st.title("AI Preprocessing Visualizer")


# =====================================================
# SIDEBAR CONTROLS
# =====================================================

st.sidebar.header("Dataset Controls")

num_points = st.sidebar.slider(
    "Number of Points",
    20,
    200,
    60
)

noise_sigma = st.sidebar.slider(
    "Noise σ",
    0.1,
    5.0,
    1.2
)

outlier_rate = st.sidebar.slider(
    "Outlier Rate %",
    0,
    30,
    8
)

missing_rate = st.sidebar.slider(
    "Missing Rate %",
    0,
    30,
    10
)


# =====================================================
# IMPUTATION METHOD
# =====================================================

st.subheader("Imputation")

imputation_method = st.radio(
    "Choose Imputation Method",
    ["raw", "mean", "median"],
    horizontal=True
)


# =====================================================
# OUTLIER DETECTION METHOD
# =====================================================

st.subheader("Outlier Detection")

outlier_method = st.radio(
    "Choose Outlier Detection",
    ["off", "IQR x1.5", "z > 2.5"],
    horizontal=True
)


# =====================================================
# NORMALIZATION METHOD
# =====================================================

st.subheader("Normalization")

normalization_method = st.radio(
    "Choose Normalization",
    ["none", "z-score", "robust scaler"],
    horizontal=True
)


# =====================================================
# DATA GENERATION
# =====================================================

np.random.seed()

data = np.random.normal(
    loc=50,
    scale=10,
    size=num_points
)

# Add noise
noise = np.random.normal(
    0,
    noise_sigma,
    num_points
)

data = data + noise


# =====================================================
# ADD OUTLIERS
# =====================================================

num_outliers = int(
    (outlier_rate / 100) * num_points
)

outlier_indices = np.random.choice(
    num_points,
    num_outliers,
    replace=False
)

data[outlier_indices] = (
    data[outlier_indices] * 3
)


# =====================================================
# ADD MISSING VALUES
# =====================================================

num_missing = int(
    (missing_rate / 100) * num_points
)

missing_indices = np.random.choice(
    num_points,
    num_missing,
    replace=False
)

data[missing_indices] = np.nan


# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(
    data,
    columns=["Data"]
)


# =====================================================
# IMPUTATION
# =====================================================

if imputation_method == "raw":

    df_processed = df.copy()

elif imputation_method == "mean":

    imputer = SimpleImputer(strategy="mean")

    df_processed = pd.DataFrame(
        imputer.fit_transform(df),
        columns=["Data"]
    )

elif imputation_method == "median":

    imputer = SimpleImputer(strategy="median")

    df_processed = pd.DataFrame(
        imputer.fit_transform(df),
        columns=["Data"]
    )


# =====================================================
# OUTLIER DETECTION
# =====================================================

outliers = pd.DataFrame()

if outlier_method == "IQR x1.5":

    Q1 = df_processed["Data"].quantile(0.25)

    Q3 = df_processed["Data"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df_processed[
        (df_processed["Data"] < lower)
        |
        (df_processed["Data"] > upper)
    ]

elif outlier_method == "z > 2.5":

    mean = df_processed["Data"].mean()

    std = df_processed["Data"].std()

    z_scores = (
        (df_processed["Data"] - mean) / std
    )

    outliers = df_processed[
        abs(z_scores) > 2.5
    ]


# =====================================================
# NORMALIZATION
# =====================================================

scaled_data = df_processed.copy()

if normalization_method == "z-score":

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        df_processed.fillna(
            df_processed.mean()
        )
    )

    scaled_data = pd.DataFrame(
        scaled,
        columns=["Data"]
    )

elif normalization_method == "robust scaler":

    scaler = RobustScaler()

    scaled = scaler.fit_transform(
        df_processed.fillna(
            df_processed.mean()
        )
    )

    scaled_data = pd.DataFrame(
        scaled,
        columns=["Data"]
    )


# =====================================================
# PLOTLY VISUALIZATION
# =====================================================

fig = go.Figure()

# Main line
fig.add_trace(
    go.Scatter(
        x=list(range(num_points)),
        y=scaled_data["Data"],
        mode="lines+markers",
        name="observed",
        line=dict(color="cyan"),
        marker=dict(size=8)
    )
)

# Missing values
missing_y = []

for i in range(num_points):

    if i in missing_indices:

        missing_y.append(
            scaled_data["Data"].mean()
        )

    else:

        missing_y.append(None)

fig.add_trace(
    go.Scatter(
        x=list(range(num_points)),
        y=missing_y,
        mode="markers",
        name="missing",
        marker=dict(
            color="gray",
            size=10
        )
    )
)

# Outliers
if len(outliers) > 0:

    fig.add_trace(
        go.Scatter(
            x=outliers.index,
            y=outliers["Data"],
            mode="markers",
            name="flagged outlier",
            marker=dict(
                color="red",
                size=14,
                symbol="circle-open"
            )
        )
    )

# Layout
fig.update_layout(
    template="plotly_dark",
    height=500,
    title="Interactive Dataset Visualization",
    xaxis_title="Index",
    yaxis_title="Value"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =====================================================
# DATA TABLES
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Processed Dataset")

    st.dataframe(df_processed)

with col2:

    st.subheader("Detected Outliers")

    st.dataframe(outliers)


# =====================================================
# SUMMARY
# =====================================================

st.subheader("Summary")

st.write(f"Noise Sigma: {noise_sigma}")

st.write(f"Outlier Rate: {outlier_rate}%")

st.write(f"Missing Rate: {missing_rate}%")

st.write(f"Imputation Method: {imputation_method}")

st.write(f"Outlier Detection: {outlier_method}")

st.write(f"Normalization: {normalization_method}")

