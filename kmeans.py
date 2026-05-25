import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="KMeans Visualizer",
    layout="wide"
)

st.title("KMeans Clustering Visualizer")

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("KMeans Controls")

k=st.sidebar.slider(
    "Number of Clusters (k)",
    2,
    8,
    3
)

num_points=st.sidebar.slider(
    "Number of Points",
    50,
    500,
    150
)

cluster_spread=st.sidebar.slider(
    "Cluster Spread",
    0.2,
    3.0,
    1.0
)

show_centroids=st.sidebar.checkbox(
    "Show Centroids",
    True
)

# =========================================================
# DATA GENERATION
# =========================================================

np.random.seed()

cluster1=np.random.randn(
    num_points//3,
    2
)*cluster_spread+[2,2]

cluster2=np.random.randn(
    num_points//3,
    2
)*cluster_spread+[7,7]

cluster3=np.random.randn(
    num_points//3,
    2
)*cluster_spread+[12,3]

X=np.vstack([
    cluster1,
    cluster2,
    cluster3
])

# =========================================================
# KMEANS MODEL
# =========================================================

kmeans=KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels=kmeans.fit_predict(X)

centroids=kmeans.cluster_centers_

# =========================================================
# VISUALIZATION
# =========================================================

fig=go.Figure()

# Data points
fig.add_trace(
    go.Scatter(
        x=X[:,0],
        y=X[:,1],
        mode="markers",
        name="Data Points",
        marker=dict(
            size=8,
            color=labels,
            colorscale="Viridis"
        )
    )
)

# Centroids
if show_centroids:

    fig.add_trace(
        go.Scatter(
            x=centroids[:,0],
            y=centroids[:,1],
            mode="markers+text",
            name="Centroids",
            text=[
                f"C{i}" for i in range(k)
            ],
            textposition="top center",
            marker=dict(
                color="red",
                size=22,
                symbol="x"
            )
        )
    )

# Layout
fig.update_layout(
    template="plotly_dark",
    height=700,
    title="KMeans Clustering",
    xaxis_title="Feature 1",
    yaxis_title="Feature 2"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# KMEANS EXPLANATION
# =========================================================

st.subheader("How KMeans Works")

st.write("""
1. Random centroids are initialized

2. Each point is assigned to nearest centroid

3. Centroids move to cluster mean

4. Process repeats until convergence
""")

# =========================================================
# CENTROID TABLE
# =========================================================

st.subheader("Centroid Coordinates")

centroid_df=pd.DataFrame(
    centroids,
    columns=["X","Y"]
)

centroid_df.index=[
    f"C{i}" for i in range(k)
]

st.dataframe(centroid_df)

# =========================================================
# CLUSTERED DATASET
# =========================================================

st.subheader("Clustered Dataset")

cluster_df=pd.DataFrame(
    X,
    columns=["Feature1","Feature2"]
)

cluster_df["Cluster"]=labels

st.dataframe(cluster_df)