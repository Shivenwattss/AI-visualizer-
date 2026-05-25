import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="DBSCAN Visualizer",layout="wide")
st.title("DBSCAN Clustering Visualizer")
st.sidebar.header("DBSCAN Controls")
num_points=st.sidebar.slider("Number of Points",50,500,200)
eps_value=st.sidebar.slider("Epsilon Radius",0.1,3.0,0.8)
min_samples=st.sidebar.slider("Minimum Samples",2,20,5)
cluster_spread=st.sidebar.slider("Cluster Spread",0.2,3.0,1.0)
show_noise=st.sidebar.checkbox("Show Noise Points",True)
np.random.seed()
cluster1=np.random.randn(num_points//3,2)*cluster_spread+[2,2]
cluster2=np.random.randn(num_points//3,2)*cluster_spread+[7,7]
cluster3=np.random.randn(num_points//3,2)*cluster_spread+[12,3]
noise=np.random.uniform(
    low=-2,
    high=15,
    size=(20,2)
)
X=np.vstack([
    cluster1,
    cluster2,
    cluster3,
    noise
])
dbscan=DBSCAN(
    eps=eps_value,
    min_samples=min_samples
)
labels=dbscan.fit_predict(X)
fig=go.Figure()
unique_labels=np.unique(labels)
for label in unique_labels:
    cluster_points=X[labels==label]
    if label==-1:
        if show_noise:
            fig.add_trace(
                go.Scatter(
                    x=cluster_points[:,0],
                    y=cluster_points[:,1],
                    mode="markers",
                    name="Noise",
                    marker=dict(
                        color="red",
                        size=10,
                        symbol="x"
                    )
                )
            )

    else:

        fig.add_trace(
            go.Scatter(
                x=cluster_points[:,0],
                y=cluster_points[:,1],
                mode="markers",
                name=f"Cluster {label}",
                marker=dict(
                    size=9
                )
            )
        )

fig.update_layout(
    template="plotly_dark",
    height=700,
    title="DBSCAN Clustering Visualization",
    xaxis_title="Feature 1",
    yaxis_title="Feature 2"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("How DBSCAN Works")

st.write("""
1. DBSCAN searches nearby points inside epsilon radius

2. If nearby points >= minimum samples:
   point becomes CORE POINT

3. Connected dense regions become clusters

4. Sparse isolated points become NOISE

5. DBSCAN automatically finds arbitrary shaped clusters
""")

st.subheader("DBSCAN Parameters")

st.write(f"Epsilon Radius (eps): {eps_value}")

st.write(f"Minimum Samples: {min_samples}")

st.subheader("Cluster Labels")

cluster_df=pd.DataFrame(
    X,
    columns=["Feature1","Feature2"]
)

cluster_df["Cluster"]=labels

st.dataframe(cluster_df)

st.subheader("Cluster Summary")

num_clusters=len(set(labels))-(
    1 if -1 in labels else 0
)

num_noise=list(labels).count(-1)

st.write(f"Detected Clusters: {num_clusters}")

st.write(f"Noise Points: {num_noise}")