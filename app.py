import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import (preprocessing, pca_reduction, clusteringKMeans,
                   clusteringKMedoids, clusteringDBSCAN, clustering, evaluate)

st.title('Clustering App')

file = st.file_uploader('Upload your dataset', type=['csv'])

if file is not None:
    df = pd.read_csv(file)
else:
    st.info('Upload a CSV file to start')
    st.stop()

st.subheader('Dataset Preview')
st.dataframe(df.head())

st.subheader('Feature Selection')
feature_mode = st.radio('Select feature mode', ['Manual Selection', 'PCA Reduction'])

features = st.multiselect('Select the features for clustering', df.columns.to_list())
if len(features) < 2:
    st.warning('Select at least 2 features')
    st.stop()

X = np.asarray(preprocessing(df, features))

if feature_mode == 'PCA Reduction':
    max_components = min(X.shape[1], 10)
    if max_components == 2:
        n_components = 2
        st.write('Number of PCA components: 2')
    else:
        n_components = st.slider('Number of PCA components', 2, max_components, 2)
    X = np.asarray(pca_reduction(X, n_components))
    st.write(f'Data reduced to {n_components} principal components')

st.subheader('Model Selection')
model_type = st.selectbox('Select the clustering model',
                          ['K-Means', 'K-Medoids', 'DBSCAN', 'Hierarchical Clustering'])

st.subheader('Hyperparameter Tuning')
if model_type == 'K-Means':
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    labels = clusteringKMeans(X, n_clusters=n_clusters)

elif model_type == 'K-Medoids':
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    metric = st.selectbox('Select metric', ['euclidean', 'manhattan', 'cosine'])
    labels = clusteringKMedoids(X, n_clusters=n_clusters, metric=metric)

elif model_type == 'DBSCAN':
    eps = st.slider('Epsilon', 0.1, 5.0, 0.5)
    min_samples = st.slider('Min Samples', 1, 10, 5)
    labels = clusteringDBSCAN(X, eps=eps, min_samples=min_samples)

else:
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    linkage = st.selectbox('Select linkage', ['ward', 'single', 'complete', 'average'])
    labels = clustering(X, n_clusters=n_clusters, linkage=linkage)

labels = np.asarray(labels)

# En DBSCAN el ruido (-1) no es un cluster
mask = labels != -1 if model_type == 'DBSCAN' else np.ones(len(labels), dtype=bool)
n_found = len(np.unique(labels[mask]))

if n_found < 2:
    st.warning('Fewer than 2 clusters were found. Try adjusting the parameters.')
    st.stop()

df_result = df.copy()
df_result['cluster'] = labels

st.subheader('Cluster Labels')
counts = df_result['cluster'].value_counts().sort_index()
st.dataframe(counts.rename_axis('cluster').reset_index(name='count'))
if model_type == 'DBSCAN':
    st.write('Clusters found:', n_found, '| Noise points:', int((labels == -1).sum()))

st.subheader('Clustered Dataset')
st.dataframe(df_result)

st.subheader('Evaluation')
silhouette, davies = evaluate(X[mask], labels[mask])
col1, col2 = st.columns(2)
col1.metric('Silhouette Score', f'{silhouette:.3f}')
col2.metric('Davies-Bouldin Index', f'{davies:.3f}')

st.subheader('Visualization')
if X.shape[1] == 2:
    X_plot = X
    if feature_mode == 'Manual Selection':
        xlabel, ylabel = features[0], features[1]
    else:
        xlabel, ylabel = 'PC1', 'PC2'
else:
    st.info('More than 2 dimensions: PCA applied to plot in 2D')
    X_plot = np.asarray(pca_reduction(X, 2))
    xlabel, ylabel = 'PC1', 'PC2'

fig, ax = plt.subplots()
scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', s=25)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.legend(*scatter.legend_elements(), title='Cluster')
st.pyplot(fig)