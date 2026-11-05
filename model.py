import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA  
from sklearn.metrics import pairwise_distances 


def preprocessing(df, feature):
    x = df[feature].copy()
    numerical = x.select_dtypes(include=['int64', 'float64']).columns
    categorical = x.select_dtypes(include=['object']).columns

    if len(numerical) > 0:
        x[numerical] = StandardScaler().fit_transform(x[numerical])

    for col in categorical:
        x[col] = LabelEncoder().fit_transform(x[col])
    return x

def clustering(x, n_clusters, linkage):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    model.fit(x)
    return model.labels_

def dbscan_clustering(x, eps=0.5, min_samples=5, metric='euclidean'):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    model.fit(x)
    return model.labels_

def evaluate(x, labels, exclude_noise=True):
    labels = np.asarray(labels)
    x = np.asarray(x)

    if exclude_noise:
        mask = labels != -1
        x, labels = x[mask], labels[mask]

    n_clusters = len(set(labels))
    if n_clusters < 2 or n_clusters >= len(labels):
        return None, None

    return silhouette_score(x, labels), davies_bouldin_score(x, labels)


def dbscan_summary(labels):
    labels = np.asarray(labels)
    return {
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'n_noise': int((labels == -1).sum()),
    }

def clusteringDBSCAN(x, eps=0.5, min_samples=5, metric='euclidean'):
    # Mismo nombre que importa app.py; reutiliza dbscan_clustering
    return dbscan_clustering(x, eps=eps, min_samples=min_samples, metric=metric)


def clusteringKMeans(x, n_clusters, random_state=42):
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    model.fit(x)
    return model.labels_


def clusteringKMedoids(x, n_clusters, metric='euclidean', max_iter=300, random_state=42):
    x = np.asarray(x, dtype=float)
    D = pairwise_distances(x, metric=metric)
    rng = np.random.default_rng(random_state)

    medoids = rng.choice(len(x), n_clusters, replace=False)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)

        new_medoids = medoids.copy()
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) > 0:
                costs = D[np.ix_(members, members)].sum(axis=1)
                new_medoids[k] = members[np.argmin(costs)]

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    return np.argmin(D[:, medoids], axis=1)


def pca_reduction(x, n_components):
    x = np.asarray(x, dtype=float)
    n_components = int(min(n_components, x.shape[1], x.shape[0]))
    return PCA(n_components=n_components).fit_transform(x)
