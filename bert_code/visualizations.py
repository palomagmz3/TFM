import pandas as pd
import os
import csv
#import hdbscan
import numpy as np

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import umap.umap_ as umap

from scipy.sparse.csr import csr_matrix
from typing import List, Tuple, Dict, Union

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "bert_data")

bert_output = os.path.join(DATA_DIR, "data_to_visualize/L6N20151128_cluster_11_labels.csv")

df = pd.read_csv(bert_output, header=None)

print(df)
# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

classes = df[0]
print(classes)
# iterating the columns
for col in df.columns:
    print(col)

embeddings = df.drop(columns=0)
embeddings = embeddings.to_numpy()
print(embeddings)
'''
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=40,
    metric='euclidean', min_cluster_size=5, min_samples=None, p=None)

clusterer.fit(embeddings)
'''

# Clústering con PCA
pca = PCA(n_components=3)
#pca_result = pca.fit_transform(df[feat_cols].values)
pca_result = pca.fit_transform(embeddings)
df = df.drop(columns=0)
df['y'] = classes
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"],
    ys=df.loc[rndperm,:]["pca-two"],
    zs=df.loc[rndperm,:]["pca-three"],
    #c=df.loc[rndperm, :]["y"],
    cmap='tab10'
)

ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
'''for i in range(df.shape[0]):
    ax.text(x=df['pca-one'][i], y=df['pca-two'][i], z=df['pca-three'][i], s=classes[i])'''
plt.show()

# new pic
plt.figure(figsize=(16,7))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3,
    ax=ax1
)
'''for i in range(df.shape[0]):
    plt.text(x=df['pca-one'][i], y=df['pca-two'][i], s=classes[i])
'''
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="pca-one", y="pca-three",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3,
    ax=ax2
)
'''for i in range(df.shape[0]):
    plt.text(x=df['pca-one'][i], y=df['pca-three'][i], s=classes[i])
'''
'''
ax2 = plt.subplot(1, 2, 3)
sns.scatterplot(
    x="pca-three", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 21),
    data=df,
    legend="full",
    alpha=0.3,
    ax=ax2
)
for i in range(df.shape[0]):
    plt.text(x=df['pca-three'][i], y=df['pca-two'][i], s=classes[i])
'''

# Clúster con TSNE
tsne = TSNE(n_components=3,
            verbose=1,
            perplexity=40,
            init="pca",
            n_iter=500)#300)
tsne_results = tsne.fit_transform(embeddings)

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
df['tsne-2d-three'] = tsne_results[:,2]

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["tsne-2d-one"],
    ys=df.loc[rndperm,:]["tsne-2d-two"],
    zs=df.loc[rndperm,:]["tsne-2d-three"],
    #c=df.loc[rndperm,:]["y"],
    cmap='tab10'
)

ax.set_xlabel('tsne-2d-one')
ax.set_ylabel('tsne-2d-two')
ax.set_zlabel('tsne-2d-three')
'''for i in range(df.shape[0]):
    ax.text(x=df['tsne-2d-one'][i], y=df['tsne-2d-two'][i], z=df['tsne-2d-three'][i], s=classes[i])'''
plt.show()

# new pic
plt.figure(figsize=(16,10))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    #palette=sns.color_palette("hls", 10),
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3
)

#print(df['tsne-2d-one'])
#print(df['tsne-2d-two'])
#print(classes)
'''for i in range(df.shape[0]):
    plt.text(x=df['tsne-2d-one'][i], y=df['tsne-2d-two'][i], s=classes[i])
'''
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-three",
    hue="y",
    #palette=sns.color_palette("hls", 10),
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3
)
'''for i in range(df.shape[0]):
    plt.text(x=df['tsne-2d-one'][i], y=df['tsne-2d-three'][i], s=classes[i])
'''
# new pic que se puede borrar si funciona lo de arriba
plt.figure(figsize=(16,7))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3,
    ax=ax1
)
'''for i in range(df.shape[0]):
    plt.text(x=df['pca-one'][i], y=df['pca-two'][i], s=classes[i])
'''
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3,
    ax=ax2
)
'''for i in range(df.shape[0]):
    plt.text(x=df['tsne-2d-one'][i], y=df['tsne-2d-two'][i], s=classes[i])
'''

n_neighbors=15
#n_neighbors=5
#n_neighbors=50
n_components=3

umap_model = umap.UMAP(n_neighbors=n_neighbors,
                            n_components=n_components,
                            min_dist=0.0,
                            metric='cosine').fit(embeddings)

umap_embeddings = umap_model.transform(embeddings)
print("Reduced dimensionality with UMAP")

df['y'] = classes
df['UMAP-one'] = umap_embeddings[:,0]
df['UMAP-two'] = umap_embeddings[:,1]
df['UMAP-three'] = umap_embeddings[:,2]

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["UMAP-one"],
    ys=df.loc[rndperm,:]["UMAP-two"],
    zs=df.loc[rndperm,:]["UMAP-three"],
    #c=df.loc[rndperm,:]["y"],
    cmap='tab10'
)

ax.set_xlabel('UMAP-one')
ax.set_ylabel('UMAP-two')
ax.set_zlabel('UMAP-three')
'''
for i in range(df.shape[0]):
    ax.text(x=df['UMAP-one'][i], y=df['UMAP-two'][i], z=df['UMAP-three'][i], s=classes[i])
plt.show()
'''
'''
# new pic
plt.figure(figsize=(16,10))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="UMAP-one", y="UMAP-two",
    hue="y",
    #palette=sns.color_palette("hls", 10),
    palette=sns.color_palette("hls", 21),
    data=df,
    legend="full",
    alpha=0.3
)
for i in range(df.shape[0]):
    plt.text(x=df['UMAP-one'][i], y=df['UMAP-two'][i], s=classes[i])

ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="UMAP-two", y="UMAP-three",
    hue="y",
    #palette=sns.color_palette("hls", 10),
    palette=sns.color_palette("hls", 21),
    data=df,
    legend="full",
    alpha=0.3
)

for i in range(df.shape[0]):
    plt.text(x=df['UMAP-two'][i], y=df['UMAP-three'][i], s=classes[i])

'''
print('FIN')

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

KMean= KMeans(n_clusters=12)
KMean.fit(embeddings)
label=KMean.predict(embeddings)

print(f'Silhouette Score(n=12): {silhouette_score(embeddings, label)}')