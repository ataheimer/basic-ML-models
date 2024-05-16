import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram

df = pd.read_csv("VS2.csv", sep=";")

df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(0)

#print(df.head())
#print(df.isnull().sum())
#print(df.info())
#print(df.describe().T)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

kmeans = KMeans(n_clusters=4, random_state=42).fit(df)

#print(kmeans.get_params())

#print(kmeans.n_clusters)
#print(kmeans.cluster_centers_)
#print(kmeans.labels_)
#print(kmeans.inertia_)

kmeans = KMeans()
ssd = []
K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
#plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
#elbow.show()


kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

#print(kmeans.n_clusters)
#print(kmeans.cluster_centers_)
#print(kmeans.labels_)

clusters = kmeans.labels_

df = pd.read_csv("VS2.csv", sep=";")

df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(0)

df["cluster"] = clusters

#print(df.head())

df["cluster"] = df["cluster"] + 1

#print(df[df["cluster"]==5])


hc_average = linkage(df, "average")

plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendrogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
            truncate_mode="lastp",
            p=10,
            show_contracted=True,
            leaf_font_size=15,
            )
plt.show()


plt.figure(figsize=(7,5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=550, color='r', linestyle='--')
plt.show()