import pandas as pd
url="https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents"
df = pd.read_csv(url)
df1 = df[["Latitude","Longitude"]]
df2 = df1.head(1000)
df2.head()
from sklearn.cluster import KMeans

k=3
kmeans = KMeans(n_clusters=k)
kmeans.fit(df2)
labels = kmeans.predict(df2)
cluster_centroids = kmeans.cluster_centers_
print(cluster_centroids)

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


fig = plt.figure(figsize=(20,20))
plt.scatter(df2["Start_Lng"],df2["Start_Lat"],c=labels.astype(np.float))
plt.scatter(cluster_centroids[:,0],cluster_centroids[:,1],c=np.arange(k), marker="^", s =150)
plt.show()
