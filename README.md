### K-Means Clustering

```python
# Cluster data points using k-means clustering
def k_means_clustering(data, k, max_iterations):

    # Randomly initialize k cluster centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False), :]
    
    # max iterations
    for i in range(max_iterations):
        
        # Assign each data point to the nearest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)
        
        # Recalculate the centroid of each cluster
        for j in range(k):
            centroids[j] = np.mean(data[cluster_assignments == j], axis=0)
    
    return cluster_assignments, centroids
```
