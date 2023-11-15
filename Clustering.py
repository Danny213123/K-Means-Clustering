import numpy as np
import matplotlib.pyplot as plt

def k_means_clustering(data, k, max_iterations):
    # Randomly initialize k cluster centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False), :]
    
    for i in range(max_iterations):
        # Assign each data point to the nearest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)
        
        # Recalculate the centroid of each cluster
        for j in range(k):
            centroids[j] = np.mean(data[cluster_assignments == j], axis=0)
    
    return cluster_assignments, centroids

# Define the number of rows and columns in the matrix
rows = 50
cols = 30

# Create a matrix of random values between 0 and 1
matrix = np.random.rand(rows, cols)

# Create a matrix of values in vector form (x, y)
vector_matrix = np.zeros((rows, 2))
vector_matrix[:, 0] = matrix[:, 0]
vector_matrix[:, 1] = matrix[:, 1]

# Perform k-means clustering
k = 5
max_iterations = 10
cluster_assignments, centroids = k_means_clustering(vector_matrix, k, max_iterations)

# Plot the data with different colors for each cluster
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for i in range(k):
    plt.scatter(vector_matrix[cluster_assignments == i, 0], vector_matrix[cluster_assignments == i, 1], color=colors[i])
    circle = plt.Circle((centroids[i, 0], centroids[i, 1]), radius=np.max(np.sqrt(((vector_matrix[cluster_assignments == i] - centroids[i])**2).sum(axis=1))), fill=False, color=colors[i])
    plt.gcf().gca().add_artist(circle)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='k')
plt.show()
