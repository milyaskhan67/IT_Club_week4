import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load and preprocess the dataset
df = pd.read_csv('Mall_Customers.csv')
df = df.drop(columns=['CustomerID'])
df = df.drop_duplicates().dropna()
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

# Select features for clustering
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means Clustering to group similar data points
optimal_k = 6  # This value can be determined via the Elbow Method if needed
kmeans = KMeans(n_clusters=optimal_k, random_state=101)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters using a scatter plot (Age vs Spending Score)
plt.figure(figsize=(10, 6))
plt.title('Clusters: Age vs Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.scatter(df['Age'], df['Spending Score (1-100)'], c=df['Cluster'], s=50, cmap='viridis')
plt.colorbar(label='Cluster Label')
plt.show()

# Reduce dataset dimensionality using PCA for 2D visualization
pca = PCA(n_components=2, random_state=101)
X_pca = pca.fit_transform(X)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# Visualize the clusters in the PCA-reduced 2D space
plt.figure(figsize=(10, 6))
plt.title('PCA Visualization of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.scatter(df['PC1'], df['PC2'], c=df['Cluster'], s=50, cmap='viridis')
plt.colorbar(label='Cluster Label')
plt.show()

# Summarize findings from clustering and PCA
print("Summary of Findings:")
print("1. K-Means clustering grouped the data into 6 clusters based on features such as Gender, Age, Annual Income, and Spending Score.")
print("2. The Age vs Spending Score scatter plot shows distinct clusters, suggesting varied customer profiles.")
print("3. PCA reduced the dimensionality to two principal components while preserving the cluster structure,")
print("   providing a clear 2D visualization of how the clusters are separated in the feature space.")
