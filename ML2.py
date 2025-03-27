import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class KMeans:
    def __init__(self, n_clusters, random_state=42):
        self.k = n_clusters
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = len(X)
        
        # Initialize centroids from data points
        random_idx = np.random.permutation(n_samples)[:self.k]
        self.centroids = X[random_idx]
        
        prev_labels = None
        for _ in range(100):  # Max iterations
            # Calculate distances
            distances = np.zeros((self.k, n_samples))
            for i in range(self.k):
                distances[i] = np.linalg.norm(X - self.centroids[i], axis=1)
            
            # Assign points to nearest centroid
            self.labels = np.argmin(distances, axis=0)
            
            # Check convergence
            if prev_labels is not None and np.all(prev_labels == self.labels):
                break
                
            # Update centroids
            prev_labels = self.labels.copy()
            for i in range(self.k):
                mask = self.labels == i
                if np.any(mask):
                    self.centroids[i] = np.mean(X[mask], axis=0)
        
        # Calculate inertia (SSE)
        self.inertia_ = 0
        for i in range(self.k):
            mask = self.labels == i
            if np.any(mask):
                self.inertia_ += np.sum((X[mask] - self.centroids[i]) ** 2)
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels

# Read and prepare data
df = pd.read_csv("C:/Users/Asarv/Documents/Manipal/coding/ML project/MLdataset.csv")

# Add time series features
df['value_lag1'] = df['value'].shift(1)
df['value_rolling_mean'] = df['value'].rolling(window=5).mean()
df['value_rolling_std'] = df['value'].rolling(window=5).std()
df.dropna(inplace=True)

# Prepare feature matrix
features = ['value', 'time_diff_in_seconds', 'value_lag1', 'value_rolling_mean', 'value_rolling_std']
X = df[features].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate SSE for different k values
sse = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Find optimal k using elbow method
optimal_k = 3  # Based on elbow curve analysis

# Final K-means clustering with optimal k
final_kmeans = KMeans(n_clusters=optimal_k)
clusters = final_kmeans.fit_predict(X_scaled)

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: K-means Clustering
plt.subplot(2, 2, 1)
plt.scatter(range(len(X)), df['value'], c=clusters, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('Time Points')
plt.ylabel('Values')

# Plot 2: Elbow Method
plt.subplot(2, 2, 2)
plt.plot(k_values, sse, 'bo-')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.legend()

# Plot 3: Value vs Lag Plot
plt.subplot(2, 2, 3)
plt.scatter(df['value_lag1'], df['value'], c=clusters, cmap='viridis')
plt.title('Value vs Lag-1 Plot')
plt.xlabel('Lag-1 Value')
plt.ylabel('Current Value')

# Plot 4: Rolling Statistics
plt.subplot(2, 2, 4)
plt.plot(df['value_rolling_mean'], label='Rolling Mean')
plt.plot(df['value_rolling_std'], label='Rolling Std')
plt.title('Rolling Statistics')
plt.xlabel('Time Points')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# Print SSE scores
print("\nSSE Scores for different k values:")
for k, score in zip(k_values, sse):
    print(f"k={k}: {score:.2f}")