import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
df = pd.read_csv("C:/Users/Asarv/Downloads/archive/Train_data.csv")

# Convert labels to binary (normal vs anomaly)
le = LabelEncoder()
true_labels = le.fit_transform(df['class'])

# Select numerical columns only
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
X = df[numeric_cols]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering with optimal k=4
optimal_k = 4
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
predicted_clusters = kmeans_optimal.fit_predict(X_scaled)

# Map cluster labels to binary labels (normal vs anomaly)
# We'll map the majority class in each cluster to the corresponding true label
cluster_to_label = {}
for cluster in range(optimal_k):
    mask = predicted_clusters == cluster
    if sum(true_labels[mask] == 1) > sum(true_labels[mask] == 0):
        cluster_to_label[cluster] = 1
    else:
        cluster_to_label[cluster] = 0

predicted_labels = np.array([cluster_to_label[cluster] for cluster in predicted_clusters])

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Print results
print(f"\nAccuracy Score: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Additional metrics
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels))

# Calculate silhouette score for the clustering
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_scaled, predicted_clusters)
print(f"\nSilhouette Score: {silhouette_avg:.4f}")

# Print cluster distribution
print("\nCluster Distribution:")
print(pd.Series(predicted_clusters).value_counts().sort_index())

# Print label distribution in each cluster
print("\nLabel Distribution in Each Cluster:")
for cluster in range(optimal_k):
    mask = predicted_clusters == cluster
    cluster_labels = true_labels[mask]
    print(f"\nCluster {cluster}:")
    print(pd.Series(cluster_labels).value_counts())