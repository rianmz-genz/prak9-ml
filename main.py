# customer_segmentation.py

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Display data types and check for missing values
def data_info(df):
    print(df.dtypes)
    print(df.isnull().sum())

# Plot histograms for specified columns
def plot_histograms(df, columns):
    plt.figure(1, figsize=(15, 6))
    for i, col in enumerate(columns):
        plt.subplot(1, 3, i + 1)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[col], bins=20, kde=True)
        plt.title(f'Distplot of {col}')
    plt.show()

# Plot gender distribution
def plot_gender_distribution(df):
    plt.figure(1, figsize=(15, 5))
    sns.countplot(y='Gender', data=df, palette={'Male': 'blue', 'Female': 'pink'})
    plt.show()

# Plot pairwise relationships
def plot_pairwise_relationships(df, columns):
    plt.figure(1, figsize=(15, 7))
    for i, x in enumerate(columns):
        for j, y in enumerate(columns):
            plt.subplot(3, 3, i * 3 + j + 1)
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            sns.regplot(x=x, y=y, data=df)
            plt.ylabel(y)
    plt.show()

# Plot relationships with respect to gender
def plot_relationships_by_gender(df, x_col, y_col):
    plt.figure(1, figsize=(15, 6))
    for gender in ['Male', 'Female']:
        plt.scatter(x=x_col, y=y_col, data=df[df['Gender'] == gender],
                    s=200, alpha=0.5, label=gender)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs {y_col} w.r.t Gender')
    plt.legend()
    plt.show()

# Determine optimal number of clusters using Silhouette Score
def optimal_clusters_silhouette(data, max_clusters=10):
    silhouette_scores = []
    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, 
                        tol=0.0001, random_state=111, algorithm='elkan')
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    
    plt.figure(1, figsize=(15, 6))
    plt.plot(np.arange(2, max_clusters + 1), silhouette_scores, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    optimal_clusters = np.argmax(silhouette_scores) + 2  # because range starts from 2
    return optimal_clusters

# Perform K-Means clustering and visualize results
def kmeans_clustering_and_plot(data, n_clusters, x_col, y_col):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan')
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    h = 0.02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(1, figsize=(15, 7))
    plt.clf()
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

    plt.scatter(x=data[:, 0], y=data[:, 1], c=labels, s=200)
    plt.scatter(x=centroids[:, 0], y=centroids[:, 1], s=300, c='red', alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs {y_col} with {n_clusters} Clusters')
    plt.show()

# Perform 3D K-Means clustering and visualize results
def kmeans_3d_clustering_and_plot(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan')
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=50, alpha=0.5, label='Data Points')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='red', marker='*', s=200, label='Centroids')

    ax.set_title('Clusters')
    ax.set_xlabel('Age', labelpad=10)
    ax.set_ylabel('Annual Income', labelpad=10)
    ax.set_zlabel('Spending Score', labelpad=10)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Main function to run the analysis
def main():
    file_path = 'src/Mall_Customers.csv'
    df = load_data(file_path)
    data_info(df)

    # Plotting histograms and distributions
    plot_histograms(df, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    plot_gender_distribution(df)
    plot_pairwise_relationships(df, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    plot_relationships_by_gender(df, 'Age', 'Annual Income (k$)')
    plot_relationships_by_gender(df, 'Annual Income (k$)', 'Spending Score (1-100)')

    # Clustering: Age and Spending Score
    X1 = df[['Age', 'Spending Score (1-100)']].values
    optimal_clusters = optimal_clusters_silhouette(X1)
    kmeans_clustering_and_plot(X1, optimal_clusters, 'Age', 'Spending Score (1-100)')

    # Clustering: Annual Income and Spending Score
    X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    optimal_clusters = optimal_clusters_silhouette(X2)
    kmeans_clustering_and_plot(X2, optimal_clusters, 'Annual Income (k$)', 'Spending Score (1-100)')

    # Clustering: Age, Annual Income, and Spending Score
    X3 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    optimal_clusters = optimal_clusters_silhouette(X3)
    kmeans_3d_clustering_and_plot(X3, optimal_clusters)

if __name__ == "__main__":
    main()
