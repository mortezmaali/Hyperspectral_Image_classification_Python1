# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:11:39 2024

@author: Morteza
"""

import spectral
import numpy as np
from spectral import *
import matplotlib.pyplot as plt
from spectral import imshow

# Specify the folder path containing the ENVI files
input_folder = "C:/Users/Morteza/OneDrive/Desktop/PhD/dataN"

# Specify the base name of the ENVI header file (without extension)
base_name = "Seurat_BEFORE"

# Construct the full paths for the header and data files
header_file = f"{input_folder}/{base_name}.hdr"
data_file = f"{input_folder}/{base_name}.dat"  # Or use the correct extension based on your data format

# Read the hyperspectral image using spectral
spectral_image = spectral.open_image(header_file)

# Access the data as a NumPy array
hyperspectral_data = spectral_image.load()

#7 centroids (endmembers) and 30 iterations

(m, c) = kmeans(hyperspectral_data, 7, 30)

# Reshape the cluster labels to match the image dimensions
cluster_labels = m.flatten()

# Assign colors to each cluster
cluster_colors = np.random.rand(7, 3)  # Generating random colors for 11 clusters

# Create a colored segmented image
segmented_image = np.zeros((hyperspectral_data.shape[0], hyperspectral_data.shape[1], 3))  # Initialize segmented image with shape (M, N, 3)

# Initialize a list to store the class labels for the legend
legend_labels = []

for i in range(hyperspectral_data.shape[0]):  # Iterate over rows
    for j in range(hyperspectral_data.shape[1]):  # Iterate over columns
        label = cluster_labels[i * hyperspectral_data.shape[1] + j]  # Calculate flattened index
        segmented_image[i, j] = cluster_colors[label]

# Display the segmented image
plt.figure(figsize=(10, 10))
imshow(segmented_image)

# Add legend
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[label], markersize=10, label=str(label + 1)) for label in range(7)], loc='upper right')

plt.title('Segmented Image using K-means Clustering')
plt.show()
