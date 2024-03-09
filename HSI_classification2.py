import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datafile = 'C:/Users/Morteza/OneDrive/Desktop/PhD/New_Data/8cal_Seurat_AFTER.img'
hdrfile = 'C:/Users/Morteza/OneDrive/Desktop/PhD/New_Data/8cal_Seurat_AFTER.hdr'

# Read the hyperspectral data
hcube = envi.open(hdrfile, datafile)
img = hcube.load()

# Convert to double
img = img.astype(float)

# Extract dimensions
m, n, l = img.shape

sig = np.zeros((7, l))

# Define 7 coordinates for signature extraction
coordinates = [(46, 164), (389, 677), (316, 174), (68, 309), (46, 92), (66, 1051), (565, 72)]

# Extract signatures
for i, (x, y) in enumerate(coordinates):
    sig[i, :] = img[x, y, :]

# Rearrange the signature matrix for further processing
sig = sig.T
sig = sig[:-1,:]  # Remove the last band

# Create a new hypercube without the last band
hcuben = img[:, :, :-1]
l = 151

# Reshape the hypercube for least squares calculation
hcuben_reshaped = hcuben.reshape(m * n, l)

# Estimate abundance using least squares method
abundanceMap = np.linalg.lstsq(sig, hcuben_reshaped.T, rcond=-1)[0]
abundanceMap = abundanceMap.T.reshape(m, n, 7)

classNames = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']

# Match indexes for plotting
matchIdx = np.argmax(abundanceMap, axis=2)

# Plot the abundance map
plt.figure()
for i in range(1, 8):
    plt.contourf(np.where(matchIdx == i, i, np.nan), levels=[i-0.5, i+0.5], colors=[plt.cm.tab10(i)])
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# Create dummy patches for legend
legend_patches = [mpatches.Patch(color=plt.cm.tab10(i), label=classNames[i-1]) for i in range(1, 8)]

# Display legend
plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
