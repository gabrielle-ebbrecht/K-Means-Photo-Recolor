# %%
import numpy as np

from skimage import io
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import math
from PIL import Image
from kneed import KneeLocator

# from colormath.color_objects import sRGBColor, LabColor
# from colormath.color_conversions import convert_color

# from colorthief import ColorThief

from palettable.cartocolors.qualitative import Vivid_3
import matplotlib.pyplot as plt

# %%
# image = Image.open('/Users/gabebb/Documents/Projects/IMG_9649.jpeg')
# image.show()


# %%
img = io.imread('/Users/gabebb/Documents/Projects/IMG_9649.jpeg')
og_shape = img.shape
io.imshow(img)
plt.show()

# %%
img_norm = img / 255.0
pixels = img_norm.reshape(-1, 3)

# %%
og_shape



# %% trying to do elbow method to find optimal clusters
inertia = []
cluster_range = range(1, 10)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    inertia.append(kmeans.inertia_)

knee_locator = KneeLocator(cluster_range, inertia, curve="convex", direction="decreasing")
k = knee_locator.knee

elbow_x = k
elbow_y = knee_locator.knee_y

plt.plot(cluster_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.axvline(x=elbow_x, color='r', linestyle='--', label=f'Optimal k = {elbow_x}')
plt.scatter(elbow_x, elbow_y, color='red', s=100, zorder=5, label='Elbow Point')

plt.legend()
plt.title('Elbow Method for Optimal k')
plt.show()



# %%
k = k + 2
kmeans = KMeans(n_clusters=k, random_state=36)
kmeans.fit(pixels)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# %% RECOLORING

palette = Vivid_3.mpl_colors #doesn't give me random gen

plt.imshow([palette])
plt.axis('off')
plt.show()


# %% HARD CODING
palette = [[138, 234, 146],  # #8AEA92
            [128, 173, 160],  # #80ADA0
            [95, 85, 102],    # #5F5566
            [51, 32, 42],     # #33202A
            [0, 0, 0]]      

palette = palette[4, 0, 2, 1, 3]
plt.imshow([palette])
plt.axis('off')
plt.show()

# have to figure out a way to get the colors not to look ridiculous, 
# and also to get a good color picker

# %%
og_shape[0]

# %%
recolored_img = np.zeros(og_shape, dtype=np.uint8)
reshaped_labels = labels.reshape(og_shape[0], og_shape[1])

# Assign each pixel the corresponding color based on its cluster label
recolor = []
for l in labels:
    recolor.append(palette[l])

recolor = np.array(recolor)
recolor = recolor.reshape(og_shape)

plt.imshow(recolor)
plt.show()


# %%
