# from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = np.array(Image.open('test_image.jpg'))

image = image / 255 # Normalizing the pixel values
row, col, colors = image.shape

print('Resolution: ', row, ' * ', col)

fig = plt.figure(figsize=(15, 10))
a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(image)
a.set_title('BMW Wallpaper')
plt.show()

# Separating colors
red_vals = image[:, :, 0]
green_vals = image[:, :, 1]
blue_vals = image[:, :, 2]

original_size = image.nbytes
print('Uncompressed size: ', image.nbytes)

k = 5 # First k singular values are considered

U_red, d_red, V_red = np.linalg.svd(red_vals, full_matrices=True)
U_green, d_green, V_green = np.linalg.svd(green_vals, full_matrices=True)
U_blue, d_blue, V_blue = np.linalg.svd(blue_vals, full_matrices=True)

U_red_k = U_red[:, 0:k]
U_green_k = U_green[:, 0:k]
U_blue_k = U_blue[:, 0:k]
d_red_k = d_red[0:k]
d_green_k = d_green[0:k]
d_blue_k = d_blue[0:k]
V_red_k = V_red[0:k, :]
V_green_k = V_green[0:k, :]
V_blue_k = V_blue[0:k, :]

compressed_size = sum([matrix.nbytes for matrix in
                        [U_red_k, d_red_k, V_red_k, U_green_k, d_green_k, V_green_k, U_blue_k, d_blue_k, V_blue_k]])

print('Compressed size: ', compressed_size)

compression_ratio = compressed_size / original_size
print('Compression ratio: ', compression_ratio)

red_approx = np.dot(U_red_k, np.dot(np.diag(d_red_k), V_red_k))
green_approx = np.dot(U_green_k, np.dot(np.diag(d_green_k), V_green_k))
blue_approx = np.dot(U_blue_k, np.dot(np.diag(d_blue_k), V_blue_k))

compressed_image = np.zeros((row, col, 3))

compressed_image[:, :, 0] = red_approx
compressed_image[:, :, 1] = green_approx
compressed_image[:, :, 2] = blue_approx

# Converting negative values to 0
compressed_image[compressed_image < 0] = 0
compressed_image[compressed_image > 1] = 1

fig = plt.figure(figsize=(15, 10))
a = fig.add_subplot(1, 1, 1)
img = plt.imshow(compressed_image)
a.set_title('Compressed image')
plt.show()
