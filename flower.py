import cv2
import numpy as np
import pandas as pd

# Labels of the differents class
labels = [
            'Daffodil', 'Snowdrop', 'LillyValley', 'Bluebell',
            'Crocus', 'Iris', 'Tigerlily', 'Tulip', 'Fritillary',
            'Sunflower', 'Daisy', 'Colts_Foot', 'Dandelion',
            'Cowslip', 'Buttercup', 'Windflower', 'Pansy',
        ]

# Loading the hole dataset on a list
images = [cv2.imread('Dataset/image_'+'%0*d'%(4,i)+'.jpg',
          flags=cv2.IMREAD_COLOR) for i in range(1,1361)]

# Compute the hog descriptor for an image
def hog_descriptor(image, n_bins = 16):
    # We get the derivatives of the image
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    # Calculate the magnitude and the angle
    magnitude, angle = cv2.cartToPolar(dx, dy)
    # Quantizing binvalues in (0..n_bins)
    binvalues = np.int32(n_bins*angle/(2*np.pi))
    # Divide the image on 4 squares and compute
    # the histogram
    magn_cells = magnitude[:10,:10], magnitude[10:,:10], magnitude[:10,10:], magnitude[10:,10:]
    B_cells = binvalues[:10,:10], binvalues[10:,:10], binvalues[:10,10:], binvalues[10:,10:]
    # With "bincount" we can count the number of occurrences of a
    # flat array to create the histogram. Those flats arrays we can
    # create it with the NumPy function "ravel"
    histogram = [np.bincount(bin_cell.ravel(), magn.ravel(), n_bins) \
                    for bin_cell, magn in zip(B_cells, magn_cells)]
    # And return an array with the histogram
    return np.hstack(histogram)

# We create a DataFrame that store the label of the images,
# and the result of the HOG descriptor
df = pd.DataFrame(columns = ['label', 'hog_values'])

# Fill the DataFrame
for i in range(0, len(images)):
    df.loc[i] = [labels[i//80], hog_descriptor(images[i])]
