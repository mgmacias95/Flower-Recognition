import cv2
import numpy as np


# % of the training subset size over the hole dataset
TR_SIZE = 0.85

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
    magn_cells = magnitude[:10, :10], magnitude[10:, :10], magnitude[:10, 10:], magnitude[10:, 10:]
    bin_cells = binvalues[:10, :10], binvalues[10:, :10], binvalues[:10, 10:], binvalues[10:, 10:]
    # With "bincount" we can count the number of occurrences of a
    # flat array to create the histogram. Those flats arrays we can
    # create it with the NumPy function "ravel"
    histogram = [np.bincount(bin_cell.ravel(), magn.ravel(), n_bins)
                    for bin_cell, magn in zip(bin_cells, magn_cells)]
    # And return an array with the histogram
    return np.hstack(histogram)


def create_train_subset():
    subset = []
    for i in range(0, 1360, 80):
        subset += np.random.randint(low=i, high=i+80, size=4).tolist()

    return np.array(subset)


def create_unclustered_geometric_vocabulary(images, detector_type):
    # Create an empty vocabulary
    vocabulary = []

    if detector_type == 'SURF':
        detector = cv2.xfeature2d.SURF_create()

    elif detector_type == 'SIFT:':
        detector = cv2.xfeature2d.SIFT_create()

    elif detector_type == 'AKAZE':
        detector = cv2.xfeature2d.AKAZE_create()

    elif detector_type == 'MSD':
        detector = cv2.xfeature2d.MSDDetector_create()

    elif detector_type == 'FFD':
        detector = cv2.xfeature2d.FastFeatureDetector_create()

    else:
        raise ValueError('Not a suitable detector')

    for img in images:
        # Detect the keypoints on the image and
        # compute the descriptor for those keypoints
        keypoints, descriptor = detector.detectAndCompute(img, None)
        vocabulary.append(descriptor)

    return np.array(vocabulary, dtype=np.float32)
#
# test_mask = create_train_subset()
# aux = np.arange(1360)
# training_mask = np.in1d(aux, test_mask) * 1
# training_mask = np.where(training_mask == 0)[0]
#
# # We create two arrays that store the label of the images,
# # and the result of the HOG descriptor
# df_labels = np.zeros(len(images), np.int32)
# df_data   = []
# # Fill the arrays
# for i in range(len(images)):
#     df_labels[i] = i//num_photos_per_class
#     df_data.append(hog_descriptor(images[i]))
#
# df_data_array = np.array(df_data, np.float32)
# df_data.clear()
