import cv2
import numpy as np


# % of the training subset size over the hole dataset
TR_SIZE = 0.85
index_img_name = 0

def show(imagen, save=False):
    if save:
        global index_img_name
        cv2.imwrite(filename="img"+str(index_img_name)+'.jpg', img=imagen)
        index_img_name+=1
    else:
        cv2.imshow('image', imagen.astype(np.uint8))
        cv2.waitKey()
        cv2.destroyAllWindows()

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


"""
function to create an unclustered vocabulary using Feature2D descriptors.
"""
def create_bag_of_words(images, detector_type, k_size = 10):
    # Create an empty vocabulary with BOWKMeans
    vocabulary = cv2.BOWKMeansTrainer(clusterCount=k_size)

    if detector_type == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()

    elif detector_type == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()

    else:
        raise ValueError('Not a suitable detector')

    print("Creating the unclustered geometric vocabulary")

    descriptors, keypoints = [], []

    for img in images:
        # Detect the keypoints on the image and
        # compute the descriptor for those keypoints
        kp, descriptor = detector.detectAndCompute(img, None)
        descriptors.append(descriptor)
        keypoints.append(kp)
        vocabulary.add(descriptor)

    print("DONE!!")
    print("Creating the clusters with K-means")
    # K-Means clustering
    BOW = vocabulary.cluster()
    print("DONE!!")
    BOW = BOW.astype(np.float32)

    return BOW, keypoints, descriptors


"""
function that converts images to HSV color space and quantizes the color of the images to simplify them.
The color quantization is based in this tutorial
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
"""
def convert_to_HSV_and_quantize(images, K=16, show_img=False,
                                criteria=(cv2.TERM_CRITERIA_EPS +
                                          cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)):
    hsv = []
    i = 1
    for img in images:
        h = cv2.cvtColor(src=img,code=cv2.COLOR_RGB2HSV).reshape(-1,3)
        h = np.float32(h)
        ret, label, center = cv2.kmeans(data=h, K=K, bestLabels=None, criteria=criteria, attempts=10,
                                        flags=cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        qimg = res.reshape((img.shape))
        hsv.append(qimg)
        cv2.imwrite("ColorQuantization/image_"+ '%0*d' % (4, i) + '.jpg', qimg)
        i += 1

    # if the flag of showing an image is set, show the 1st one
    if show_img:
        show(hsv[1])

    return np.array(hsv)


def compute_BOW_response(BOW, images, detector_type,
                         keypoints, descriptors, k_size):

    # Create the Brute-Force Matcher
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)

    if detector_type == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()

    elif detector_type == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()

    else:
        raise ValueError('Not a suitable detector')

    BOW_extractor = cv2.BOWImgDescriptorExtractor(dextractor=detector,
                                                  dmatcher=matcher)

    # Set the vocabulary for the BOW extractor,
    # in order to compute the histograms for the images
    BOW_extractor.setVocabulary(BOW)
    BOW_descriptors = np.zeros([1360, k_size], dtype=np.float32)
    print(BOW_descriptors.shape)

    print("Computing the descriptors for the images")
    # Compute the histograms
    i = 0
    for img in images:
        hist = BOW_extractor.compute(img, detector.detect(img))
        BOW_descriptors[i] = hist[0].flatten()
        i+=1
    print("DONE!!")

    return np.array(BOW_descriptors)
