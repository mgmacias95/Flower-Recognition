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

num_photos_per_class = 80

# Loading the hole dataset on a list
images = [cv2.imread('Dataset/image_'+'%0*d'%(4,i)+'.jpg',
          flags=cv2.IMREAD_COLOR) for i in range(1,1361)]

# % of the training subset size over the hole dataset
TR_SIZE = 0.85

# do a subset for train and test
training_mask = np.random.choice(len(images), size=int(len(images)*TR_SIZE), 
                                 replace=False)

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

# train a model
def train_model(model, label, data, training_mask=training_mask, 
                lay=cv2.ml.ROW_SAMPLE):
    model.train(samples=data[training_mask], layout=lay, 
                responses=label[training_mask])

# We create two arrays that store the label of the images,
# and the result of the HOG descriptor
df_labels = np.zeros(len(images), np.int32)
df_data   = []
print("Empiezo a crear el DataFrame")
# Fill the arrays
for i in range(len(images)):
    df_labels[i] = i//num_photos_per_class
    df_data.append(hog_descriptor(images[i]))

df_data_array = np.array(df_data, np.float32)
df_data.clear()

# Declare SVM model
svm_model = cv2.ml.SVM_create()
svm_model.setGamma(5.4)
svm_model.setC(2.67)
svm_model.setNu(0.05)
svm_model.setKernel(cv2.ml.SVM_RBF)
svm_model.setType(cv2.ml.SVM_NU_SVC)

# train the svm model
train_model(model=svm_model, label=df_labels, data=df_data_array)

