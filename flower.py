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
# df = pd.DataFrame(columns = ['label', 'hog_values'])
df_labels = np.zeros(len(images), np.int32)
df_data   = []
print("Empiezo a crear el DataFrame")
# Fill the DataFrame
for i in range(len(images)):
    # df.loc[i] = [labels[i//80], hog_descriptor(images[i])]
    df_labels[i] = i//num_photos_per_class
    df_data.append(hog_descriptor(images[i]))

df_data_array = np.array(df_data, np.float32)
df_data.clear()

# Once we got the hole dataset in a DataFrame, we must do
# a subset for train and test
training_mask = np.random.choice(len(images), size=int(len(images)*TR_SIZE), replace=False)

# Declare SVM model
svm_model = cv2.ml.SVM_create()
svm_model.setGamma(5.4)
svm_model.setC(2.67)
svm_model.setNu(0.05)
svm_model.setKernel(cv2.ml.SVM_RBF)
svm_model.setType(cv2.ml.SVM_NU_SVC)

# declare responses vector
# responses = np.array(df.size)

svm_model.train(samples=df_data_array[training_mask], 
                layout=cv2.ml.ROW_SAMPLE, 
                responses=df_labels[training_mask])

# svm_params = dict(svm_type = cv2.ml.SVM_NU_SVC, # SVM for n-class classification (n >= 2)
#                   kernel_type = cv2.ml.SVM_RBF, # e^{-gamma||x_i-x_j||^2}, gamma > 0
#                   C = 2.67,
#                   gamma = 5.4
#                  )
# print("voy a entrenar el modelo")
# # train the model with train subset
# svm_model.train(df.ix[training_mask]['hog_values'],
#                 df.ix[training_mask]['label'], params=svm_params)
