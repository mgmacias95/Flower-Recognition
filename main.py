import cv2
import sys
# import numpy as np
import flower as fl
import cluster_bow as cb

if __name__ == '__main__':
    # Check the parameters
    if len(sys.argv) < 2:
        sys.exit('Usage: %s <descriptor type>' % sys.argv[0])

    # Labels of the differents class
    labels = [
        'Daffodil', 'Snowdrop', 'LillyValley', 'Bluebell',
        'Crocus', 'Iris', 'Tigerlily', 'Tulip', 'Fritillary',
        'Sunflower', 'Daisy', 'Colts_Foot', 'Dandelion',
        'Cowslip', 'Buttercup', 'Windflower', 'Pansy',
    ]

    # Load all the images
    images = [cv2.imread('Dataset/image_' + '%0*d' % (4, i) + '.jpg',
                         flags=cv2.IMREAD_COLOR) for i in range(1, 50)]

    # num_photos_per_class = 80


    # Create geometric vocabulary of the images and then, we do K-Means
    # clustering to create the Bag Of Words and get the
    # labels and histograms of every class


    bag_of_words = fl.create_bag_of_words(images, sys.argv[1].upper())
