import cv2
import numpy as np

# Loading the hole dataset on a list
images = [cv2.imread('Dataset/image_'+'%0*d'%(3,i)+'.jpg',
          flags=cv2.IMREAD_COLOR) for i in range(1,1361)]
