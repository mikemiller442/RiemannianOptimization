import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from six.moves import cPickle
import constant
import spectralClustering

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

df = pd.read_csv('fashion-mnist/fashion-mnist_train.csv')

data = df.to_numpy()
X = data[:,1:785]
Y = np.array(data[:,0])


print('data dim ', data.shape)
print('Y dim ', Y.shape)
print(X)
print(data[:,0])
print('X dim ', X.shape)


trainingImages = X[0:constant.numTrain,:]
testImages = X[constant.numTrain:10000,:]


trainingLabels = Y[0:constant.numTrain]
testLabels = Y[constant.numTrain:30000]


print(trainingImages)
print(sum(trainingImages[200]))


new_basis = spectralClustering.reducedBasis(trainingImages)
print(new_basis)
print(new_basis.shape)
print('inner product of columns 1 and 2 ', np.dot(new_basis[:,0],new_basis[:,1]))


np.save('mnistBasis.npy', new_basis)


# pixels = X[1,:]
# pixels = np.array(pixels, dtype='uint8')
# pixels = pixels.reshape((28, 28))
#
#
# cv.imshow(str(class_names[Y[1]]), pixels)
# cv.waitKey(0)
# cv.destroyAllWindows()




#
#
#
#
#
#
#
#
#
#
