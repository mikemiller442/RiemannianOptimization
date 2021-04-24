import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import rnn
import constant

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

weightMatrixNames = ["w1.npy", "w2.npy", "w3.npy", "w4.npy", "w5.npy"]

## load data for training
df = pd.read_csv('fashion-mnist/fashion-mnist_train.csv')

## set up data for just two classes
df = df[(df['label'] == 9) | (df['label'] == 7)]
## finished filtering data for just two classes

data = df.to_numpy()
desMat = data[0:11000,1:785]
validDesMat = data[11000:12000,1:785]
Y = np.array(data[:,0])
validY = np.array(data[11000:12000,0])

desMat = (desMat - np.mean(desMat)) / np.std(desMat)

# class label 9 is the target
ytrain = Y
for i in range(len(ytrain)):
    if (ytrain[i] == 9):
        ytrain[i] = 1
    else:
        ytrain[i] = 0

# class label 9 is the target
vyTrain = validY
for i in range(len(vyTrain)):
    if (vyTrain[i] == 9):
        vyTrain[i] = 1
    else:
        vyTrain[i] = 0

kernels = []
biases = []
weights = []

np.random.seed(0)
kernels.append(np.random.rand(constant.numKernels[0],constant.maskHeight,constant.maskWidth))
biases.append(np.random.rand(constant.numKernels[0],1))

if (constant.riemann == 1):
    # Constructs random positive definite matrix and chooses the eigenvectors
    # as rows for the weight matrix, which are orthogonal by the Spectral Theorem
    A = np.random.rand(constant.weightDims[0]+1,constant.weightDims[0]+1)
    P = (A + np.transpose(A))/2 + (constant.weightDims[0]+1)*np.eye(constant.weightDims[0]+1)
    vals, vecs = np.linalg.eig(P)
    w = vecs[:,0:constant.weightDims[1]]
    print(w.shape)
    weights.append(w.transpose())
    for l in range(1,constant.numLayers):
        scale = 1/max(1.0,(constant.weightDims[l+1] + constant.weightDims[l]+1)/2.0)
        limit = math.sqrt(3.0*scale)
        weights.append(np.random.uniform(-limit, limit, size = (constant.weightDims[l+1],constant.weightDims[l]+1)))
else:
    # Xavier Initialization
    for l in range(constant.numLayers):
        scale = 1/max(1.0,(constant.weightDims[l+1] + constant.weightDims[l]+1)/2.0)
        limit = math.sqrt(3.0*scale)
        weights.append(np.random.uniform(-limit, limit, size = (constant.weightDims[l+1],constant.weightDims[l]+1)))


validDesMat = (validDesMat - np.mean(validDesMat)) / np.std(validDesMat)
weights = rnn.learn(kernels, biases, weights, desMat, ytrain, validDesMat, vyTrain)


for l in range(constant.numLayers):
    np.save(weightMatrixNames[l], weights[l])


np.save("biases.npy", biases[0])
np.save("kernels.npy", kernels[0])


## Printing orthogonality properties
w1 = weights[0]
u, s, vh = np.linalg.svd(w1)
svDev = 0.0
for i in range(w1.shape[0]):
    svDev += (s[i]-1)**2

print("avg squared deviation of sv from 1: ", svDev/(w1.shape[0]))
print("singular values sum: ", s.sum())
## Finish printing orthogonality properties

## finished training weights


# ## load weights for testing
# kernels = []
# biases = []
# weights = []
#
# kernels.append(np.load("kernels.npy"))
# biases.append(np.load("biases.npy"))
#
#
# for l in range(constant.numLayers):
#     weights.append(np.load(weightMatrixNames[l]))
# ## finished loading weights


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoidPrime(x):
    return sigmoid(x)*(1-sigmoid(x))


def ReLu(x):
    return max(0,x)


def ReLuPrime(x):
    if (x > 0):
        return 1
    else:
        return 0


df_test = pd.read_csv('fashion-mnist/fashion-mnist_test.csv')
## set up data for just two classes
df_test = df_test[(df_test['label'] == 9) | (df_test['label'] == 7)]
## finished filtering data for just two classes
testData = df_test.to_numpy()
testDesMat = testData[:,1:785]
ytest = np.array(testData[:,0])


for i in range(len(ytest)):
    if (ytest[i] == 9):
        ytest[i] = 1
    else:
        ytest[i] = 0


# Tests the model on a test set
# TODO: This code is duplicated in rnn.py. This should be implemented as a
# functional call to rnn.validate
OutputList = []
ActivationList = []

for l in range(constant.numLayers):
    ActivationList.append(np.zeros((constant.weightDims[l],1)))


for l in range(constant.numLayers):
    OutputList.append(np.zeros((constant.weightDims[l+1],1)))


ActivationList.append(np.zeros((1,1)))
testDesMat = (testDesMat - np.mean(testDesMat)) / np.std(testDesMat)
count = 0
predictedY = np.zeros((testDesMat.shape[0],1))

if (constant.sigmoidFunction == 1):
    hg = sigmoid
    hgp = sigmoidPrime
else:
    hg = ReLu
    hgp = ReLuPrime

g = sigmoid
gp = sigmoidPrime

r = ReLu
rp = ReLuPrime

for i in range(len(ytest)):

    x = testDesMat[i]

    ## Starting Convolutional and Pooling Layers
    map = rnn.convolve(x, kernels[0], biases[0])
    pooled, poolingTensor = rnn.pool(map)
    ActivationList[0] = np.vstack(([1],pooled.reshape(pooled.shape[0]*pooled.shape[1]*pooled.shape[2],1)))
    ## Done with Convolutional and Pooling Layers

    # # First activation if there are zero convolutional layers
    # ActivationList[0] = np.vstack(([1],x.reshape(constant.weightDims[0],1)))

    ## Starting FC Layers
    # forward propagation
    for l in range(constant.numLayers-1):
        OutputList[l] = weights[l] @ ActivationList[l]
        ActivationList[l+1] = np.vstack(([1],hg(OutputList[l]).reshape(OutputList[l].shape[0],1)))


    OutputList[constant.numLayers-1] = weights[constant.numLayers-1] @ ActivationList[constant.numLayers-1]
    ActivationList[constant.numLayers] = g(OutputList[constant.numLayers-1]).reshape(OutputList[constant.numLayers-1].shape[0],1)
    print("ytest, output, ", ytest[i], ActivationList[constant.numLayers])

    if (OutputList[constant.numLayers-1] >= 0):
        predictedY[i] = 1
        if (ytest[i] == 1):
            count+=1
    else:
        predictedY[i] = 0
        if (ytest[i] == 0):
            count+=1


print(count)
print(testDesMat.shape[0])

w1 = weights[0]
u, s, vh = np.linalg.svd(w1)
svDev = 0.0
for j in range(w1.shape[0]):
    svDev += (s[j]-1)**2

print("avg squared deviation of sv from 1: ", svDev/(w1.shape[0]))
print("singular values sum: ", s.sum())


## Code for displaying images
# print('data dim ', data.shape)
# print('Y dim ', Y.shape)
# print(X)
# print(data[:,0])
# print('X dim ', X.shape)
# print('Y value', Y[4])
#
# pixels = X[4,:]
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
