import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import constant
import rnn

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

weightMatrixNames = ["w1.npy", "w2.npy", "w3.npy", "w4.npy", "w5.npy"]

# # unit tests
#
# print("pool unit tests")
# fm1 = np.zeros((2,5,5))
# fm1[0] = np.array([1,2,3,4,-1,-7,8,9,10,11,12,-13,13,14,15,16,5,-5,7,-7,8,-2,-14,19,1]).reshape(5,5)
# fm1[1] = np.array([1,12,3,-6,-1,-2,18,-94,1,-2,2,-103,3,-14,-15,3,5,-5,7,-7,8,-2,-14,19,1]).reshape(5,5)
# print(fm1)
# res, poolingTensor = rnn.pool(fm1)
# print(res)
# print("poolingTensor")
# print(poolingTensor)
#
#
# print("convolve unit tests")
# biases = np.zeros((1,1))
# biases[0] = [.5]
# kt = np.zeros((1,2,2))
# kt[0] = np.array([1,1,1,1]).reshape(2,2)
# x = np.array([1,2,3,4,-1,-7,8,9,10,11,12,-13,13,14,15,16,5,-5,7,-7,8,-2,-14,19,1])
# print(x.reshape(5,5))
# mapX = rnn.convolve(x,kt,biases)
# print(mapX)
#
# y = np.array([1,12,3,-6,-1,-2,18,-94,1,-2,2,-103,3,-14,-15,3,5,-5,7,-7,8,-2,-14,19,1])
# print(y.reshape(5,5))
# mapY = rnn.convolve(y,kt,biases)
# print(mapY)
#
#
# print("convBP unit tests")
# dy = np.array([-1,-2,3.5,0.5]).reshape(1,2,2)
# print(dy)
# pt = np.zeros((1,4,4))
# pt[0] = np.array([0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1]).reshape(4,4)
# print("dim pt ", pt.shape)
# pixels = np.array([1,2,3,4,-1,-7,8,9,10,11,12,-13,13,14,15,16,5,-5,7,-7,8,-2,-14,19,1]).reshape(5,5)
# db, dw = rnn.convBP(dy, pt, pixels)
# print(db)
# print(dw)
#
# ## finished unit tests

## load weights for testing
kernels = []
biases = []
weights = []

kernels.append(np.load("kernels.npy"))
biases.append(np.load("biases.npy"))


for l in range(constant.numLayers):
    weights.append(np.load(weightMatrixNames[l]))
## finished loading weights

## Testing for Orthogonality

## Finished Testing for Orthogonality

w1 = weights[0]
dotSum = 0.0
for i in range(w1.shape[1]):
    dotSum += np.dot(w1[:,i],w1[:,i]).item()

print(dotSum/(w1.shape[1]))

u, s, vh = np.linalg.svd(w1)
svDev = 0.0
for i in range(w1.shape[0]):
    svDev += (s[i]-1)**2

print("avg squared deviation of sv from 1: ", svDev/(w1.shape[0]))
print("singular values sum: ", s.sum())
print("dim w1: ", w1.shape)
print("num singular values: ", s.shape)



## exit if doing unit tests
exit()


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    s = sigmoid(x)
    return np.multiply(s,1-s)

def ReLu(x):
    x[x < 0] = 0
    return x

def ReluPrime(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x



csvArray = pd.read_csv('data1b.csv', header=None).to_numpy()

ytrain = np.zeros(200, dtype=np.uint8)
ytrain[:50] = 1
ytrain[150:] = 1
# note: the first 50 and last 50 of csvArray are class 0
#       the middle 100 of csvArray are class 1

m = len(csvArray[:,0])
desMat = np.dstack((np.ones(m),csvArray[:,0], csvArray[:,1])).reshape(m,3)
desMat = np.expand_dims(desMat, axis=1)


# visualization
class0 = []
class1 = []
for row in range(len(desMat)):
    if ytrain[row] == 1:
        class1.append(desMat[row,0,1:].tolist())
    else:
        class0.append(desMat[row,0,1:].tolist())

plt.plot(np.asarray(class0)[:,0], np.asarray(class0)[:,1], 'go')
plt.plot(np.asarray(class1)[:,0], np.asarray(class1)[:,1], 'rx')
plt.title('Training data')
plt.show()


# if you leave the sizes of the layers as: 2 4 1
w1 = pd.read_csv('w1b.csv', header=None).to_numpy() # w1 will be 4 by 3
w2 = pd.read_csv('w2b.csv', header=None).to_numpy() # w2 will be 1 by 5
# these sizes include an extra column for the bias unit in previous layer




# w1, w2 = learn(w1, w2, desMat, ytrain)

weights = [w1,w2]
desMat2 = np.dstack((csvArray[:,0], csvArray[:,1])).reshape(m,2)
desMat2 = np.expand_dims(desMat2, axis=1)
weights = rnn.learn([], [], weights, desMat2, ytrain)
w1 = weights[0]
w2 = weights[1]
print("printing weights")
print(w1)
print(w2)

### done learning weights

if (constant.sigmoidFunction == 1):
    hg = sigmoid
else:
    hg = ReLu

g = sigmoid


# to be done after completed training
testInput = pd.read_csv('testdata.csv', header=None).to_numpy()
testDesMat = np.dstack((np.ones(m),testInput[:,0], testInput[:,1])).reshape(m,3)
testDesMat = np.expand_dims(testDesMat, axis=1)

ytest = np.zeros(200, dtype=np.uint8)
ytest[:50] = 1
ytest[150:] = 1
# note: the first 50 and last 50 of testInput are class 0
#       the middle 100 of csvArray are class 1


print("testDesMat dim :", testDesMat.shape)


count = 0
predictedY = np.zeros((200,1))

for i in range(len(ytest)):
    x = testDesMat[i]
    a1 = np.matrix(x).reshape(w1.shape[1],1)
    z2 = w1 @ a1
    a2 = hg(z2).reshape(z2.shape[0],1)
    a2 = np.vstack(([1],a2))
    z3 = w2 @ a2
    # print(z3)
    print("ytest, output, ", ytest[i], g(z3))
    if (z3 >= 0):
        predictedY[i] = 1
        if (ytest[i] == 1):
            count+=1
    else:
        predictedY[i] = 0
        if (ytest[i] == 0):
            count+=1

print(count)

for row in range(len(testDesMat)):
    if predictedY[row] == 1:
        class1.append(testDesMat[row,0,1:].tolist())
    else:
        class0.append(testDesMat[row,0,1:].tolist())

plt.plot(np.asarray(class0)[:,0], np.asarray(class0)[:,1], 'gv')
plt.plot(np.asarray(class1)[:,0], np.asarray(class1)[:,1], 'r^')
plt.title('Test data')
plt.show()




#
#
#
#
#
