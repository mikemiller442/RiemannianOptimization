import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import constant
import rnn
# 2 input units, ? hidden units and 1 output unit

NUM_EPOCHS = 2500
learnRate = 0.03
batchSize = 10
L = 3
# 2 input units, 4 hidden units, 1 output unit
s1 = 2
s2 = 4
s3 = 1


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoidPrime(x):
    return sigmoid(x)*(1-sigmoid(x))


def ReLu(x):
    return max(0,x)


def ReluPrime(x):
    if (x > 0):
        return 1
    else:
        return 0



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
weights = rnn.learn(weights, desMat2, ytrain)
w1 = weights[0]
w2 = weights[1]
print("printing weights")
print(w1)
print(w2)

### done learning weights

if (constant.sigmoidFunction == 1):
    hg = np.vectorize(sigmoid)
else:
    hg = np.vectorize(ReLu)

g = np.vectorize(sigmoid)


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
    a2 = hg(z2)
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
