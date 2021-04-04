import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import constant



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



def CayleySGD(matrix, momentum, euclidGrad):
    momentum = constant.momentumCoef*momentum - euclidGrad
    What = momentum @ matrix.transpose() - (1/2)*matrix @ (matrix.transpose() @ momentum @ matrix.transpose())
    # print("dim what ", What.shape)
    # print(What)
    W = What - What.transpose()
    momentum = W @ matrix
    alpha = min(constant.learnRate, (2*constant.q)/(np.linalg.norm(W) + constant.epsilon))
    Y = matrix + alpha*momentum
    for i in range(constant.s):
        Y = matrix + (alpha/2)*W @ (matrix + Y)
    return Y, momentum


def learn(weights, desMat, ytrain):
    print("In nn.learn")
    if (constant.sigmoidFunction == 1):
        hg = np.vectorize(sigmoid)
        hgp = np.vectorize(sigmoidPrime)
    else:
        hg = np.vectorize(ReLu)
        hgp = np.vectorize(ReluPrime)

    g = np.vectorize(sigmoid)
    gp = np.vectorize(sigmoidPrime)


    OutputList = []
    ActivationList = []
    OuterDeltaList = []
    DeltaList = []


    # RGD momentum vector for orthonormal w1 matrix
    momentum = np.zeros(weights[0].shape)


    for i in range(constant.numLayers):
        ActivationList.append(np.zeros((weights[i].shape[1] - 1,1)))


    for i in range(constant.numLayers):
        OutputList.append(np.zeros((weights[i].shape[0],1)))


    ActivationList.append(np.zeros((1,1)))
    OutputList.append(np.zeros((1,1)))


    for i in range(constant.numLayers):
        OuterDeltaList.append(np.zeros((weights[i].shape[0] - 1,1)))

    for i in range(constant.numLayers):
        DeltaList.append(np.zeros(weights[i].shape))


    cost = 0
    for i in range(constant.NUM_EPOCHS):
        if (i % 50 == 0):
            print("cost: ", cost)
            print("epoch: ", i)
            constant.learnRate = constant.learnRate*constant.decayParam
        cost = 0
        for j in range(desMat.shape[0]//constant.batchSize):

            for l in range(constant.numLayers):
                OuterDeltaList[l] = np.zeros((weights[l].shape[0] - 1,1))

            for l in range(constant.numLayers):
                DeltaList[l] = np.zeros(weights[l].shape)


            for k in range(constant.batchSize):

                x = desMat[k + j*constant.batchSize]
                ActivationList[0] = x.reshape(weights[0].shape[1] - 1,1)


                # forward propagation
                for l in range(constant.numLayers-1):
                    ActivationList[l] = np.vstack(([1],ActivationList[l]))
                    OutputList[l] = weights[l] @ ActivationList[l]
                    ActivationList[l+1] = hg(OutputList[l])

                ActivationList[constant.numLayers-1] = np.vstack(([1],ActivationList[constant.numLayers-1]))
                OutputList[constant.numLayers-1] = weights[constant.numLayers-1] @ ActivationList[constant.numLayers-1]
                ActivationList[constant.numLayers] = g(OutputList[constant.numLayers-1])


                OuterDeltaList[constant.numLayers - 1] = ActivationList[constant.numLayers] - ytrain[k + j*constant.batchSize]
                cost += ((ActivationList[constant.numLayers] - ytrain[k + j*constant.batchSize])**2)/desMat.shape[0]


                # backprop
                for l in range(constant.numLayers - 2, -1, -1):
                    wob = np.delete(weights[l+1],0,1)
                    OuterDeltaList[l] = np.multiply(hgp(OutputList[l]), wob.transpose() @ OuterDeltaList[l+1])


                # accumulate partials
                for l in range(constant.numLayers):
                    DeltaList[l] += OuterDeltaList[l] @ ActivationList[l].transpose()


            # RGD batch update with orthonormal w1 matrix
            w1 = weights[0]
            euclidGrad = constant.orthoLearnRate*(DeltaList[0]/constant.batchSize)
            w1, momentum = CayleySGD(w1, momentum, euclidGrad)
            weights[0] = w1

            # gradient descent update of the weights for 1 batch
            for l in range(1, constant.numLayers):
                weights[l] = weights[l] - constant.learnRate*(DeltaList[l]/constant.batchSize)

            # # GD batch update
            # for l in range(constant.numLayers):
            #     weights[l] = weights[l] - constant.learnRate*(DeltaList[l]/constant.batchSize)

    return weights







#
#
#
#
#
