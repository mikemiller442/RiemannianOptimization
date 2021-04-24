import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import constant



def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    s = sigmoid(x)
    return np.multiply(s,1-s)

def ReLu(x):
    x[x < 0] = 0
    return x

def ReLuPrime(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x


def convolve(x, kernels, biases):
    r = ReLu
    pixels = x.reshape(constant.inputImgDim[0],constant.inputImgDim[1])
    map = np.zeros((constant.numKernels[0], constant.inputImgDim[0] - constant.dimReduce, constant.inputImgDim[1] - constant.dimReduce))

    for u in range(constant.numKernels[0]):
        corr = signal.correlate(pixels,kernels[u],mode="valid")
        map[u] = r(corr + biases[u]).reshape(constant.inputImgDim[0] - constant.dimReduce,constant.inputImgDim[0] - constant.dimReduce)
    return map


def pool(featureMap):
    poolingTensor = np.zeros((featureMap.shape[0],featureMap.shape[1],featureMap.shape[2]))
    pooled = np.zeros((featureMap.shape[0], featureMap.shape[1]//2,featureMap.shape[2]//2))
    for k in range(pooled.shape[0]):
        for i in range(pooled.shape[1]):
            for j in range(pooled.shape[2]):
                max = featureMap[k,i*2,j*2]
                maxIndex = (k,i*2,j*2)
                if (featureMap[k,i*2 + 1,j*2] > max):
                    max = featureMap[k,i*2 + 1,j*2]
                    maxIndex = (k,i*2 + 1,j*2)
                if (featureMap[k,i*2,j*2 + 1] > max):
                    max = featureMap[k,i*2,j*2 + 1]
                    maxIndex = (k,i*2,j*2 + 1)
                if (featureMap[k,i*2 + 1,j*2 + 1] > max):
                    max = featureMap[k,i*2 + 1,j*2 + 1]
                    maxIndex = (k,i*2 + 1,j*2 + 1)
                pooled[k,i,j] = max
                poolingTensor[maxIndex[0],maxIndex[1],maxIndex[2]] = 1
    return pooled, poolingTensor


def convBP(dy, poolingTensor, pixels):

    # propagate through pooling layer
    for p in range(poolingTensor.shape[0]):
        for u in range(poolingTensor.shape[1]):
            for v in range(poolingTensor.shape[2]):
                if (poolingTensor[p,u,v] == 1):
                    poolingTensor[p,u,v] = dy[p, u // 2, v // 2]

    db = np.zeros((constant.numKernels[0],1))

    for p in range(constant.numKernels[0]):
        db[p] = poolingTensor[p].sum()

    dw = np.zeros((constant.numKernels[0],constant.maskHeight,constant.maskWidth))

    for p in range(constant.numKernels[0]):
        dw[p] = signal.correlate(pixels,poolingTensor[p],mode="valid")


    return db, dw


def CayleySGD(matrix, momentum, euclidGrad):
    momentum = constant.momentumCoef*momentum - euclidGrad
    What = momentum @ matrix.transpose() - (1/2)*matrix @ (matrix.transpose() @ momentum @ matrix.transpose())
    W = What - What.transpose()
    momentum = W @ matrix
    alpha = min(constant.learnRate, (2*constant.q)/(np.linalg.norm(W) + constant.epsilon))
    Y = matrix + alpha*momentum
    for i in range(constant.s):
        Y = matrix + (alpha/2)*W @ (matrix + Y)
    return Y, momentum


def learn(kernels, biases, weights, desMat, ytrain, validDesMat, vyTrain):
    print("In rnn.learn")
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


    OutputList = []
    ActivationList = []
    OuterDeltaList = []
    DeltaList = []

    # RGD momentum vector for orthonormal w1 matrix
    if (constant.riemann == 1):
        momentum = np.zeros(weights[0].shape)


    for l in range(constant.numLayers):
        ActivationList.append(np.zeros((constant.weightDims[l],1)))


    for l in range(constant.numLayers):
        OutputList.append(np.zeros((constant.weightDims[l+1],1)))


    ActivationList.append(np.zeros((1,1)))


    for l in range(constant.numLayers):
        OuterDeltaList.append(np.zeros((constant.weightDims[l],1)))


    for l in range(constant.numLayers):
        DeltaList.append(np.zeros(weights[l].shape))


    # Deltas for the parameters in the convolutional layers
    DeltaW = np.zeros((constant.numKernels[0], constant.maskHeight, constant.maskWidth))
    DeltaB = np.zeros((constant.numKernels[0], 1))


    cost = 0
    for i in range(constant.NUM_EPOCHS):
        if (i % constant.printCost == 0):
            if (i > 0):
                validationAccuracy = validate(kernels, biases, weights, validDesMat, vyTrain)
                print("validationAccuracy: ", validationAccuracy)
                w1 = weights[0]
                u, s, vh = np.linalg.svd(w1)
                svDev = 0.0
                for j in range(w1.shape[0]):
                    svDev += (s[j]-1)**2

                print("avg squared deviation of sv from 1: ", svDev/(w1.shape[0]))
                print("singular values sum: ", s.sum())


            print("cost: ", cost)
            print("epoch: ", i)
            print("kernels ", kernels)
            print("biases ", biases)


        cost = 0
        for j in range(desMat.shape[0]//constant.batchSize):

            if ((j % 50 == 0) & (j > 0)):
                print("j: ", j)
                print("cost: ", cost*(desMat.shape[0]/(j*constant.batchSize)))

            for l in range(constant.numLayers):
                OuterDeltaList[l] = np.zeros((weights[l].shape[0],1))

            for l in range(constant.numLayers):
                DeltaList[l] = np.zeros(weights[l].shape)


            # Deltas for the parameters in the convolutional layers
            DeltaW = np.zeros((constant.numKernels[0], constant.maskHeight, constant.maskWidth))
            DeltaB = np.zeros((constant.numKernels[0], 1))


            for k in range(constant.batchSize):

                x = desMat[k + j*constant.batchSize]

                # Starting Convolutional and Pooling Layers
                map = convolve(x, kernels[0], biases[0])
                pooled, poolingTensor = pool(map)
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



                OuterDeltaList[constant.numLayers - 1] = ActivationList[constant.numLayers] - ytrain[k + j*constant.batchSize]
                cost += ((ActivationList[constant.numLayers] - ytrain[k + j*constant.batchSize])**2)/desMat.shape[0]


                # backprop for FC layers
                for l in range(constant.numLayers - 2, -1, -1):
                    wob = np.delete(weights[l+1],0,1)
                    OuterDeltaList[l] = np.multiply(hgp(OutputList[l]).reshape(OutputList[l].shape[0],1), wob.transpose() @ OuterDeltaList[l+1])


                # accumulate partials for FC layers
                for l in range(constant.numLayers):
                    DeltaList[l] += OuterDeltaList[l] @ ActivationList[l].transpose()


                # backprop for convolutional layers
                wob = np.delete(weights[0],0,1)
                dy = wob.transpose() @ OuterDeltaList[0]
                dy = dy.reshape(constant.numKernels[0],(constant.inputImgDim[0] - constant.dimReduce)//2,(constant.inputImgDim[0] - constant.dimReduce)//2)
                pixels = x.reshape(constant.inputImgDim[0],constant.inputImgDim[1])
                db, dw = convBP(dy, poolingTensor,pixels)

                # accumulate partials for convolutional layer
                DeltaB += db
                DeltaW += dw


            # RGD batch update with orthonormal w1 matrix
            if (constant.riemann == 1):
                w1 = weights[0]
                euclidGrad = constant.orthoLearnRate*(DeltaList[0]/constant.batchSize)
                w1, momentum = CayleySGD(w1, momentum, euclidGrad)
                weights[0] = w1

                # gradient descent update of the weights for 1 batch
                for l in range(1, constant.numLayers):
                    weights[l] = weights[l] - constant.learnRate*(DeltaList[l]/constant.batchSize)
            else:
                # GD batch update for FC layers
                for l in range(constant.numLayers):
                    weights[l] = weights[l] - constant.learnRate*(DeltaList[l]/constant.batchSize)


            # GD batch update for Convolutional layers
            kernels[0] = kernels[0] - constant.learnRate*(DeltaW/constant.batchSize)
            biases[0] = biases[0] - constant.learnRate*(DeltaB/constant.batchSize)


    return weights


def validate(kernels, biases, weights, validDesMat, vyTrain):
    OutputList = []
    ActivationList = []

    for l in range(constant.numLayers):
        ActivationList.append(np.zeros((constant.weightDims[l],1)))


    for l in range(constant.numLayers):
        OutputList.append(np.zeros((constant.weightDims[l+1],1)))


    ActivationList.append(np.zeros((1,1)))
    count = 0.0
    predictedY = np.zeros((validDesMat.shape[0],1))

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

    for i in range(len(vyTrain)):

        x = validDesMat[i]

        # Starting Convolutional and Pooling Layers
        map = convolve(x, kernels[0], biases[0])
        pooled, poolingTensor = pool(map)
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

        if (OutputList[constant.numLayers-1] >= 0):
            predictedY[i] = 1
            if (vyTrain[i] == 1):
                count+=1
        else:
            predictedY[i] = 0
            if (vyTrain[i] == 0):
                count+=1

    print(count)
    print(validDesMat.shape[0])
    return count





#
#
#
#
#
