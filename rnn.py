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


# Cross-correlates every kernel in the kernel tensor with the intput x
# @return numpy nD array This returns the feature map tensor
def convolve(pixels, kernels, biases):
    r = ReLu
    map = np.zeros((constant.numKernels[0], pixels.shape[0] - (kernels[0].shape[0] - 1), pixels.shape[1] - (kernels[0].shape[0] - 1)))
    for u in range(constant.numKernels[0]):
        corr = signal.correlate(pixels,kernels[u],mode="valid")
        map[u] = r(corr + biases[u]).reshape(pixels.shape[0] - (kernels[0].shape[0] - 1),pixels.shape[1] - (kernels[0].shape[0] - 1))
    return map


# Performs pooling on the output of a convolutional layer
# WARNING: only pools in 2 x 2 grids because rnn has not been extended
# to pool feature maps with odd dimensions.
# TODO: extend rnn to pool for arbitrary feature maps
def pool(featureMap):
    # print(featureMap)
    # exit()
    if (featureMap.shape[1] % 2 == 0):
        poolingTensor = np.zeros((featureMap.shape[0],featureMap.shape[1],featureMap.shape[2]))
        pooled = np.zeros((featureMap.shape[0], featureMap.shape[1]//2,featureMap.shape[2]//2))
        upperBound = pooled.shape[1]
        # print(upperBound)
        # exit()
    else:
        # print("odd dim featureMap")
        # print(featureMap.shape[1])
        # print(featureMap.shape[1]//2 + 1)
        poolingTensor = np.zeros((featureMap.shape[0],featureMap.shape[1],featureMap.shape[2]))
        pooled = np.zeros((featureMap.shape[0], featureMap.shape[1]//2 + 1,featureMap.shape[2]//2 + 1))
        upperBound = pooled.shape[1] - 1
        for k in range(pooled.shape[0]):
            for i in range(pooled.shape[1] - 1):
                max = featureMap[k,i*2,featureMap.shape[1] - 1]
                maxIndex = (k,i*2,featureMap.shape[1] - 1)
                if (featureMap[k,i*2 + 1,featureMap.shape[1] - 1] > max):
                    max = featureMap[k,i*2 + 1,featureMap.shape[1] - 1]
                    maxIndex = (k,i*2 + 1,featureMap.shape[1] - 1)
                pooled[k,i,pooled.shape[1] - 1] = max
                poolingTensor[maxIndex[0],maxIndex[1],maxIndex[2]] = 1

                # pooled[k,i,(pooled.shape[1] // 2) + 1] = featureMap[k,i,featureMap.shape[1] - 1]
                # poolingTensor[k,i,featureMap.shape[1] - 1] = 1
        for k in range(pooled.shape[0]):
            for j in range(pooled.shape[2] - 1):
                max = featureMap[k,featureMap.shape[1] - 1,j*2]
                maxIndex = (k,featureMap.shape[1] - 1,j*2)
                if (featureMap[k,featureMap.shape[1] - 1,j*2 + 1] > max):
                    max = featureMap[k,featureMap.shape[1] - 1,j*2 + 1]
                    maxIndex = (k,featureMap.shape[1] - 1,j*2 + 1)
                pooled[k,pooled.shape[1] - 1,j] = max
                poolingTensor[maxIndex[0],maxIndex[1],maxIndex[2]] = 1

                # pooled[k,(pooled.shape[1] // 2) + 1,j] = featureMap[k,featureMap.shape[1] - 1,j]
                # poolingTensor[k,featureMap.shape[1] - 1,j] = 1

            # gets the bottom right corner of the feature map
            pooled[k,pooled.shape[1] - 1,pooled.shape[2] - 1] = featureMap[k,featureMap.shape[1] - 1, featureMap.shape[2] - 1]
            poolingTensor[k,featureMap.shape[1] - 1, featureMap.shape[2] - 1] = 1

    for k in range(pooled.shape[0]):
        for i in range(upperBound):
            for j in range(upperBound):
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
    # for k in range(pooled.shape[0]):
    #     for i in range(lowerBound, pooled.shape[1]):
    #         for j in range(lowerBound, pooled.shape[2]):
    #             max = featureMap[k,i*2,j*2]
    #             maxIndex = (k,i*2,j*2)
    #             if (featureMap[k,i*2 + 1,j*2] > max):
    #                 max = featureMap[k,i*2 + 1,j*2]
    #                 maxIndex = (k,i*2 + 1,j*2)
    #             if (featureMap[k,i*2,j*2 + 1] > max):
    #                 max = featureMap[k,i*2,j*2 + 1]
    #                 maxIndex = (k,i*2,j*2 + 1)
    #             if (featureMap[k,i*2 + 1,j*2 + 1] > max):
    #                 max = featureMap[k,i*2 + 1,j*2 + 1]
    #                 maxIndex = (k,i*2 + 1,j*2 + 1)
    #             pooled[k,i,j] = max
    #             poolingTensor[maxIndex[0],maxIndex[1],maxIndex[2]] = 1
    return pooled, poolingTensor


# Performs backpropagation through a convolutional + pooling layer
# WARNING: doesn't return the downstream gradient because this program
# has not been extended to multiple convolutional layers yet.
# TODO: extend rnn to have multiple conv + pool layers
def convBP(dy, poolingTensor, pixels, kernels):
    # print("shapes")
    # print(dy.shape)
    # print(poolingTensor.shape)
    # exit()
    if (poolingTensor.shape[1] % 2 == 0):
        upperBound = poolingTensor.shape[1]
    else:
        upperBound = poolingTensor.shape[1] - 1
        for k in range(poolingTensor.shape[0]):
            for i in range(upperBound):
                # print(k)
                if (poolingTensor[k,i,upperBound] == 1):
                    poolingTensor[k,i,upperBound] = dy[k, i // 2, dy.shape[2] - 1]
        for k in range(poolingTensor.shape[0]):
            for j in range(upperBound):
                if (poolingTensor[k,upperBound,j] == 1):
                    poolingTensor[k,upperBound,j] = dy[k, dy.shape[1] - 1, j // 2]

            # gets the bottom right corner
            if (poolingTensor[k,upperBound,upperBound] == 1):
                poolingTensor[k,upperBound,upperBound] = dy[k, dy.shape[1] - 1, dy.shape[1] - 1]

    # print(upperBound)
    # exit()
    # propagate through pooling layer
    for p in range(poolingTensor.shape[0]):
        for u in range(upperBound):
            for v in range(upperBound):
                if (poolingTensor[p,u,v] == 1):
                    # print(v // 2)
                    poolingTensor[p,u,v] = dy[p, u // 2, v // 2]

    db = np.zeros((constant.numKernels[0],1))

    for p in range(constant.numKernels[0]):
        db[p] = poolingTensor[p].sum()

    dw = np.zeros((constant.numKernels[0],kernels[0].shape[0],kernels[0].shape[0]))

    for p in range(constant.numKernels[0]):
        dw[p] = signal.correlate(pixels,poolingTensor[p],mode="valid")


    paddedPT = []
    dx = np.zeros((constant.numKernels[1], pixels.shape[0], pixels.shape[1]))

    if (pixels.shape[0] % 2 == 0):
        padding = (kernels[0].shape[0] - 1)
    else:
        padding = (kernels[0].shape[0] - 1)

    for p in range(constant.numKernels[1]):
        # paddedPT.append(np.pad(dy[p], pad_width = constant.dimReduce * 2))
        paddedPT.append(np.pad(poolingTensor[p], pad_width = padding))
        # print("printing dim")
        # print(paddedPT[p].shape)
        # print(kernels[p].shape)
        # print(pixels.shape[0], pixels.shape[1])
        # print(dy[p].shape)
        dx[p] = signal.correlate(paddedPT[p],kernels[p],mode="valid")


    return db, dw, dx


# Implements Cayley SGD with Momentum as in Li, Fuxin, Todorovic
# @param matrix This is the matrix with the enforced orthogonality constraints
# @param momentum This is the momentum vector used in the algorithm
# @param euclidGrad This is the euclidean gradient calculated in normal backpropagation
# @return This returns the update parameter matrix and momentum vector
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


# def dropOutMatrix(numRows, numColumns):
#     zeroOneMatrix = np.random.rand(numRows, numColumns) < constant.dropOut
#     return zeroOneMatrix


# Learns the convolutional neural network
# TODO: extend learn to accommodate more than one convolutional + pooling
# layer in the architecture
# returns the parameters for the neural network
def learn(weights, kernels, biases, desMat, ytrain, validDesMat, vyTrain):
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
    DropOutMatrices = []

    # RGD momentum vector for orthonormal w1 matrix
    if (constant.riemann == 1):
        momentumFC = np.zeros(weights[0].shape)


    for l in range(constant.numLayers):
        ActivationList.append(np.zeros((constant.weightDims[l],1)))


    for l in range(constant.numLayers):
        OutputList.append(np.zeros((constant.weightDims[l+1],1)))


    ActivationList.append(np.zeros((1,1)))


    for l in range(constant.numLayers):
        OuterDeltaList.append(np.zeros((constant.weightDims[l],1)))


    for l in range(constant.numLayers):
        DeltaList.append(np.zeros(weights[l].shape))


    for l in range(1,constant.numLayers):
        DropOutMatrices.append(np.zeros((constant.weightDims[l],1)))



    # Deltas for the parameters in the convolutional layers
    DeltaW = np.zeros((constant.numKernels[0], constant.maskDim[0], constant.maskDim[0]))
    DeltaB = np.zeros((constant.numKernels[0], 1))


    cost = 0
    for i in range(constant.NUM_EPOCHS):

        p = np.random.permutation(desMat.shape[0])
        desMat = desMat[p]
        ytrain = ytrain[p]

        if ((i == 4) or (i == 8)):
            constant.learnRate = constant.decayParam*constant.learnRate
            constant.orthoLearnRate = constant.decayParam*constant.orthoLearnRate

        if (i % constant.printCost == 0):
            if (i > 0):
                validationAccuracy = validate(weights, kernels, biases, validDesMat, vyTrain)
                print("training count: ", trainingAccuracy)
                print("trainingAccuracy: ", trainingAccuracy / ytrain.shape[0])
                print("validation count: ", validationAccuracy)
                print("validationAccuracy: ", validationAccuracy / vyTrain.shape[0])
                w1 = weights[0]
                u, s, vh = np.linalg.svd(w1)
                svDev = 0.0
                for j in range(w1.shape[0]):
                    svDev += (s[j]-1)**2

                print("avg squared deviation of sv from 1: ", svDev/(w1.shape[0]))
                print("singular values sum: ", s.sum())


            print("cost: ", cost)
            print("epoch: ", i)
            # print("kernels ", kernels)
            # print("biases ", biases)


        cost = 0
        trainingAccuracy = 0

        for j in range(desMat.shape[0]//constant.batchSize):

            if ((j % 50 == 0) & (j > 0)):
                print("j: ", j)
                print("cost: ", cost*(desMat.shape[0]/(j*constant.batchSize)))

            for l in range(constant.numLayers):
                OuterDeltaList[l] = np.zeros((weights[l].shape[0],1))

            for l in range(constant.numLayers):
                DeltaList[l] = np.zeros(weights[l].shape)


            # Deltas for the parameters in the convolutional layers
            DeltaW = np.zeros((constant.numKernels[0], constant.maskDim[0], constant.maskDim[0]))
            DeltaB = np.zeros((constant.numKernels[0], 1))

            DeltaW2 = []
            DeltaB2 = []

            for l in range(constant.numKernels[0]):
                DeltaW2.append(np.zeros((constant.numKernels[1], constant.maskDim[1], constant.maskDim[1])))
                DeltaB2.append(np.zeros((constant.numKernels[1], 1)))

            for k in range(constant.batchSize):

                PooledList = []
                PoolingTensorList = []

                for l in range(1, constant.numLayers):
                    DropOutMatrices[l-1] = np.random.rand(constant.weightDims[l], 1) < (1 - constant.dropOut[l-1])
                    DropOutMatrices[l-1] = np.vstack(([1],DropOutMatrices[l-1]))

                x = desMat[k + j*constant.batchSize]
                x = x.reshape(constant.inputImgDim[0],constant.inputImgDim[1])

                ## Starting Convolutional and Pooling Layers

                pixels = np.zeros((1, constant.inputImgDim[0],constant.inputImgDim[1]))
                pixels[0] = x

                # print("pixels")
                # print(pixels[0])

                map = convolve(pixels[0], kernels[0][0], biases[0][0])
                pooled, poolingTensor = pool(map)
                PooledList.append(pooled)
                PoolingTensorList.append(poolingTensor)

                # print("pooled")
                # print(pooled)

                featureMaps = []
                featureMapTensors = []

                for l in range(constant.numKernels[0]):
                    map = convolve(PooledList[0][l], kernels[1][l], biases[1][l])
                    pooled, poolingTensor = pool(map)
                    featureMaps.append(pooled)
                    featureMapTensors.append(poolingTensor)

                cumulativeMap = featureMaps[0]
                cumulatePoolingTensor = featureMapTensors[0]

                for l in range(1, pooled.shape[0]):
                    cumulativeMap = np.hstack((cumulativeMap, featureMaps[l]))
                    cumulatePoolingTensor = np.hstack((cumulatePoolingTensor, featureMapTensors[l]))


                ActivationList[0] = np.vstack(([1],cumulativeMap.reshape(cumulativeMap.shape[0]*cumulativeMap.shape[1]*cumulativeMap.shape[2],1)))
                # ActivationList[0] = np.vstack(([1],PooledList[len(numKernels) - 1].reshape(PooledList[len(numKernels) - 1].shape[0]*PooledList[len(numKernels) - 1].shape[1]*PooledList[len(numKernels) - 1].shape[2],1)))

                # map = convolve(x, kernels[0], biases[0])
                # pooled, poolingTensor = pool(map)
                # ActivationList[0] = np.vstack(([1],pooled.reshape(pooled.shape[0]*pooled.shape[1]*pooled.shape[2],1)))
                ## Done with Convolutional and Pooling Layers

                # # First activation if there are zero convolutional layers
                # ActivationList[0] = np.vstack(([1],x.reshape(constant.weightDims[0],1)))

                ## Starting FC Layers
                # forward propagation
                for l in range(constant.numLayers-1):
                    OutputList[l] = weights[l] @ ActivationList[l]
                    ActivationList[l+1] = np.vstack(([1],hg(OutputList[l]).reshape(OutputList[l].shape[0],1)))
                    ActivationList[l+1] = np.multiply(ActivationList[l+1],DropOutMatrices[l]) / (1 - constant.dropOut[l])


                OutputList[constant.numLayers-1] = weights[constant.numLayers-1] @ ActivationList[constant.numLayers-1]
                ActivationList[constant.numLayers] = g(OutputList[constant.numLayers-1]).reshape(OutputList[constant.numLayers-1].shape[0],1)


                OuterDeltaList[constant.numLayers - 1] = ActivationList[constant.numLayers] - ytrain[k + j*constant.batchSize]
                cost += ((ActivationList[constant.numLayers] - ytrain[k + j*constant.batchSize])**2)/desMat.shape[0]

                if (OutputList[constant.numLayers-1] >= 0):
                    if (ytrain[k + j*constant.batchSize] == 1):
                        trainingAccuracy += 1
                else:
                    if (ytrain[k + j*constant.batchSize] == 0):
                        trainingAccuracy += 1


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
                dy = dy.reshape(constant.numKernels[0] * constant.numKernels[1],featureMaps[0][0].shape[0],featureMaps[0][0].shape[1])

                # print("featureMap dims")
                # print(featureMaps[0][0].shape[0])
                # print(featureMaps[0][0].shape[1])

                kernelDW = []
                biasDB = []
                featureDY = []

                # print("backprop for convolutional layer")
                for t in range(PooledList[0].shape[0]):
                    tempDY = dy[t:(t + constant.numKernels[0]),:,:]
                    # print(PooledList[0][t].shape)
                    # print(tempDY.shape)
                    db, dw, dx = convBP(tempDY,featureMapTensors[t],PooledList[0][t],kernels[1][t])
                    kernelDW.append(dw)
                    biasDB.append(db)
                    featureDY.append(dx)
                    # print("printing dw")
                    # print(dw)
                    # print("printing tempDY")
                    # print(tempDY)
                    # print("printing featureMapTensors[t]")
                    # print(featureMapTensors[t])
                    # print("printing PooledList[0][t]")
                    # print(PooledList[0][t])

                    # exit()

                cumulativeDY = featureDY[0]

                for l in range(1, pooled.shape[0]):
                    cumulativeDY = np.hstack((cumulativeDY, featureDY[l]))

                pixels = x.reshape(constant.inputImgDim[0],constant.inputImgDim[1])
                db, dw, dx = convBP(cumulativeDY, PoolingTensorList[0], pixels, kernels[0][0])


                # wob = np.delete(weights[0],0,1)
                # dy = wob.transpose() @ OuterDeltaList[0]
                # dy = dy.reshape(constant.numKernels[0],(constant.inputImgDim[0] - constant.dimReduce)//2,(constant.inputImgDim[0] - constant.dimReduce)//2)
                # pixels = x.reshape(constant.inputImgDim[0],constant.inputImgDim[1])
                # db, dw = convBP(dy, poolingTensor,pixels)

                # accumulate partials for second covolutional layer
                for t in range(PooledList[0].shape[0]):
                    DeltaB2[t] += biasDB[t]
                    DeltaW2[t] += kernelDW[t]

                # accumulate partials for first convolutional layer
                DeltaB += db
                DeltaW += dw


            # RGD batch update with orthonormal w1 matrix
            if (constant.riemann == 1):
                w1 = weights[0]
                euclidGrad = constant.orthoLearnRate*(DeltaList[0]/constant.batchSize)
                w1, momentumFC = CayleySGD(w1, momentumFC, euclidGrad)
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

            # print("printing gradients")
            for t in range(PooledList[0].shape[0]):
                # print(DeltaW2[t])
                # print(DeltaB2[t])
                kernels[1][t] = kernels[1][t] - constant.learnRate*(DeltaW2[t]/constant.batchSize)
                biases[1][t] = biases[1][t] - constant.learnRate*(DeltaB2[t]/constant.batchSize)

    return weights, kernels, biases


# validates the model on a validation set during training
# @return int This returns the number of correct predictions in validation set
def validate(weights, kernels, biases, validDesMat, vyTrain):
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

        PooledList = []
        PoolingTensorList = []

        x = validDesMat[i]

        x = x.reshape(constant.inputImgDim[0],constant.inputImgDim[1])

        ## Starting Convolutional and Pooling Layers

        pixels = np.zeros((1, constant.inputImgDim[0],constant.inputImgDim[1]))
        pixels[0] = x

        map = convolve(x, kernels[0][0], biases[0][0])
        pooled, poolingTensor = pool(map)
        PooledList.append(pooled)
        PoolingTensorList.append(poolingTensor)

        featureMaps = []
        featureMapTensors = []

        for l in range(pooled.shape[0]):
            map = convolve(PooledList[0][l], kernels[1][l], biases[1][l])
            pooled, poolingTensor = pool(map)
            featureMaps.append(pooled)
            featureMapTensors.append(poolingTensor)

        cumulativeMap = featureMaps[0]
        cumulatePoolingTensor = featureMapTensors[0]

        for l in range(1, pooled.shape[0]):
            cumulativeMap = np.hstack((cumulativeMap, featureMaps[1]))
            cumulatePoolingTensor = np.hstack((cumulatePoolingTensor, featureMapTensors[1]))


        ActivationList[0] = np.vstack(([1],cumulativeMap.reshape(cumulativeMap.shape[0]*cumulativeMap.shape[1]*cumulativeMap.shape[2],1)))

        # # Starting Convolutional and Pooling Layers
        # map = convolve(x, kernels[0], biases[0])
        # pooled, poolingTensor = pool(map)
        # ActivationList[0] = np.vstack(([1],pooled.reshape(pooled.shape[0]*pooled.shape[1]*pooled.shape[2],1)))

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

    return count



#
#
#
#
#
