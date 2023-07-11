'''
Main tools for MBEANN in Python.
pyMBEANN (Author: Motoaki Hiraga)

Details of MBEANN could be found in the following:
    K. Ohkura, et al., "MBEANN: Mutation-Based Evolving Artificial Neural Networks",
    ECAL 2007, pp. 936-945, 2007.
'''

import copy
import itertools
import math
import random

import numpy as np


class Node:
    def __init__(self, id, type, bias=None):
        self.id = id
        self.type = type
        self.bias = bias
        self.value = 0.0


class Link:
    def __init__(self, id, fromNodeID, toNodeID, weight):
        self.id = id
        self.fromNodeID = fromNodeID
        self.toNodeID = toNodeID
        self.weight = weight


class Operon:
    def __init__(self, id, nodeList, linkList):
        self.id = id
        self.nodeList = nodeList
        self.linkList = linkList

    def setDisabledLinkList(self, linkList):
        # List of available links for mutateAddLink.
        self.disabledLinkList = linkList

    def addLinkToDisabledLinkList(self, fromNodeID, toNodeID):
        isExists = (fromNodeID, toNodeID) in self.disabledLinkList
        if isExists == True:
            # This warning might apperar when using "isReccurent = False".
            # mutateAddNode: Trying to delete a selected link again.
            # print("WARNING: {} to {} connection already exists in the disabledLinkList of operon {}"
            #       .format(fromNodeID, toNodeID, self.id))
            pass
        self.disabledLinkList += [(fromNodeID, toNodeID)]

    def deleteLinkFromDisabledLinkList(self, fromNodeID, toNodeID):
        isExists = (fromNodeID, toNodeID) in self.disabledLinkList
        if isExists == False:
            print("WARNING: Cannot find {} to {} connection in the disabledLinkList of operon {}"
                  .format(fromNodeID, toNodeID, self.id))
        newDisabledLinkList = [i for i in self.disabledLinkList if i != (fromNodeID, toNodeID)]
        self.disabledLinkList = newDisabledLinkList


class Individual:
    # TODO: planning to use "configparser" to make the settings clean and organized.
    def __init__(self, inputSize, outputSize, hiddenSize, initialConnection,
                 maxWeight, minWeight, initialWeightType, initialWeightMean, initialWeightScale,
                 maxBias, minBias, initialBiasType, initialBiasMean, initialBiasScale,
                 isReccurent, activationFunc, addNodeAlpha, addNodeBeta):
        self.fitness = 0.0
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.initialConnection = initialConnection
        self.maxWeight = maxWeight
        self.minWeight = minWeight
        self.initialWeightType = initialWeightType
        self.initialWeightMean = initialWeightMean
        self.initialWeightScale = initialWeightScale
        self.maxBias = maxBias
        self.minBias = minBias
        self.initialBiasType = initialBiasType
        self.initialBiasMean = initialBiasMean
        self.initialBiasScale = initialBiasScale
        self.isReccurent = isReccurent
        self.fitness = 0.0

        self.activationFunc = activationFunc
        self.addNodeAlpha = addNodeAlpha
        self.addNodeBeta = addNodeBeta

        if self.initialBiasType == 'gaussian':
            initialBiases = [random.gauss(self.initialBiasMean, self.initialBiasScale)
                             for i in range(self.outputSize + self.hiddenSize)]
        elif self.initialBiasType == 'cauchy':
            initialBiases = [self.initialBiasMean +
                             self.initialBiasScale * math.tan(math.pi * (random.random() - 0.5))
                             for i in range(self.outputSize + self.hiddenSize)]
        else:
            if self.initialBiasType != 'uniform':
                print("WARNING: undefined 'initialBiasType' using 'uniform' instead")
            initialBiases = [random.uniform(self.minBias, self.maxBias)
                             for i in range(self.outputSize + self.hiddenSize)]
        initialBiases = np.clip(initialBiases, self.minBias, self.maxBias)

        inputNodeList = [Node(id=i, type='input')
                         for i in range(self.inputSize)]
        outputNodeList = [Node(id=i + self.inputSize, type='output', bias=initialBiases[i])
                          for i in range(self.outputSize)]
        hiddenNodeList = [Node(id=i + self.inputSize + self.outputSize,
                          type='hidden', bias=initialBiases[i + self.outputSize])
                          for i in range(self.hiddenSize)]

        self.maxNodeID = self.inputSize + self.outputSize + self.hiddenSize - 1

        availableConnection = 0
        if self.hiddenSize == 0:
            availableConnection = self.inputSize * self.outputSize
        else:
            availableConnection = (self.inputSize * self.outputSize +
                                   self.inputSize * self.hiddenSize +
                                   self.outputSize * self.hiddenSize)

        connectionNumber = int(round(availableConnection * self.initialConnection))
        connectionIndex = random.sample(range(availableConnection), connectionNumber)

        connections = np.zeros(availableConnection)
        for i in connectionIndex:
            connections[i] = 1

        if self.initialWeightType == 'gaussian':
            initialWeights = [random.gauss(self.initialWeightMean, self.initialWeightScale)
                              for i in range(connectionNumber)]
        elif self.initialWeightType == 'cauchy':
            initialWeights = [self.initialWeightMean +
                              self.initialWeightScale * math.tan(math.pi * (random.random() - 0.5))
                              for i in range(connectionNumber)]
        else:
            if self.initialWeightType != 'uniform':
                print("WARNING: undefined 'initialWeightType', using 'uniform' instead")
            initialWeights = [random.uniform(self.minWeight, self.maxWeight)
                              for i in range(connectionNumber)]
        initialWeights = np.clip(initialWeights, self.minWeight, self.maxWeight)

        # Conduct the initial topology.
        linkID = 0
        linkList = []
        disabledLinkList = []

        for i, (input, output) in enumerate(itertools.product(inputNodeList, outputNodeList)):
            if connections[i] == 1:
                linkList += [Link(id=linkID,
                                  fromNodeID=input.id,
                                  toNodeID=output.id,
                                  weight=initialWeights[linkID])]
                linkID += 1
            else:
                disabledLinkList += [(input.id, output.id)]

        # No output-to-output reccurent connections in the initial topology.
        if self.isReccurent == True:
            for output in outputNodeList:
                disabledLinkList += [(output.id, output.id)]

        self.operonList = [Operon(id=0,
                                  nodeList=np.concatenate([inputNodeList, outputNodeList]),
                                  linkList=np.array(linkList))]
        self.operonList[0].setDisabledLinkList(disabledLinkList)
        maxOperonID = 0

        if self.hiddenSize != 0:
            # Initializing the network with hidden nodes is not recommended, as similarly discussed in the NEAT algorithm.
            # In addition, there are no mutations to reduce nodes or links.
            # ---
            # Note:
            # Each hidden node is assigned to a different operon.
            # Only feed-forward connections (input to hidden, hidden to output connections) are configured.
            # If the initial topology is not defined with full connections, some hidden nodes might have zero in/out-degree.
            # ---
            for i, hidden in enumerate(hiddenNodeList):
                maxOperonID = i + 1
                linkList = []
                disabledLinkList = []
                for j, input in enumerate(inputNodeList):
                    if connections[i * (self.inputSize + self.outputSize) + j +
                                   self.inputSize * self.outputSize] == 1:
                        linkList += [Link(id=linkID,
                                          fromNodeID=input.id,
                                          toNodeID=hidden.id,
                                          weight=initialWeights[linkID])]
                        linkID += 1
                    else:
                        disabledLinkList += [(input.id, hidden.id)]

                for j, output in enumerate(outputNodeList):
                    if connections[i * (self.inputSize + self.outputSize) + j +
                                   self.inputSize * self.outputSize + self.inputSize] == 1:
                        linkList += [Link(id=linkID,
                                          fromNodeID=hidden.id,
                                          toNodeID=output.id,
                                          weight=initialWeights[linkID])]
                        linkID += 1
                    else:
                        disabledLinkList += [(hidden.id, output.id)]
                    if self.isReccurent == True:
                        disabledLinkList += [(output.id, hidden.id)]
                if self.isReccurent == True:
                    disabledLinkList += [(hidden.id, hidden.id)]

                self.operonList += [Operon(id=maxOperonID,
                                           nodeList=np.array([hidden]),
                                           linkList=np.array(linkList))]
                self.operonList[maxOperonID].setDisabledLinkList(disabledLinkList)

        self.maxOperonID = maxOperonID
        self.maxLinkID = linkID - 1

    def calculateNetwork(self, inputsList):

        if len(inputsList) != self.inputSize:
            raise ValueError("The number of inputs doesn't match")

        nodeList, linkList = [], []
        for operon in self.operonList:
            nodeList = np.concatenate([nodeList, operon.nodeList])
            linkList = np.concatenate([linkList, operon.linkList])

        inputNodeList = [node for node in nodeList if node.type == 'input']
        hiddenNodeList = [node for node in nodeList if node.type == 'hidden']
        outputNodeList = [node for node in nodeList if node.type == 'output']

        # Sorting by ID just in case.
        inputNodeList.sort(key=lambda node: node.id)
        hiddenNodeList.sort(key=lambda node: node.id)
        outputNodeList.sort(key=lambda node: node.id)

        for node, input in zip(inputNodeList, inputsList):
            node.value = input

        weightMatrix = np.zeros((self.maxNodeID + 1, self.maxNodeID + 1))
        for link in linkList:
            weightMatrix[link.toNodeID][link.fromNodeID] = link.weight

        nodeValueVec = [node.value for node in sorted(nodeList, key=lambda node: node.id)]

        hiddenNodeValueSum = np.matmul(weightMatrix[[i.id for i in hiddenNodeList], :], nodeValueVec)

        for node, value in zip(hiddenNodeList, hiddenNodeValueSum):
            if self.activationFunc == 'sigmoid':
                node.value = 1.0 / (1.0 + np.exp(self.addNodeBeta * (node.bias - value)))
            elif self.activationFunc == 'tanh':
                node.value = np.tanh(self.addNodeBeta * (value - node.bias))
            else:
                raise NameError("Activation function '{}' is not defined".format(self.activationFunc))

        # Update nodeValueVec.
        nodeValueVec = [node.value for node in sorted(nodeList, key=lambda node: node.id)]

        outputNodeValueSum = np.matmul(weightMatrix[[i.id for i in outputNodeList], :], nodeValueVec)

        for node, value in zip(outputNodeList, outputNodeValueSum):
            if self.activationFunc == 'sigmoid':
                node.value = 1.0 / (1.0 + np.exp(self.addNodeBeta * (node.bias - value)))
            elif self.activationFunc == 'tanh':
                node.value = np.tanh(self.addNodeBeta * (value - node.bias))
            else:
                raise NameError("Activation function '{}' is not defined".format(self.activationFunc))

        # Update nodeValueVec.
        outputValueVec = [node.value for node in sorted(outputNodeList, key=lambda node: node.id)]

        return outputValueVec


class ToolboxMBEANN:
    def __init__(self, p_addNode, p_addLink, p_weight, p_bias,
                 mutWeightType, mutWeightScale,
                 mutBiasType, mutBiasScale, addNodeWeight):
        self.p_addNode = p_addNode
        self.p_addLink = p_addLink
        self.p_weight = p_weight
        self.p_bias = p_bias
        self.mutWeightType = mutWeightType
        self.mutWeightScale = mutWeightScale
        self.mutBiasType = mutBiasType
        self.mutBiasScale = mutBiasScale
        self.addNodeWeight = addNodeWeight

        if mutWeightType not in ['gaussian', 'cauchy', 'uniform']:
            print("WARNING: undefined 'mutWeightType', using 'gaussian' instead")
            self.mutWeightType = 'gaussian'

        if mutBiasType not in ['gaussian', 'cauchy', 'uniform']:
            print("WARNING: undefined 'mutBiasType', using 'gaussian' instead")
            self.mutBiasType = 'gaussian'

    def mutateWeightValue(self, ind):
        for operon in ind.operonList:
            for link in operon.linkList:
                if random.random() < self.p_weight:
                    if self.mutWeightType == 'gaussian':
                        link.weight += random.gauss(0.0, self.mutWeightScale)
                    elif self.mutWeightType == 'cauchy':
                        link.weight += self.mutWeightScale * math.tan(math.pi * (random.random() - 0.5))
                    elif self.mutWeightType == 'uniform':
                        link.weight = random.uniform(ind.minWeight, ind.maxWeight)
                    link.weight = np.clip(link.weight, ind.minWeight, ind.maxWeight)

    def mutateBiasValue(self, ind):
        for operon in ind.operonList:
            for node in operon.nodeList:
                if node.type != 'input' and random.random() < self.p_bias:
                    if self.mutBiasType == 'gaussian':
                        node.bias += random.gauss(0.0, self.mutBiasScale)
                    elif self.mutBiasType == 'cauchy':
                        node.bias += self.mutBiasScale * math.tan(math.pi * (random.random() - 0.5))
                    elif self.mutBiasType == 'uniform':
                        node.bias = random.uniform(ind.minBias, ind.maxBias)
                    node.bias = np.clip(node.bias, ind.minBias, ind.maxBias)

    def mutateAddNode(self, ind):
        newOperonID = None

        for operon in ind.operonList:

            # If this operon is the newly generated operon.
            if operon.id == newOperonID:
                break

            if random.random() < self.p_addNode:
                if len(operon.linkList) != 0:
                    randomIndex = random.randint(0, len(operon.linkList) - 1)
                else:
                    break
                oldLink = operon.linkList[randomIndex]

                operon.addLinkToDisabledLinkList(oldLink.fromNodeID, oldLink.toNodeID)
                operon.linkList = np.delete(operon.linkList, randomIndex)

                newNode = Node(id=ind.maxNodeID + 1,
                               type='hidden',
                               bias=ind.addNodeAlpha)
                newLinkA = Link(id=ind.maxLinkID + 1,
                                fromNodeID=ind.maxNodeID + 1,
                                toNodeID=oldLink.toNodeID,
                                weight=oldLink.weight)
                newLinkB = Link(id=ind.maxLinkID + 2,
                                fromNodeID=oldLink.fromNodeID,
                                toNodeID=ind.maxNodeID + 1,
                                weight=self.addNodeWeight)

                ind.maxNodeID += 1
                ind.maxLinkID += 2

                if operon.id == 0:
                    newOperonID = ind.maxOperonID + 1
                    ind.maxOperonID += 1
                    ind.operonList += [Operon(id=newOperonID,
                                              nodeList=np.array([newNode]),
                                              linkList=np.array([newLinkA, newLinkB]))]

                    disabledLinkList = []
                    if ind.isReccurent == True:
                        for nodeOperon0 in operon.nodeList:
                            if nodeOperon0.type != 'input':
                                disabledLinkList += [(newNode.id, nodeOperon0.id)]
                            disabledLinkList += [(nodeOperon0.id, newNode.id)]
                        disabledLinkList += [(newNode.id, newNode.id)]
                    else:
                        for nodeOperon0 in operon.nodeList:
                            if nodeOperon0.type == 'input':
                                disabledLinkList += [(nodeOperon0.id, newNode.id)]
                            if nodeOperon0.type == 'output':
                                disabledLinkList += [(newNode.id, nodeOperon0.id)]

                    ind.operonList[newOperonID].setDisabledLinkList(disabledLinkList)
                    ind.operonList[newOperonID].deleteLinkFromDisabledLinkList(newLinkA.fromNodeID, newLinkA.toNodeID)
                    ind.operonList[newOperonID].deleteLinkFromDisabledLinkList(newLinkB.fromNodeID, newLinkB.toNodeID)

                else:
                    if ind.isReccurent == True:
                        for nodeOperon0 in ind.operonList[0].nodeList:
                            if nodeOperon0.type != 'input':
                                operon.addLinkToDisabledLinkList(newNode.id, nodeOperon0.id)
                            operon.addLinkToDisabledLinkList(nodeOperon0.id, newNode.id)
                        for node in operon.nodeList:
                            operon.addLinkToDisabledLinkList(newNode.id, node.id)
                            operon.addLinkToDisabledLinkList(node.id, newNode.id)
                        operon.addLinkToDisabledLinkList(newNode.id, newNode.id)

                        operon.deleteLinkFromDisabledLinkList(newLinkA.fromNodeID, newLinkA.toNodeID)
                        operon.deleteLinkFromDisabledLinkList(newLinkB.fromNodeID, newLinkB.toNodeID)

                    else:
                        # TODO: Trying to delete a selected link that is already deleted.
                        for nodeOperon0 in ind.operonList[0].nodeList:
                            if nodeOperon0.type == 'input':
                                operon.addLinkToDisabledLinkList(nodeOperon0.id, newNode.id)
                            if nodeOperon0.type == 'output':
                                operon.addLinkToDisabledLinkList(newNode.id, nodeOperon0.id)

                    operon.nodeList = np.append(operon.nodeList, [newNode])
                    operon.linkList = np.append(operon.linkList, [newLinkA, newLinkB])

    def mutateAddLink(self, ind):
        for operon in ind.operonList:
            if len(operon.disabledLinkList) != 0:
                if random.random() < self.p_addLink:
                    randomIndex = random.randint(0, len(operon.disabledLinkList) - 1)
                    newLinkFromNodeID = operon.disabledLinkList[randomIndex][0]
                    newLinkToNodeID = operon.disabledLinkList[randomIndex][1]
                    newLink = Link(id=ind.maxLinkID + 1,
                                   fromNodeID=newLinkFromNodeID,
                                   toNodeID=newLinkToNodeID,
                                   weight=0.0)
                    ind.maxLinkID += 1
                    operon.deleteLinkFromDisabledLinkList(newLinkFromNodeID, newLinkToNodeID)
                    operon.linkList = np.append(operon.linkList, [newLink])

    def selectionSettings(self, pop, popSize, isMaximizingFit, eliteSize=0):
        pop.sort(key=lambda ind: ind.fitness, reverse=isMaximizingFit)
        self.pop = pop
        self.popSize = popSize
        self.isMaximizingFit = isMaximizingFit
        self.eliteSize = eliteSize

    def preserveElite(self):
        return self.pop[0:self.eliteSize]

    def selectionRandom(self):
        newPop = random.choices(self.pop, k=self.popSize - self.eliteSize)
        newPop.sort(key=lambda ind: ind.fitness, reverse=self.isMaximizingFit)
        newPop = [copy.deepcopy(ind) for ind in newPop]
        return newPop

    def selectionTournament(self, tournamentSize, bestN=1):
        # Select "bestN" individuals from each tournament.
        if self.popSize < tournamentSize:
            raise ValueError("Tournament size larger than population size")
        if tournamentSize < bestN:
            raise ValueError("'bestN' should be smaller than tournament size")
        newPop = []
        while len(newPop) < self.popSize - self.eliteSize:
            tournament = random.sample(self.pop, tournamentSize)
            tournament.sort(key=lambda ind: ind.fitness, reverse=self.isMaximizingFit)
            newPop += tournament[0:bestN]
        newPop = [copy.deepcopy(ind) for ind in newPop]
        return newPop
