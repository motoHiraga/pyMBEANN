'''
Example of MBEANN in Python solving XOR.
'''

import numpy as np
import multiprocessing
import pickle
import os
import time
import random

from mbeann.base import Individual, ToolboxMBEANN
from mbeann.visualize import visualizeIndividual

from examples.xor.settings import SettingsMBEANN, SettingsEA

def evaluateIndividual(ind):

    # XOR settings
    # Third value in the inputsSet is for the bias.
    # inputsSet = np.array([[0.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5]])

    # XOR without bias inputs.
    inputsSet = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    outputsSet = np.array([[0.0], [1.0], [1.0], [0.0]])

    outputsFromNetwork = []
    for inputs in inputsSet:
        outputsFromNetwork += [ind.calculateNetwork(inputs)]

    fitness = 0.0
    for a, b in zip(outputsSet, outputsFromNetwork):
        fitness += np.abs(a - b)
    return fitness


if __name__ == '__main__':

    # Number of worker processes to run evolution.
    numProcesses = multiprocessing.cpu_count()

    # Evolutionary algorithm settings.
    popSize = SettingsEA.popSize
    maxGeneration = SettingsEA.maxGeneration
    isMaximizingFit = SettingsEA.isMaximizingFit
    eliteSize = SettingsEA.eliteSize
    tournamentSize = SettingsEA.tournamentSize
    tournamentBestN = SettingsEA.tournamentBestN

    randomSeed = 0 # int(time.time())
    random.seed(randomSeed)
    st = random.getstate()

    data_dir = os.path.join(os.path.dirname(__file__), 'results_xor_{}'.format(randomSeed))
    os.makedirs(data_dir, exist_ok=True)

    with open('{}/random_state.pkl'.format(data_dir), mode='wb') as out_pkl:
        # Saving the random state just in case.
        pickle.dump(st, out_pkl)

    if numProcesses > 1:
        pool = multiprocessing.Pool(processes=numProcesses)

    pop = [Individual(SettingsMBEANN.inSize, SettingsMBEANN.outSize, SettingsMBEANN.hidSize,
                      SettingsMBEANN.initialConnection,
                      SettingsMBEANN.maxWeight, SettingsMBEANN.minWeight, SettingsMBEANN.initialWeightType,
                      SettingsMBEANN.initialMean, SettingsMBEANN.initialGaussSTD,
                      SettingsMBEANN.maxBias, SettingsMBEANN.minBias, SettingsMBEANN.initialBiasType,
                      SettingsMBEANN.initialBiasMean, SettingsMBEANN.initialBiasGaussSTD,
                      SettingsMBEANN.isReccurent, SettingsMBEANN.activationFunc,
                      SettingsMBEANN.actFunc_Alpha, SettingsMBEANN.actFunc_Beta) for i in range(popSize)]
    tools = ToolboxMBEANN(SettingsMBEANN.p_addNode, SettingsMBEANN.p_addLink,
                          SettingsMBEANN.p_weight, SettingsMBEANN.p_bias,
                          SettingsMBEANN.weightMutationGaussStd, SettingsMBEANN.biasMutationGaussStd,
                          SettingsMBEANN.addNodeWeightValue)

    log_stats = ['Gen', 'Mean', 'Std', 'Max', 'Min']
    with open('{}/log_stats.pkl'.format(data_dir), mode='wb') as out_pkl:
        pickle.dump(log_stats, out_pkl)


    for gen in range(maxGeneration):
        print("------")
        print("Gen {}".format(gen))

        if numProcesses > 1:
            fitnessValues = pool.map(evaluateIndividual, pop)
        else:
            fitnessValues = []
            for ind in pop:
                fitnessValues += [evaluateIndividual(ind)]

        for ind, fit in zip(pop, fitnessValues):
            ind.fitness = fit[0]

        log_stats = [gen, np.mean(fitnessValues), np.std(fitnessValues),
                     np.max(fitnessValues), np.min(fitnessValues)]

        with open('{}/log_stats.pkl'.format(data_dir), mode='ab') as out_pkl:
            pickle.dump(log_stats, out_pkl)

        print("Mean: " + str(np.mean(fitnessValues)) +
              "\tStd: " + str(np.std(fitnessValues)) +
              "\tMax: " + str(np.max(fitnessValues)) +
              "\tMin: " + str(np.min(fitnessValues)))

        # Save the best individual.
        with open('{}/data_ind_gen{:0>4}.pkl'.format(data_dir, gen),
                  mode='wb') as out_pkl:
            pop.sort(key=lambda ind: ind.fitness, reverse=isMaximizingFit)
            pickle.dump(pop[0], out_pkl)

        visualizeIndividual(
            pop[0], '{}/mbeann_ind_gen{:0>4}.pdf'.format(data_dir, gen))

        tools.selectionSettings(pop, popSize, isMaximizingFit, eliteSize)

        if eliteSize > 0:
            elite = tools.preserveElite()

        # pop = tools.selectionRandom()
        pop = tools.selectionTournament(tournamentSize, tournamentBestN)

        for i, ind in enumerate(pop):
            tools.mutateAddNode(ind)
            tools.mutateAddLink(ind)
            tools.mutateWeightValue(ind)
            tools.mutateBiasValue(ind)

        if eliteSize > 0:
            pop = elite + pop
