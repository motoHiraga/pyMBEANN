'''
Example of MBEANN in Python for solving the double pole balancing problem.

Details of MBEANN could be found in the following:
    K. Ohkura, et al., "MBEANN: Mutation-Based Evolving Artificial Neural Networks",
    ECAL 2007, pp. 936-945, 2007.
'''

import math
import multiprocessing
import os
import pickle
import random
import time

import numpy as np

import examples.cart2pole.cart_two_pole_base as cart
from examples.cart2pole.settings import SettingsEA, SettingsMBEANN
from mbeann.base import Individual, ToolboxMBEANN
from mbeann.visualize import visualizeIndividual


def evaluateIndividual(ind):
    '''
    Fitness function for the double pole balancing problem.
    Designed based on Gruau et al., 1996.

    See the inputs in "cart_two_pole_base.run_simulation"
    to check whether or not the simulation uses velocity inputs.
    '''
    max_bal_steps = 1000
    fitness = cart.run_simulation_gruau(ind, max_bal_steps)
    return fitness,


# def evaluateIndividual(ind):
#     '''
#     The double pole balancing problem.
#     This is a fitness function in the book
#     "Hands-On Neuroevolution with Python" published by Packt.
#
#     Original code:
#     https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter4/cart_two_pole.py
#
#     MIT License
#     Copyright (c) 2019 Packt
#     https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/LICENSE
#     '''
#     """
#     Evaluates fitness of the genome that was used to generate
#     provided net
#     Arguments:
#         net: The feed-forward neural network generated from genome
#         max_bal_steps: The maximum number of time steps to
#             execute simulation.
#     Returns:
#         The phenotype fitness score in range [0, 1]
#     """
#     # First we run simulation loop returning number of successful
#     # simulation steps
#     steps = cart.run_simulation(ind, max_bal_steps)
#
#     if steps == max_bal_steps:
#         # the maximal fitness
#         return 1.0,
#     elif steps == 0: # needed to avoid math error when taking log(0)
#         # the minimal fitness
#         return 0.0,
#     else:
#         # we use logarithmic scale because most cart-pole runs fails
#         # too early - within ~100 steps, but we are testing against
#         # 100'000 balancing steps
#         log_steps = math.log(steps)
#         log_max_steps = math.log(max_bal_steps)
#         # The loss value is in range [0, 1]
#         error = (log_max_steps - log_steps) / log_max_steps
#         # The fitness value is a complement of the loss value
#         fitness = 1.0 - error
#         return fitness,

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

    randomSeed = 0  # int(time.time())
    random.seed(randomSeed)
    st = random.getstate()

    data_dir = os.path.join(os.path.dirname(__file__), 'results_cart_{}'.format(randomSeed))
    os.makedirs(data_dir, exist_ok=True)

    with open('{}/random_state.pkl'.format(data_dir), mode='wb') as out_pkl:
        # Saving the random state just in case.
        pickle.dump(st, out_pkl)

    if numProcesses > 1:
        pool = multiprocessing.Pool(processes=numProcesses)

    pop = [Individual(inputSize=SettingsMBEANN.inSize, 
                      outputSize=SettingsMBEANN.outSize, 
                      hiddenSize=SettingsMBEANN.hidSize,
                      initialConnection=SettingsMBEANN.initialConnection,
                      maxWeight=SettingsMBEANN.maxWeight, 
                      minWeight=SettingsMBEANN.minWeight, 
                      initialWeightType=SettingsMBEANN.initialWeightType,
                      initialWeightMean=SettingsMBEANN.initialWeighMean, 
                      initialWeightScale=SettingsMBEANN.initialWeightScale,
                      maxBias=SettingsMBEANN.maxBias, 
                      minBias=SettingsMBEANN.minBias, 
                      initialBiasType=SettingsMBEANN.initialBiasType,
                      initialBiasMean=SettingsMBEANN.initialBiasMean, 
                      initialBiasScale=SettingsMBEANN.initialBiasScale,
                      maxStrategy=SettingsMBEANN.maxStrategy,
                      minStrategy=SettingsMBEANN.minStrategy,
                      initialStrategy=SettingsMBEANN.initialStrategy,
                      isRecurrent=SettingsMBEANN.isRecurrent,
                      activationFunc=SettingsMBEANN.activationFunc,
                      addNodeBias=SettingsMBEANN.actFuncBias, 
                      addNodeGain=SettingsMBEANN.actFuncGain) 
                      for i in range(popSize)]
    tools = ToolboxMBEANN(p_addNode=SettingsMBEANN.p_addNode, 
                          p_addLink=SettingsMBEANN.p_addLink,
                          p_weight=SettingsMBEANN.p_weight, 
                          p_bias=SettingsMBEANN.p_bias,
                          mutWeightType=SettingsMBEANN.weightMutationType, 
                          mutWeightScale=SettingsMBEANN.weightMutationScale,
                          mutBiasType=SettingsMBEANN.biasMutationType, 
                          mutBiasScale=SettingsMBEANN.biasMutationScale,
                          addNodeWeight=SettingsMBEANN.addNodeWeightValue)

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
        with open('{}/data_ind_gen{:0>4}.pkl'.format(data_dir, gen), mode='wb') as out_pkl:
            pop.sort(key=lambda ind: ind.fitness, reverse=isMaximizingFit)
            pickle.dump(pop[0], out_pkl)

        visualizeIndividual(pop[0], '{}/mbeann_ind_gen{:0>4}.pdf'.format(data_dir, gen))

        tools.selectionSettings(pop, popSize, isMaximizingFit, eliteSize)

        if eliteSize > 0:
            elite = tools.preserveElite()

        # pop = tools.selectionRandom()
        pop = tools.selectionTournament(tournamentSize, tournamentBestN)

        for i, ind in enumerate(pop):
            tools.mutateWeightAndBiasValue(ind)
            # tools.mutateWeightValue(ind)
            # tools.mutateBiasValue(ind)
            tools.mutateAddNode(ind)
            tools.mutateAddLink(ind)

        if eliteSize > 0:
            pop = elite + pop
