'''
MBEANN settings for solving XOR.
'''

class SettingsEA:

    # --- Evolutionary algorithm settings. --- #
    popSize = 100
    maxGeneration = 300  # 0 to (max_generation - 1)
    isMaximizingFit = False
    eliteSize = 0
    tournamentSize = 5
    tournamentBestN = 1  # Select best N individuals from each tournament.


class SettingsMBEANN:

    # --- Neural network settings. --- #
    inSize = 2
    outSize = 1
    hidSize = 0

    isReccurent = False

    # initialConnection: (0.0, 1.0]
    # 1.0 for initialization using the fully-connected topology.
    # Between 0.0 to 1.0 for partial connections.
    initialConnection = 1.0

    # initialWeightType: 'uniform' or 'gaussian'
    # uniform  - Uniform random numbers between minWeight to maxWeight.
    # gaussian - Sampled from Gaussian distribution with initialMean and initialGaussSTD.
    #            Weights out of the range [minWeight, maxWeight] are clipped.
    initialWeightType = 'gaussian'
    initialMean = 0.0
    initialGaussSTD = 0.5
    maxWeight = 5.0
    minWeight = -5.0

    # Bias settings.
    initialBiasType = 'gaussian'
    initialBiasMean = 0.0
    initialBiasGaussSTD = 0.5
    maxBias = 5.0
    minBias = -5.0


    # --- Mutation settings. --- #
    # Probability of mutations.
    p_addNode = 0.05
    p_addLink = 0.3
    p_weight = 1.0
    p_bias = 1.0

    # Parameter settings of mutations.
    weightMutationGaussStd = 0.5
    biasMutationGaussStd = 0.5


    # --- Activation function settings. --- #
    activationFunc = 'sigmoid'  # 'sigmoid' or 'tanh'
    addNodeWeightValue = 1.0

    # Recommended settings for 'sigmoid':
    actFunc_Alpha = 0.5 * addNodeWeightValue
    actFunc_Beta = 4.629740 / addNodeWeightValue

    # Recommended settings for 'tanh':
    # actFunc_Alpha = 0.0
    # actFunc_Beta = 1.157435 / addNodeWeightValue
