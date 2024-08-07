# pyMBEANN
An implementation of Mutation-Based Evolving Artificial Neural Network (MBEANN) written in Python.

## Background
MBEANN is an alternative algorithm for *Topology and Weight Evolving Artificial Neural Networks (TWEANNs)*, which evolves both the values of the synaptic weights and the topological structure of neural networks.
For details, see the following research papers:

- M. Hiraga, et al., "Improving the performance of mutation-based evolving artificial neural networks with self-adaptive mutations," PLOS ONE, Vol. 19, No.7, e0307084, 2024.<br/>
  DOI: [10.1371/journal.pone.0307084](https://doi.org/10.1371/journal.pone.0307084)
- K. Ohkura, et al., "MBEANN: Mutation-Based Evolving Artificial Neural Networks," ECAL 2007, pp. 936-945, 2007.<br/>
  DOI: [10.1007/978-3-540-74913-4_94](https://doi.org/10.1007/978-3-540-74913-4_94)
- M. Hiraga and K. Ohkura, "Topology and Weight Evolving Artificial Neural Networks in Cooperative Transport by a Robotic Swarm," Artificial Life and Robotics, Vol. 27, No.2, pp. 324-332, 2022. <br/>
  DOI: [10.1007/s10015-021-00716-9](https://doi.org/10.1007/s10015-021-00716-9)

## Requirements
- The pyMBEANN is developed and tested with Python 3.
- Requires [NumPy](https://numpy.org).
- Requires [matplotlib](https://matplotlib.org) and [NetworkX](https://networkx.org) for visualizing MBEANN individuals.

## Installation
#### Installing the latest version using pip
```
pip install git+https://github.com/motoHiraga/pyMBEANN
```

#### Installing from source
```
git clone https://github.com/motoHiraga/pyMBEANN.git
cd pyMBEANN
python setup.py install
```

## Examples
#### XOR
- Run evolution with the following:
```
python3 -m examples.xor.main
```

#### Double Pole Balancing Problem
- Run evolution with the following:
```
python3 -m examples.cart2pole.main
```
- Run the animation with the following (see animation.py to check which result you are using):
```
python3 -m examples.cart2pole.animation
```

#### Gym Environment
- Install [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
- Run evolution with the following:
```
python3 -m examples.gym.main
```
- Run the animation with the following (see animation.py to check which result you are using):
```
python3 -m examples.gym.animation
```

#### Other tools
- "vis_fit.py" is an example for visualizing the fitness transition throughout the evolutionary process (requires [pandas](https://pandas.pydata.org)).
- "vis_ind.py" is an example for custom visualization of an MBEANN individual.
