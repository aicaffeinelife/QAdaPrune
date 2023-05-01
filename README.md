# Code Accompanying Paper: QAdaPrune

This repo consists of code used to produce results in the QAdaPrune paper.

## Requirements:

The following packages are required for runnning the code:

- `pennylane`
- `numpy`
- `scikit-learn` for performing train/test splits on data.

## Code Map:

- `qnn_iris_expt` consists of code to run QAdaPrune for a QNN on Iris dataset. It can be easily modified to run for other datasets, just be careful that the dimensionality of the dataset matches the number of wires on the quantum device.

- `BaselinePruningRes` notebook consists of code to run a simple identity learning ansatz and observe the effects of QAdaPrune algorithm on barren plateaus.

- `VQE_QAdaPrune` implements the algorithm for estimating the ground state energy of the hydrogen molecule with a UCCSD ansatz. 

