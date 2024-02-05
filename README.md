## Description
A multi-layer, fully connected, feed forward neural network of sigmoidal neurons. The network is implemented to train with gradient descent and back propagation. It's intended purpose is to recognize handwritten digits from 28 x 28 images of digits from the MNIST's dataset of handwritten digits.
- The amount of layers and the size of each layer are configurable, but to serve its intended purpose:
  - The first layer must be of size 784 (28 x 28 pixels)
  - The final layer must be of size 10 (digits 0-9)
 
- The user has the option to:
  - Train a network using stochastic gradient descent
  - Test with ASCII output
      - For all test cases or
      - Only for misclassified test cases
  - Test with Accuracy results output for both the training and testing datasets
  - Save and load a network's state
  - Exit the program

## Usage
This is a CLI text-based program. The user can interact with the program from the CLI in which the program is run. The user will be presented with the available options and explanations of each option

#### Setup
- The `mnist_train.csv` and `mnist_test.csv` files are expected to be in the directory of the program being executed
- Weights files are saved and read from the directory of the program being executed

#### Notes on Training
- The program outputs weight files with the following name format: "weights-{yyyy-MM-dd__hh-mm-ssa}.txt"

#### Notes on Testing
- You must load a weights file to test the neural network
- After this is done, the program will present the user with options for testing 
