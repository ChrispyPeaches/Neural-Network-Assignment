## Description
A multi-layer, fully connected, feed forward neural network of sigmoidal neurons. The network is implmented to train with gradient descent and back propagation. It's intended purpose is to recognize handwritten digits from 28 x 28 images of digits from the MNIST's dataset of handwritten digits
- The amount of layers and the size of each layer are configurable, but to serve its intended purpose:
  - The first layer must be of size 784 (28 x 28 pixels)
  - The final layer must be of size 10 (digits 0-9)
