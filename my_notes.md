# Sequence of Lectures

1. bigram_model: using a simple bigram model to generate text

- loss function: negative log likelihood
- not a good model, the names produced does not resemble "names"

2. a very simple NN: using a neural network to generate text

- We see similar results compared to first because NN is very simple

3. MLP

- lit: A Neural Probabilistic Language Model
  - 17000 vocab associated with a point in vector space (30 features eg)
  - components:
    - lookup table: C 17000 x 30
    - index of incoming word
    - input layer: 90 neurons total (30 neurons for 3 words)
    - hidden layer: arbitrary number of neurons (100 neurons)
      - hyperparameter (this term means can be as large as you want)
      - fully connected with input layer
    - tanh activation function
    - output layer (expensive layer: also fully connected with hidden layer)
    - softmax (exponentiated, normalized)
    - pluck out probability of word and compare to actual word
    - backpropagation optimization

