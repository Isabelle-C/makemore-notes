This repo contains my notes from Andrej Karpathy's lectures.

# 0. Concepts to start with

```mermaid
---
title: Learning map
---
flowchart TB
    id1[function backpropagation + activation] --> id2[neuron backprop] --> id3["NN backprop + loss function (update gradient every step)"]

```

# Lectures

## 1. bigram_model: using a simple bigram model to generate text

see notebook [here](/notebooks/bigram_model.ipynb)

- loss function: negative log likelihood
- not a good model, the names produced does not resemble "names"

## 2. a very simple NN: using a neural network to generate text
see notebook [here](/notebooks/NN.ipynb)
- We see similar results compared to first because NN is very simple

## 3. MLP
see notebook [here](/notebooks/MLP.ipynb) and [here](/notebooks/MLP_full_data.ipynb)

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

# Part 3 video notes
see [here](/notebooks/build_makemore_batchnorm.ipynb)

## The initial loss is too high

we would expect uniform distribution of next-letter probability, i.e. log of 1/27

- the shape of the loss looks like a hockey stick
- the initial iterations are squashing down the logits
- taking away that drastic drop in loss by making the weights and biases closer to 0 so less likely of vanishing gradient
```python
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0
```

- Note W2 cannot be all 0!
- the multiplier = the value of the sd of W2

## Tanh

- it is a squashing function
- if the value is 1 or -1 in backpropagation, the gradient is 0 so backpropagation stops: "dead neuron"
  - neuron output is all 1 or -1
![histogram](/figs/histogram.png)

- i.e. one column completely white
![white_column](/figs/white_column.png)

- same issue with sigmoid and relu
  - alternative: leaky relu or elu

![activation_functions](/figs/af.png)

- can happen at initialization or optimization (with high learning rate)

`solution`

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
```

- deeper the network the more significant the problem
- the multiplication above is trying to preserve the guassian distribution of the input (i.e. keeping a small standard deviation)
  - the factor is square root of (5/3)/ (n_embd \* block_size)
![init](/figs/initialization.png)
  - Kaiming initialization

![kaiming](/figs/kaiming.png)

## Batch normalization
- based on [paper](https://arxiv.org/abs/1502.03167)
- normalize the hidden layer
```python
hpreact = embcat @ W1 #+ b1 # hidden layer pre-activation
```

1. Calculate the mean and standard deviation of the hidden layer
```python
bnmeani = hpreact.mean(0, keepdim=True)
bnstdi = hpreact.std(0, keepdim=True)
```
- mean: taking mean of everything in the batch (average of any neuron activation)
- std: standard deviation of the batch
- remember we only want this at initialization, not during training

2. Scale and shift! (offset with gain and bias)
- note `bngain` and `bnbias` below
    - initialize
```python
# BatchNorm parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))
```
    - scaling
```python
  hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
  with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
```
The batches kind of create a "jittering" effect because which samples in batch affect h. But in a way this introduces entropy and help prevent model overfitting.

Final training code below:

```python
# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
  
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
  # forward pass
  emb = C[Xb] # embed the characters into vectors
  embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
  # Linear layer
  hpreact = embcat @ W1 #+ b1 # hidden layer pre-activation
  # BatchNorm layer
  # -------------------------------------------------------------
  bnmeani = hpreact.mean(0, keepdim=True)
  bnstdi = hpreact.std(0, keepdim=True)
  hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
  with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
  # -------------------------------------------------------------
  # Non-linearity
  h = torch.tanh(hpreact) # hidden layer
  logits = h @ W2 + b2 # output layer
  loss = F.cross_entropy(logits, Yb) # loss function
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
```

The `bnmean_running` and `bnstd_running` are supposed to be estimate for `bnmean` and `bnstd` which could be caliberated after training but nobody wants to do that.
```
# calibrate the batch norm at the end of training

with torch.no_grad():
  # pass the training set through
  emb = C[Xtr]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1 # + b1
  # measure the mean/std over the entire training set
  bnmean = hpreact.mean(0, keepdim=True)
  bnstd = hpreact.std(0, keepdim=True)
```
