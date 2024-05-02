
# GPT building

see [here](/notebooks/gpt_dev.ipynb)
1. Tokenization: set up encoding and decoding

- [sentencepiece](https://github.com/google/sentencepiece): subword tokenization
- [tiktoken](https://github.com/openai/tiktoken)

2. Embedding:

- nn.Embedding: lookup table (Andrej implemented it as bigram model class)

  - each row is index of charater
  - (B,T,C): stands for batch, time, channel
    - batch: 4 (how many independently processed at once)
    - time: 8 (context window of 8 characters)
    - channel: 65 (65 characters in total)
  - logits: 4 x 8 x 65
  - loss: cross entropy loss
    - need to reshape logits to (B\*T, C)
    ```python
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    ```

  3. Optimizer

  - Adam optimizer
  - previously, just simple backpropagation

  4. Training loop (same thing)

  5. Self-attention

  - Using bag of words (average value of all previous tokens)

