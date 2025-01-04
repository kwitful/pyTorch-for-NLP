# Tutorial: Working with Word Embeddings in PyTorch

Just like tokenizers, almost every NLP/AI workflow out there will have its own embedding process. I put together this tutorial not to give you a one-size-fits-all solution (which is not quite feasible), but to help you comprehend this topic with an entry into fundamentals. 

---

## 1. Introduction to Word Embeddings
Word embeddings are numerical representations of words, capturing their meaning and relationships with other words. Instead of treating words as independent entities, embeddings enable models to understand context and semantic similarity.

### Why Are Word Embeddings Important?
- **Captures Meaning**: Words with similar meanings have similar vector representations.
- **Reduces Dimensionality**: Instead of one-hot encoding (huge sparse vectors), embeddings create compact, dense representations.
- **Improves Model Efficiency**: Pre-trained embeddings save training time and improve accuracy, especially with small datasets.

Examples of embeddings include:
1. **Pre-trained Embeddings** - Ready-to-use embeddings like GloVe and Word2Vec.
2. **Custom Embedding Layers** - Learned embeddings during training.

---

## 2. Pre-trained Embeddings (GloVe)

### Loading GloVe Vectors
```python
from torchtext.vocab import GloVe

glove = GloVe(name='6B', dim=50)  # Load 50-dimensional GloVe embeddings
print(glove['king'][:10])  # Example for 'king'
```
### Code Explanation:
- **`GloVe`**: Loads pre-trained GloVe embeddings from TorchText.
- **`name='6B'`**: Specifies the 6-billion token dataset.
- **`dim=50`**: Uses 50-dimensional vectors for each word.
- **`glove['king']`**: Fetches the vector representation of the word "king." The first 10 values of the vector are printed as an example.

### Why Use Pre-trained Embeddings?
1. **Saves Time**: Already trained on large datasets.
2. **Captures Context**: Embeddings reflect semantic relationships between words.
3. **Generalization**: Works well for unseen or related data.

### Error Handling:
If loading fails, we print an error message and proceed without pre-trained embeddings:
```python
except Exception as e:
    print(f"Error loading GloVe: {e}")
    glove = None
```

---

## 3. Custom Embedding Layers
### Defining Custom Embeddings
```python
import torch
import torch.nn as nn

VOCAB_SIZE = 10000
EMBEDDING_DIM = 300
embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

word_indices = torch.randint(0, VOCAB_SIZE, (5,))
embedded_words = embedding_layer(word_indices)
print(embedded_words.shape)  # Output shape: (5, 300)
```
### Code Explanation:
1. **`nn.Embedding`**: Creates a layer that maps integer indices to dense vectors.
2. **`VOCAB_SIZE`**: Total number of unique words (vocabulary size).
3. **`EMBEDDING_DIM`**: Each word is represented by a 300-dimensional vector.
4. **`torch.randint`**: Generates random integers to simulate word indices.
5. **`embedding_layer(word_indices)`**: Maps word indices to their corresponding embeddings.

---

## 4. Building a Sentiment Analysis Model
### Model Definition
```python
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings=None):
        super(SentimentModel, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
```
### Code Explanation:
1. **Constructor (`__init__`)**: Initializes model components.
2. **`pretrained_embeddings`**: Allows using GloVe or other embeddings.
3. **`freeze=False`**: Enables fine-tuning the embeddings.
4. **`nn.LSTM`**: LSTM processes sequences, learning dependencies between words.
5. **`nn.Linear`**: Fully connected layer maps LSTM outputs to predictions.

### Forward Pass
```python
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        last_hidden = torch.gather(output, 1, (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.size(2))).squeeze(1)
        logits = self.fc(last_hidden)
        return logits
```
### Explanation:
1. **Embedding Layer**: Converts input words to dense vectors.
2. **Packing Sequences**: Handles varying sequence lengths efficiently.
3. **LSTM Layer**: Processes sequences for contextual information.
4. **Gather Last Hidden State**: Selects the last relevant output for each sequence.
5. **Fully Connected Layer**: Produces logits (predictions).

---

## 5. Handling Variable-length Sequences
### Padding and Packing
```python
from torch.nn.utils.rnn import pad_sequence
input_lengths = torch.randint(10, 51, (32,))
input_data = [torch.randint(0, VOCAB_SIZE, (length,)) for length in input_lengths]
padded_input = pad_sequence(input_data, batch_first=True)
```
### Explanation:
1. **Padding**: Pads shorter sequences to match the longest.
2. **Sorting Lengths**: Required for efficient processing by LSTM.

---

## 6. Model Training
### Training Loop
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dummy_labels = torch.randint(0, OUTPUT_DIM, (32,))

optimizer.zero_grad()
loss = criterion(output, dummy_labels)
loss.backward()
optimizer.step()
```
### Explanation:
1. **Loss Function**: Computes error between predictions and labels.
2. **Optimizer**: Updates model weights to reduce loss.
3. **Gradient Backpropagation**: Adjusts weights based on computed gradients.

---

## Conclusion 
You now have a fundamental understanding of word embeddings. In the coming chapters, we will learn how to build cool stuff with neural networks. Keep on nerding!


