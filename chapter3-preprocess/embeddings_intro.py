import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pad_sequence

# Define constants
VOCAB_SIZE = 10000
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
OUTPUT_DIM = 2  # Binary classification (e.g., positive/negative)
BATCH_SIZE = 32
SEQ_LEN = 50

# 1. Pre-trained Embeddings (GloVe)
print("--- Pre-trained Embeddings ---")
try:
    glove = GloVe(name='6B', dim=50)
    print("Vector for 'king':", glove['king'][:10], "...")  # Print a snippet
except Exception as e:
    print(f"Error loading GloVe: {e}")
    glove = None  # Handle the case where GloVe fails to load

# 2. Custom Embedding Layer (Example)
print("\n--- Custom Embedding Layer ---")
embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
word_indices = torch.randint(0, VOCAB_SIZE, (5,))  # More indices for better demonstration
embedded_words = embedding_layer(word_indices)
print("Custom embeddings shape:", embedded_words.shape)

# 3. Sentiment Analysis Model
print("\n--- Sentiment Analysis Model ---")

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings=None):
        super(SentimentModel, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False) # Use Pretrained embeddings
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths): # Add lengths parameter
        embedded = self.embedding(x)

        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, _) = self.lstm(packed_embedded)
        # Unpack padded sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Use the last hidden state for each sequence
        # Get the last hidden state for each sequence
        last_hidden = torch.gather(output, 1, (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.size(2))).squeeze(1)

        logits = self.fc(last_hidden)
        return logits

# Initialize the model (with or without pre-trained embeddings)
if glove is not None:
    pretrained_embeddings = glove.vectors
    model = SentimentModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, pretrained_embeddings)
else:
    model = SentimentModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    print("Model initialized without pre-trained embeddings.")


# Example input with variable lengths (demonstrates padding and packing)
input_lengths = torch.randint(10, SEQ_LEN + 1, (BATCH_SIZE,))  # Random lengths
input_data = [torch.randint(0, VOCAB_SIZE, (length,)) for length in input_lengths]
padded_input = pad_sequence(input_data, batch_first=True)
input_lengths, indices = torch.sort(input_lengths, descending=True)
padded_input = padded_input[indices]

output = model(padded_input, input_lengths)
print("Model output shape:", output.shape)

# Example Training (Illustrative)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dummy_labels = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))

optimizer.zero_grad()
loss = criterion(output, dummy_labels)
loss.backward()
optimizer.step()

print("Training step complete")