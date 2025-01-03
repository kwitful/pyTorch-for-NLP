from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# Sample dataset
sentences = [
    "I love PyTorch!",
    "Text preprocessing is essential for NLP.",
    "PyTorch makes deep learning fun and intuitive."
]

# Tokenization
tokenizer = get_tokenizer("basic_english")
tokens = [tokenizer(sentence) for sentence in sentences]

# Build Vocabulary
def yield_tokens(data):
    for sentence in data:
        yield tokenizer(sentence)

vocab = build_vocab_from_iterator(yield_tokens(sentences), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Convert Tokens to Indices
token_indices = [[vocab[token] for token in sentence] for sentence in tokens]

# Padding
max_len = max(len(sentence) for sentence in token_indices)
padded_sequences = [
    sentence + [vocab["<pad>"]] * (max_len - len(sentence))
    for sentence in token_indices
]

# Output
print("Tokens:", tokens)
print("Vocabulary:", vocab.get_stoi())  # Print vocabulary with indices
print("Token Indices:", token_indices)
print("Padded Sequences:", padded_sequences)