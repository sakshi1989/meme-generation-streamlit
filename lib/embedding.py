import json
import os
import torch
from torch import nn

cwd = os.getcwd()
DATA_DIR = f"{cwd}/data-processed"

EMBEDDING_DIMENSIONS = 50
VOCABULARY_PATH = os.path.join(DATA_DIR, "vocabulary.json")
VECTORS_PATH = os.path.join(DATA_DIR, f"vectors{EMBEDDING_DIMENSIONS}.pt")
SPECIAL_TOKENS = {
    'UNK': '<u>',
    'PAD': '<p>',
    'EOS': '<s>',
}

# Read vocabulary from file
with open(VOCABULARY_PATH, "r") as f:
    vocabulary = json.load(f)

# nn.embedding is a sparse layer. It is a lookup table that stores embeddings of a fixed dictionary and size.
# This module is often used to store word embeddings and retrieve them using indices.
embedding = nn.Embedding(len(vocabulary), EMBEDDING_DIMENSIONS)

# We do not want to train the embeddings (as we are using the GloVe embeddings)
embedding.weight.requires_grad = False

# stoi is a dictionary that returns the index of a word in the embeddings vocabulary
embedding.stoi = {word: i for i, word in enumerate(vocabulary)}

# itos is a dictionary that returns the word given the index
embedding.itos = {i: word for i, word in enumerate(vocabulary)}

# load weights
embedding.weight.data = torch.load(VECTORS_PATH)
