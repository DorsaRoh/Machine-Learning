
import numpy as np
import gensim.downloader as api

# Embedding matrix
embedding_model = api.load('glove-wiki-gigaword-300')

def get_embedding(word: str, embedding_model) -> np.ndarray:
    """
    Retrieve the embedding vector for a given word.

    Args:
        word (str): The word to be embedded.
        embedding_model: Pre-trained embedding model.

    Returns:
        np.ndarray: Embedding vector of the word.
    """
    if word in embedding_model:
        return embedding_model[word]
    else: # if word is not in vocab
        return np.zeros(embedding_model.vector_size)

def tokenize_and_embed(word: str, embedding_model) -> list:
    """
    Tokenize the input sentence and obtain embeddings for each token.

    Args:
        word (str): Input sentence.
        embedding_model: Pre-trained embedding model.

    Returns:
        list: List of embedding vectors for each token.
    """
    tokens = word.split()  # split input sentence into words (tokens)
    embeddings = np.array([get_embedding(word, embedding_model) for word in tokens])
    return embeddings

def add_positional_encoding(embeddings: np.ndarray) -> np.ndarray:
    """
    Add positional encoding to the input embeddings.

    Args:
        embeddings (np.ndarray): Input embeddings.

    Returns:
        np.ndarray: Embeddings with added positional encodings.
    """
    sequence_len = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]

    # Initialize positional encoding matrix
    pos_enc_matrix = np.zeros((sequence_len, embedding_dim))

    # Calculate the positional encodings
    for pos in range(sequence_len):
        for i in range(embedding_dim): 
            # even index
            if i % 2 == 0: 
                pos_enc_matrix[pos, i] = np.sin(pos / (10000 ** (i/embedding_dim)))
            else: # odd index
                pos_enc_matrix[pos, i] = np.cos(pos/(10000**(i/ embedding_dim)))

    # Add positional encodings
    embeddings_with_pos = embeddings + pos_enc_matrix
    return embeddings_with_pos

