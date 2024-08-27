import numpy as np
from embed import get_embedding

def softmax(scores):
    """
    Apply softmax to normalize attention scores.

    Args:
        scores (np.ndarray): Attention scores.

    Returns:
        np.ndarray: Normalized attention scores.
    """
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # numerical stability, and normalizes across each row (i.e. across all key vectors for each query)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)



class SelfAttention:
    """
    Self-Attention mechanism to compute attention scores and apply them to value vectors.

    Attributes:
        embedding_dim (int): Dimension of the embeddings.
        W_q (np.ndarray): Weight matrix for the Query.
        W_k (np.ndarray): Weight matrix for the Key.
        W_v (np.ndarray): Weight matrix for the Value.
    """

    def __init__(self, embedding_dim):
        """
        Initialize the SelfAttention mechanism.

        Args:
            embedding_dim (int): Dimension of the embeddings.
        """
        self.embedding_dim = embedding_dim

        # Initialize weight matrices (with small random values)
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(1. / embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(1. / embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(1. / embedding_dim)

    def forward(self, embeddings, mask=None):
        """
        Forward pass through the Self-Attention mechanism.

        Args:
            embeddings (np.ndarray): Input embeddings.
            mask (np.ndarray, optional): Mask to be applied to attention scores.

        Returns:
            np.ndarray: Output after applying attention to value vectors.
        """
        query = np.dot(embeddings, self.W_q)
        key = np.dot(embeddings, self.W_k)
        values = np.dot(embeddings, self.W_v)

        # Calculate attention scores
        attention_scores = self.calculate_attention_score(query, key)

        # Masking
        if mask is not None:
            attention_scores = np.where(mask == 0, -1e9, attention_scores)      # where mask is 0, turns to -infinity. where mask is 1, keeps original values

        # Apply softmax to attention scores
        attention_weights = softmax(attention_scores)

        # Compute weighted sum of value vectors
        output = self.values_weighted_sum(attention_weights, values)

        return output

    def calculate_attention_score(self, query, key):
        """
        Calculate the attention scores based on the Query and Key matrices.

        Args:
            query (np.ndarray): Query matrix.
            key (np.ndarray): Key matrix.

        Returns:
            np.ndarray: Attention scores.
        """
        d_k = key.shape[-1]     # scaling factor to ensure no too large values are fed to softmax (would push softmax into regions where it has extremely small gradients)
        dot = np.dot(query, key.T) # key.T : transpose of the key matrix 
                                    # i.e. flipping the matrix over its diagonal, so that the rows become columns and the colums become rows
        return dot / np.sqrt(d_k)  # scale by the square root of the key dimension
    
    def values_weighted_sum(self, weights, values):
        """
        Calculate the weighted sum of value vectors based on attention weights.

        Args:
            weights (np.ndarray): Attention weights.
            values (np.ndarray): Value vectors.

        Returns:
            np.ndarray: Weighted sum of value vectors.
        """
        return np.dot(weights, values)

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism consisting of multiple self-attention heads.

    Attributes:
        head_dim (int): Dimension of each attention head.
        attention_heads (list): List of SelfAttention instances.
        W_o (np.ndarray): Final transformation matrix.
    """

    def __init__(self, embedding_dim, num_heads):
        """
        Initialize the MultiHeadAttention mechanism.

        Args:
            embedding_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.

        Raises:
            ValueError: If embedding_dim is not divisible by num_heads.
        """
        # `embedding_dim` must be divisible by `num_heads`
            # otherwise, the context window will not be consistent (i.e. the input of each head will be different sizes)
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        
        # Calculate dimension of each head
        self.head_dim = embedding_dim // num_heads
        
        # Initialize heads (instances of self attention class)
        self.attention_heads = [SelfAttention(self.head_dim) for _ in range(num_heads)]
        
        # Final transformation matrix (transform the concatenated outputs back to the original embedding dimension)
        self.W_o = np.random.rand(embedding_dim, embedding_dim)

    def forward(self, embeddings):
        """
        Forward pass through the Multi-Head Attention mechanism.

        Args:
            embeddings (np.ndarray): Input embeddings.

        Returns:
            np.ndarray: Output after applying multi-head attention and final transformation.
        """
        # Split the embeddings into multiple heads
        sequence_length, embedding_dim = embeddings.shape
        split_embeddings = np.reshape(embeddings, (sequence_length, len(self.attention_heads), self.head_dim))

        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head_output = head.forward(split_embeddings[:, i, :])
            head_outputs.append(head_output)
        
        # Concatenate outputs of all heads along the last axis
        concatenated_output = np.concatenate(head_outputs, axis=-1)
        
        # Apply final linear transformation
        output = self.linear_transformation(concatenated_output, self.W_o)
        
        return output

    def linear_transformation(self, concatenated_output, weight_matrix):
        """
        Apply a linear transformation to the concatenated output.

        Args:
            concatenated_output (np.ndarray): Concatenated output from attention heads.
            weight_matrix (np.ndarray): Weight matrix for final transformation.

        Returns:
            np.ndarray: Transformed output.
        """
        return np.dot(concatenated_output, weight_matrix)