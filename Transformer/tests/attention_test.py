import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import SelfAttention, MultiHeadAttention
import numpy as np

embedding_dim = 8 
num_heads = 2
sequence_length = 4

# Dummy embedding matrix
dummy_embeddings = np.random.rand(sequence_length, embedding_dim)

# SELF ATTENTION
self_attention = SelfAttention(embedding_dim)
self_attention_output = self_attention.forward(dummy_embeddings)
print("SelfAttention Output:")
print(self_attention_output)
print("Shape:", self_attention_output.shape)

# MULTI ATTENTION
multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
multi_head_output = multi_head_attention.forward(dummy_embeddings)
print("MultiHeadAttention Output:")
print(multi_head_output)
print("Shape:", multi_head_output.shape)