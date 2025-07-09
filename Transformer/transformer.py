import numpy as np
from attention import MultiHeadAttention, softmax
from embed import tokenize_and_embed, add_positional_encoding, embedding_model
import random

class Transformer:
    def __init__(self, embedding_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.output_projection = np.random.randn(embedding_dim, embedding_dim) 
        self.output_projection = self.output_projection * np.sqrt(1. / embedding_dim) # scale values down

    def forward(self, embeddings):
        # add positional encoding 
        embeddings_with_pos = add_positional_encoding(embeddings) 

        # output of MultiHeadAttention class
        attention_output = self.multi_head_attention.forward(embeddings_with_pos)

        # apply final linear transformation
        output = self.linear_transformation(attention_output, self.output_projection)
        return output

    # calculate linear transformation
    def linear_transformation(self, x, weight_matrix):
        return np.dot(x, weight_matrix)

    # calulate next token
    def predict_next_word(self, sentence, temperature, top_k=5):
        # tokenize and embed input sentence
        embeddings = tokenize_and_embed(sentence, embedding_model)
        output = self.forward(embeddings)
        
        # apply softmax to get probabilities
        probs = softmax(output[-1] / temperature)
        
        # sample from the top-k words instead of greedy argmax
        top_k_indices = np.argsort(probs)[-top_k:]
        chosen_index = random.choice(top_k_indices)
        next_word = embedding_model.index_to_key[chosen_index]
        
        return next_word
    
    # complete the sentence from given input 
    def complete_sentence(self, sentence, max_length=20, temperature=1.0):
        words = sentence.split()
        for _ in range(max_length - len(words)):
            next_word = self.predict_next_word(" ".join(words), temperature)
            if next_word == "<EOS>":  # assuming <EOS> is the end of sequence token
                break
            words.append(next_word)
        return " ".join(words)