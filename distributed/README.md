# Distribution

Deep learning models (like transformers with attention) require compute-heavy operations. One common example is matrix multiplication, which can become extremely resource-intensive when models are large and process lots of data.

To make this faster and more scalable, we can distribute the computation across multiple devices. Each device performs a portion of the work - such as a slice of a large matrix multiplication - in parallel.

## How?
**Parallelism**. 
<br>
To *distribute* a workload means to run it in **parallel** across multiple devices. There are several kinds of parallelism, with the most common being:

- [Data Parallelism (DP)](#1-data-parallelism-dp) - parallelism over the **data**

- [Tensor Parallelism (TP)](#2-tensor-parallelism-tp) - parallelism over the **model weights**

- [Sequence Parallelism (SP)](#3-sequence-parallelism-sp) - parallelism over the **sequence length**


#### Terminology:
- **Shard**: a **slice** / partition of a tensor or model weight
- **Sharding (verb)**: the act of **splitting** a tensor or weight matrix into **multiple** parts
- **Replicate**: copying the **same** tensor (eg., input data or model weights) onto **multiple** devices
- **Gather**: **collecting distributed data or outputs from all devices into one place** (can precede concatenation)
- **Concatenating**: **joining/aggregating** multiple shards along a dimension to **reconstruct** the **full** tensor
- **Reduce**: **Concatenates** the shards back into the full tensor, **replicated** across all devices (ie. all devices have the full tensor now)

TODO: add link to attention parallelized example at bottom somewhere here (make index maybe?)

### 1. Data Parallelism (DP)

In DP, each device gets a copy of the full model, but works on a different shard of the input data.

Let’s say the input tensor is shaped like:
`tensor = [batch, height, width, dimension]`

Then:

1. **Shard** the tensor along the batch dimension → each device gets one mini-batch:

`sharded_tensor = [batch / num_devices, height, width, dimension]`

2. **Replicate** the model weights across all devices → every device runs the same model

    *To emphasize: Effectively, each device contains a replica of the model and is responsible for computing a shard (of the final output tensor).*

3. 	**Run the model on each device** - each device uses its replicated model to process its own sharded input data independently

3. **Gather** all shards across the devices

4. **Concatenate** the shards to form the final output tensor

> ✅ Use Data Parallelism when model weights fit on a single device, but batch size is large

TODO: add excalidraw illustration here

### 2. Tensor Parallelism (TP)

In Tensor Parallelism, the model weights are too large to fit on a single device. So, instead of sharding the data, we **shard the model (model parameters/weights) itself**.

Steps:

1. **Shard** the model parameters across devices
    - For example, split a large weight matrix along its columns or rows.

2. **Replicate** the input data across all devices (or just the necessary part)

3. Each device performs computations with its shard of the weights

4. **Gather** partial outputs from all devices

5. Depending on how the weights were sharded:
	- [Column-sharded](#1-column-sharding): → Concatenate outputs across the last dimension
	- [Row-sharded](#2-row-sharding): → Reduce (e.g., sum) across the last dimension

6.	Continue to the next layer 

> ✅ Use Tensor Parallelism when model weights are too large for a single device.


TODO: add exclalidraw pic

TODO: add differences between DP and TP

### 3. Sequence Parallelism (SP)

Sequence Parallelism is an extension of Tensor Parallelism. It **shards the input along the sequence length** - especially helpful for operations like **LayerNorm** or **RMSNorm**, which depend on the full sequence.

Why use it?
	- During training of large models, activation memory (i.e., memory used to store intermediate results like hidden states) becomes a bottleneck
	- SP reduces activation memory needs by distributing LayerNorm-like computations across devices, in parallel over the sequence

> ✅ Sequence Parallelism is typically applied only to specific layers like LayerNorm in large-scale TP setups.

TODO: add exclalidraw pic here

---

### Column and Row Sharding

When performing Tensor Parallelism, large weight matrices are split across devices so each device only stores and processes a portion. There are two main ways to do this: Column Sharding and Row Sharding.

#### 1. Column Sharding

Assume we have a weight matrix `W` with shape `[input_dim, output_dim]`.

TODO: add exclalidraw pic here

📌 In column sharding, you split the matrix along the **output** dimension - i.e., the **columns**

TODO: add exclalidaw pic here

For example, let `W = [input_dim, output_dim] = [4, 6]`. If we have 2 devices, then:
- Device 1 gets [4, 3] 
- Device 2 gets [4, 3]
<br>
*→ (last dimension is halved/sharded)*


#### Next, we perform the computation:

Suppose the input is shaped `[batch, input_dim]`

Each device gets the same input (i.e. input is replicated), and performs matrix multiplication with its own shard of `W`:
- Device 1 computes: `partial_1 = input @ W1 → shape [batch, 3]`
- Device 2 computes: `partial_2 = input @ W2 → shape [batch, 3]`

These are partial outputs - each covering a subset of the output features.

What do we do next?

The final step is to reconstruct the full output by joining the partial results from each device.

Do this by concatenating along the last dimension:

`final_output = torch.cat([partial1, partial2], dim=-1)`

This gives you the complete [batch, output_dim] output.


#### 2. Row Sharding

In row sharding, we split the weight matrix `W` along the **input** dimension - i.e., the **rows**.

📌 That means each device holds a subset of the input features that the full matrix would have processed.

TODO: add exclalidraw pic here

Assume the full weight matrix is `W = [4,6]`

If we have 2 devices, then:
- Device 1 gets W1 = [2, 6]
- Device 2 gets W2 = [2, 6]
<br>
→ *(ie., the first dimension is split evenly across devices)*

*Follow the same subsequent process as column sharding...*


### When to Use Column vs. Row Sharding?


| Sharding Type  | How the Weight Matrix is Split     | How the Input is Handled       | How the Output is Combined     | When to Use                                                                 |
|----------------|------------------------------------|---------------------------------|--------------------------------|------------------------------------------------------------------------------|
| **Column**     | Split along **output** dimension   | Input is **replicated**         | **Concatenate** along last dim | When output can be built from feature slices (e.g., MLPs, attention outputs) |
| **Row**        | Split along **input** dimension    | Input is **sharded/split**      | **Reduce** (e.g., sum)         | When each shard contributes to the full output (e.g., QKV projections)      |



TODO: move this to transformer folder / better place

## Self-Attention, parallelized


Self-attention is used to update each token’s embedding to include **ITS CONTEXT**.
Each output vector is a **contextualized representation of a token**, meaning it **encodes the original token plus relevant information from all other tokens**, so the model can understand the meaning of each token in context, not in isolation.

Self-attention is a sequence-to-sequence operation: a sequence of vectors goes in, and a sequence of vectors comes out.



### High level steps of self-attention:

1. **Each token is projected into 3 vectors (3 distinct pieces of information about it)**
    - **Query (Q)**: what the current token is looking for in other tokens
    - **Key (K)**:  how each token describes itself *(acting as a label that tells other tokens what kind of information it has to offer if they decide to pay attention to it)*
    - **Value (V)**: the actual information a token provides if it's chosen

These are computed as linear projections:
```
Q = x @ Weights_Q
K = x @ Weights_K
V = x @ Weights_V
```

TODO: add exclaidraw pic (each vector & matrix with embedding dims)

2. **Compute attention scores (relevance between tokens)**

To determine which *other* tokens are most relvant to the *current* token, calculate the similarity between the current token's `Q` (what its looking for) and every other token's `K` (what they can offer). 

So for each token, we compare its `Q` to every other token’s `K` using a dot product:
`score[i][j] = dot(Q_i, K_j)`

This gives us a matrix of attention scores - raw relevance scores between every token pair.

Then, we apply softmax to each row (i.e., for each token’s scores over the others):

TODO: add exclalidarw pic

`attention_weights = softmax(score)`


>This turns the raw scores into a probability distribution - i.e., a set of weights that sum to 1, indicating how much each token should pay attention to others (like a percentage). This is handy for the weighted average in the next step.

> Recall: A weighted average gives different importance to each value.
For example, given values [2, 5, 9] and weights [0.1, 0.3, 0.6], the weighted average is: 
`(0.1 x 2) + (0.3 x 5) + (0.6 x 9) = 7.1`

TODO: add exclaidraw pic

3. **Apply attention weights to Value vectors**

Now, each token will actually get what it wants from the other tokens.
This is done by taking the **weighted sum** of the attention weights and the `V` vectors.

I.e. for token `i`, compute the weighted sum of all the Value vectors `V_j` and the attention matrix `W_qk` *(derived from step 2: Q · K and softmax)*:

`output_i = (W_qk[i][0] * V_0) + (W_qk[i][1] * V_1) + (W_qk[i][2] * V_2) + ... + (W_qk[i][n] * V_n)`

Or more compactly:

    output_i = sum over j of ( W_qk[i][j] * V_j )

Where:
- `W_qk[i][j]` is the attention weight from token `i` to token `j`
- `V_j` is the Value vector of token `j`
- `output_i` is the final context-aware embedding for token `i`

TODO: add exclaidraw pic

4. **Output**

The output is a sequence of vectors, where each vector represents the original tokens **WITH ITS CONTEXT** (i.e. each vector is a contextualized representation of a token, meaning it **encodes the original token plus relevant information from all other tokens**.)

The shape of the output is: `[sequence_length, hidden_dim]`
where:
- `sequence_length`: the number of tokens in the input / length of input sequence
- `hidden_dim`: the dimension of the Value vectors (typically same as input embedding dim)

#### In summary: 
Self-attention takes each token, lets it query all other tokens’ keys, figures out how relevant they are, then gathers the most relevant values to build a new, smarter, context-aware embedding of the token.


- Intuitive example
- Technical example

