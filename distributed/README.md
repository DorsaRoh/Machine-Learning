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

