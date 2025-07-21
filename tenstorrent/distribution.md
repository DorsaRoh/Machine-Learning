#### Let `latents` be a tensor with dimensions:
- **B: batch size**
    - The number of training data samples (ex. images) that were processed in one forward/backward pass to create this tensor
- **H: height**
    - the height of the latent tensor
- **W width**: 
    - the width of the latent tensor
- **C: channels**
    - the number of different "feature filters" the network learned

    ex. <br>
    Channel 1: Detects horizontal edges
    <br>
    Channel 2: Detects round shapes  
    Channel 3: Detects fur texture
    <br>
    ...
    <br>
    Channel 512: Detects cat ears

#### `latents: [B, H, W, C]`


#### And assume we are using a Tenstorrent T3K machine. 

This means our mesh device is a grid of shape **[1, 8]**, or **[2, 4]**.

In other words, we have 8 total chips in our cluster, organized in either:
- 1 row with 8 chips (*[1,8]*)
- 2 rows with 4 chips each (*[2, 4]*)

A **submesh device** is defined as a **portion of the mesh device**. 

For example, in this case with a [2,4] T3K, we can have a submesh device of [2,2]. This means the submesh is 2 rows of 2 chips each. So in total, there are two submeshes of [2,2] each.

Let `submesh_device = [x, y] = [2, 2]`

---

**Sharding** means to spread/distribute data across multiple machines.


## Example: Sharding

Say we want our latents sharded like so:
```
sharded_lantents = [B, H/x, W, C]
# sharded on the height on cluster axis x
```

Thus, we can use
```
from_torch(mesh_mapper=ShardTensor2DMesh(
    dims=(1, None)) # fracture dim 1 (H) on cluster axis x, don't fracture on cluster axis y
)
```
**dims: mapping from cluster axis to tensor dims**

So every device has a shard of this shape
`sharded_lantents = [B, H/x, W, C] # sharded on the height on cluster x`

If you all-gather across the x-axis, you'd get a fully replicated tensor of shape [B, H, W, C]

If we have tensor A: [B, S, D] -> shard -> [B, S/y, D/x]
```
mesh_mapper=ShardTensor2dMesh(
    dims=(2,1)  # x is on dim 2 (D), y is on dim 1 (S)
)
```


## Example: bring back to device / concatenate

So now we have `some_tensor: [B, S/y, D/x]`.<br>


**To bring it back to host as [B, S, D]: we use a mesh composer**

```
mesh_composer = ConcatMesh2DToTensor(
    dims=[2,1]
)
```
We need to specify how to concatenate back. We need to concatenate on two axes.


For example, to get [B, H/x, W, C] back to device as [B, H, W, C]:
```
mesh_composer = ConcatMesh2DToTensor(
    dims=[1, 0] 
    # we are replicated on y, not fracturing on y. 0 is just some other dimension. 
)
```
*Can't put None in the y position because it is malformed in the API. have to give it something (in this case, 0)*

We are not done. This gives us **[B * y, H, W, C]**


**We need to index in B to get B**
<br>
i.e. `[:B, ...] # [B, H, W, C]`



