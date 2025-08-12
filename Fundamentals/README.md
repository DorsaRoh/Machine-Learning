# 1. Matrix Multiplication

### Concept
Matrix multiplication is like **applying transformations in sequence**.  
Each multiplication changes the data step-by-step.



### 2. Important Rules

| Rule | Explanation | Example |
|------|-------------|---------|
| **Order matters** | `(M1)(M2) ≠ (M2)(M1)` in general. | Rotating then scaling ≠ scaling then rotating |
| **Right-to-left in notation** | In `ABC`, you first apply **C**, then **B**, then **A**. | If C = translate, B = rotate, A = scale → you scale last |
| **Shape rule for 2D** | If `A.shape = [m, n]` and `B.shape = [n, p]`, then `(A × B).shape = [m, p]`. The `n` cancels out. | `[3, 4] × [4, 2] → [3, 2]` |
| **Shape rule for higher dimensions** | For batched/multidim tensors, only the **last two dimensions** follow the multiplication rule. Earlier dimensions must match for broadcasting. | `[a, b, m, n] × [a, b, n, p] → [a, b, m, p]` |

