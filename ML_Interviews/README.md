# ML Interview Questions & Answers

*Questions are from [Introduction to ML Interviews Book](https://huyenchip.com/ml-interviews-book/) by Chip Huyen.*

## 1.1 Vectors

### 1. **Dot Product**

**i. [E] What’s the geometric interpretation of the dot product of two vectors?**

- The dot product represents the magnitude of one vector projected onto another, capturing the "shadow"/component of one vector in the direction of the other
- Mathematically, if `a` is the angle between vectors `A` and `B`, the dot product is given by:
  `A · B = |A| |B| cos(a)`
  where `|A|` and `|B|` are the magnitudes of vectors `A` and `B`, respectively

**ii. [E] Given a vector `u`, find a vector `v` of unit length such that the dot product of `u` and `v` is maximized.**

- The dot product is defined as `u · v = |u| |v| cos(θ)`, where `θ` is the angle between the vectors
- To maximize the dot product, `cos(θ)` should be maximized, which occurs when `θ = 0`. At this angle, `cos(0) = 1`
- Therefore, the maximum dot product is achieved when `v` is in the same direction as `u`, and its magnitude is 1
    - Isolating for `v`,  `v` is given by: `v = u / |u|` where `|u|` is the magnitude of vector `u`


### 2. **Outer Product**

**i. [E] Given two vectors `a = [3, 2, 1]` and `b = [-1, 0, 1]`, calculate the outer product `aTb`.**

**ii. [M] Give an example of how the outer product can be useful in ML.**

**iii. [E] What does it mean for two vectors to be linearly independent?**

**iv. [M] Given two sets of vectors `A = {a1, a2, a3, ..., an}` and `B = {b1, b2, b3, ..., bm}`, how do you check that they share the same basis?**

**v. [M] Given `n` vectors, each of `d` dimensions, what is the dimension of their span?**

### 3. **Norms and Metrics**

**i. [E] What's a norm? What is `L0, L1, L2, Lnorm`?**

**ii. [M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?**
