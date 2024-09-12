Certainly. Let's break down the matrix multiplication method used in gradient descent, focusing on the shapes of the matrices involved. We'll use our simple linear regression example to illustrate this.

Given:
- X: input matrix (features)
- y: target vector
- θ (theta): parameter vector

1. X (input matrix)
   Shape: (m, n) where m is the number of samples, n is the number of features (including bias)
   
   For our example with 5 samples and 1 feature (plus bias):
   ```
   X = [
       [1, 1],
       [1, 2],
       [1, 3],
       [1, 4],
       [1, 5]
   ]
   ```
   Shape: (5, 2)

2. y (target vector)
   Shape: (m,) or (m, 1)
   
   For our example:
   ```
   y = [2, 4, 6, 8, 10]
   ```
   Shape: (5,) or (5, 1)

3. θ (parameter vector)
   Shape: (n,) or (n, 1)
   
   Initially:
   ```
   θ = [0, 0]
   ```
   Shape: (2,) or (2, 1)

Now, let's go through the matrix multiplications:

1. Hypothesis calculation: h = X.dot(θ)
   - X shape: (5, 2)
   - θ shape: (2,) or (2, 1)
   - Resulting h shape: (5,) or (5, 1)
   
   Calculation:
   ```
   [1 1] [θ0]   [1*θ0 + 1*θ1]
   [1 2] [θ1] = [1*θ0 + 2*θ1]
   [1 3]        [1*θ0 + 3*θ1]
   [1 4]        [1*θ0 + 4*θ1]
   [1 5]        [1*θ0 + 5*θ1]
   ```

2. Error calculation: (h - y)
   - h shape: (5,) or (5, 1)
   - y shape: (5,) or (5, 1)
   - Resulting shape: (5,) or (5, 1)

3. Gradient calculation: gradient = (1/m) * X.T.dot(h - y)
   - X.T shape: (2, 5) (transpose of X)
   - (h - y) shape: (5,) or (5, 1)
   - Resulting gradient shape: (2,) or (2, 1)
   
   Calculation:
   ```
   1/m * [1 1 1 1 1] [e1]   [1*e1 + 1*e2 + 1*e3 + 1*e4 + 1*e5]
         [1 2 3 4 5] [e2] = [1*e1 + 2*e2 + 3*e3 + 4*e4 + 5*e5]
                     [e3]
                     [e4]
                     [e5]
   ```
   Where e1, e2, etc., are the elements of (h - y).

4. Parameter update: θ -= learning_rate * gradient
   - θ shape: (2,) or (2, 1)
   - gradient shape: (2,) or (2, 1)
   - Resulting θ shape: (2,) or (2, 1)

   This is element-wise subtraction:
   ```
   [θ0]   [g0]
   [θ1] - [g1]
   ```
   Where g0 and g1 are the elements of the gradient multiplied by the learning rate.

Key points:
- The dot product X.dot(θ) multiplies each row of X with θ to get a prediction for each sample.
- X.T.dot(h - y) multiplies each column of X (now rows of X.T) with the error vector, summing the errors weighted by each feature.
- The shapes must be compatible for matrix multiplication: (m, n) * (n, p) results in (m, p).
- Broadcasting in numpy allows operations between (m,) and (m, 1) shaped arrays.

This matrix approach efficiently computes the hypothesis, error, and gradient for all samples simultaneously, making the algorithm much faster than calculating each sample individually.