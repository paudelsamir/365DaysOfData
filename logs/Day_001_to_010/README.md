# [2024/12/14 - 2024/12/24]


Before starting this challenge, I already had [**intermediate proficiency**](https://github.com/paudelsamir/python-mastery) in **Python** and had worked on several basic to advanced [**EDA projects**](https://github.com/paudelsamir/EDA-Projects). I’m also familiar with **Statistics** and **Mathematics**, so now my goal is to revisit these topics with a **practical, data-driven focus**.

# Day 01 : Setting Up + Basics of Linear Algebra
![importance of linear algebra](img/importance_of_linear_algebra.png)
<br>


### **1. Scalars, Vectors, Matrices, and Tensors**
- Basic data structures; feature vectors as inputs, datasets as matrices, and images as 3D tensors.
![](./img/Linear%20Algebra%20and%20Calculus/example_of_tensor.png)
### **2. Linear Combination and Span**
-  Represents data points as weighted sums, used in models like **Linear Regression** and neural networks.
![](./img/Linear%20Algebra%20and%20Calculus/3dlinear_transformation.png)

### **3. Determinant**
- Checks matrix invertibility, ensuring unique solutions in systems like **linear regression**.

### **4. Dot and Cross Product**
- **Dot product** measures similarity (e.g., in **SVMs**), while **cross product** is used for vector transformations (less common). <br>
![](./img/Linear%20Algebra%20and%20Calculus/dot%20product.png)

# Day 02 : Decomposition, Derivation Integration and Gradient Descents

### **5. Identity and Inverse Matrices**
- Used for solving equations (e.g., **linear regression**) and optimization (e.g., in **gradient descent**).

### **6. Eigenvalues and Eigenvectors**
- Essential for **PCA**, **SVD**, and feature extraction; eigenvalues capture variance, eigenvectors define directions.

### **7. Singular Value Decomposition (SVD)**
- Used for **PCA**, image compression, and **collaborative filtering** (e.g., recommendation systems).

![image](./img/eigenvalue_eigenvector.png)
[Notes Here](./data/Linear%20Algebra%20for%20ML.pdf)

## Basic Understanding of Calculus
### 1. **Functions and Graphs**

- A function represents the relationship between input and output.
- In **house price prediction**, a function maps **house size** (input) to **price** (output). Continuity ensures no sudden jumps in prices.


### 2. **Derivatives**

- Derivatives measure how fast a function is changing at a given point.
- In **linear regression**, derivatives of the **cost function** tell you how to adjust model parameters (like weights) to minimize error, i.e., to predict house prices more accurately.
- **Derivatives** help adjust parameters.

![](./img/Linear%20Algebra%20and%20Calculus/area_of_circle.png)

### 3. **Partial Derivatives**

- Partial derivatives measure the change of a function with respect to one variable while keeping others constant.
- In **neural networks**, partial derivatives help adjust each weight individually during training to minimize the **loss function**.

### 4. **Gradient Descent**

- An optimization method that uses derivatives to iteratively update parameters (weights) to minimize the cost function. like backpropagation
- In **linear regression**, gradient descent is used to find the optimal weights that minimize the difference between predicted and actual house prices.

### 5. **Optimization**

- Finding the best (minimum or maximum) values of a function to improve performance.
- In **machine learning models**, you optimize the **cost function** (error) to improve predictions, such as minimizing the error in house price predictions.


### 6. **Integrals**

- Integrals calculate the total accumulation or area under a curve.
- In **probabilistic models** like Naive Bayes, integration is used to compute the total probability of an event (e.g., the probability that a house price falls within a given range).
![](./img/Linear%20Algebra%20and%20Calculus/integration.png)

*recently revised statistics and probability concepts so i'll move into the course: ML specialization next*
# Day 03 :