
![Last Updated](https://img.shields.io/github/last-commit/paudelsamir/365DaysOfData)
![Repo Size](https://img.shields.io/github/repo-size/paudelsamir/365DaysOfData) 

![cover](./resources/images/cover.png)

> Before starting this challenge, I already had [**intermediate proficiency**](https://github.com/paudelsamir/python-mastery) in **Python** and had worked on several basic to advanced [**EDA projects**](https://github.com/paudelsamir/EDA-Projects). I’m also familiar with Statistics and Mathematics.

<br>

Follow my journey with daily posts on [linkedin](https://www.linkedin.com/in/paudelsamir/)


| Books & Resources  | Completion Status |
|--------------------|-------------------|
| [Machine Learning Specialization @Coursera](https://www.coursera.org/specializations/machine-learning-introduction) | 🏊 |
| [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://github.com/yanshengjia/ml-road/blob/master/resources/Hands%20On%20Machine%20Learning%20with%20Scikit%20Learn%20and%20TensorFlow.pdf)| 🏊 | 

<br>



# Progress
| Days | Date               | Topics                      | Resources    |
|------|--------------------|-----------------------------|--------------|
| [Day1](#day-01-basics-of-linear-algebra) |2024ᚑ12ᚑ14 | Setting Up + Basics of Linear Algebra  | [3blue1brown](https://www.3blue1brown.com/topics/linear-algebra) |
| [Day2](#day-02-decomposition-derivation-integration-and-gradient-descent) | 2024-12-15 | Decomposition, Derivation, Integration, and Gradient Descent | [3blue1brown](https://www.3blue1brown.com/topics/calculus) |
| [Day3](#day-03-supervised-machine-learning-regression-and-classificaiton) | 2024-12-16 |Supervised Learning, Regression and classification|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day4](#day-04-unsupervised-learning-clustering-dimensionality-reduction) | 2024-12-17 |Unsupervised Learning: Clustering and dimensionality reduction|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day5](#day-05-univariate-linear-regression) | 2024-12-18 |Univariate linear Regression|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day6](#day-06-cost-function) | 2024-12-19 |Cost Functions|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day7](#day-07-gradient-descent) | 2024-12-20 |Gradient Descent|[CampusX](https://www.youtube.com/watch?v=ORyfPJypKuU), [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day8](#day-08-effect-of-learning-rate-cost-function-and-data-on-gd) | 2024-12-21 |Effect of learning Rate, Cost function and Data on GD|[CampusX](https://www.youtube.com/watch?v=ORyfPJypKuU), [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day9](#day-09-linear-regression-with-multiple-features-vectorization) | 2024-12-22 |Linear Regression with multiple features, Vectorization| [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day10](#day10-feature-scaling) | 2024-12-23 |Feature Scaling, Visualization of Multiple Regression and Polynomial Regression| [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day11](#day-11-feature-engineering-and-polynomial-regression) | 2024-12-24 |Feature Engineering, Polynomial Regression| [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day12](#day-12-linear-regression-using-scikit-learn) | 2024-12-25 |Linear Regression using Scikit Learn from Scratch| [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) |
| [Day13](#day-13-classification) | 2024-12-26 |-----------------|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)|
| [Day14]() | 2024-12-27 |-----------------|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)|
| [Day15]() | 2024-12-28 |-----------------|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)|
| [Day16]() | 2024-12-29 |-----------------|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)|
| [Day17]() | 2024-12-30 |-----------------|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)|
| [Day18]() | 2024-12-31 |-----------------|[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)|

<br>
<br>
<br>














# Day 01: Basics of Linear Algebra 
<!-- <img src="01-Supervised-Learning/images/importance_of_linear_algebra.png" width="400" /> -->

<!-- 
![Importance of Linear Algebra](./01-Supervised-Learning/images/importance_of_linear_algebra.png) -->

- Scalars, Vectors, Matrices, Tensors: Basic data structures for ML.
    ![](./01-Supervised-Learning/images/example_of_tensor.png)

- Linear Combination and Span: Representing data points as weighted sums. Used in Linear Regression and neural networks.
    ![](./01-Supervised-Learning/images/3dlinear_transformation.png)

- Determinants: Matrix invertibility, unique solutions in linear regression.

- Dot and Cross Product: Similarity (e.g., in SVMs) and vector transformations.
    ![](./01-Supervised-Learning/images/dot_product.png)

*Slow progress right?? but consistent wins the race!*
---

# Day 02: Decomposition, Derivation, Integration, and Gradient Descent

- Identity and Inverse Matrices: Solving equations (e.g., linear regression) and optimization (e.g., gradient descent).

- Eigenvalues and Eigenvectors: PCA, SVD, feature extraction; eigenvalues capture variance.
    ![](./01-Supervised-Learning/images/eigenvalue_eigenvector.png)

- Singular Value Decomposition (SVD): PCA, image compression, and collaborative filtering.

[Notes Here]()

### Calculus Overview:
- Functions & Graphs: Relationship between input (e.g., house size) and output (e.g., house price).

- Derivatives: Adjust model parameters to minimize error in predictions (e.g., house price).
    ![](./01-Supervised-Learning/images/area_of_circle.png)

- Partial Derivatives: Measure change with respect to one variable, used in neural networks for weight updates.

- Gradient Descent: Optimization to minimize the cost function (error).

- Optimization: Finding the best values (minima/maxima) of a function to improve predictions.

- Integrals: Calculate area under a curve, used in probabilistic models (e.g., Naive Bayes).
    ![](./01-Supervised-Learning/images/integration.png)

Revised statistics and probability concepts. Ready for the ML Specialization course!
---

# Day 03: Supervised Machine Learning: Regression and Classificaiton
<!-- [Notes credit](https://drive.google.com/file/d/1SO3WJZGSPx2jypBUugJkkwO8LZozBK7B/view?usp=sharing) -->
- Supervised Learning: <br>
![](./01-Supervised-Learning/images/Supervised%20Learning.png)
- Regression:<br>
![](./01-Supervised-Learning/images/Regression_model.png)
- Classification:<br>
![](./01-Supervised-Learning/images/classification_model.png)
![](./01-Supervised-Learning/images/classification_model2.png)
---

# Day 04: Unsupervised Learning: Clustering, dimensionality reduction

data only comes with input x, but not output labels y. Algorithm has to find structure in data.

- Clustering: group similar data points together <br>
![alt text](./01-Supervised-Learning/images/clustering.png)
- dimensionality reduction: compress data using fewer numbers eg image compression<br> <img src="./01-Supervised-Learning/images/dimensionality_reduction.png" width = "300">
<!-- ![alt text](./01-Supervised-Learning/images/dimensionality_reduction.png) -->
- anomaly detection: find unusual data points eg fraud detection<br>
---

# Day 05: Univariate Linear Regression:
- Learned univariate linear regression and practiced building a model to predict house prices using size as input, including defining the hypothesis function, making predictions, and visualizing results.

[Notebook: Model Representation](./01-Supervised-Learning/code/day04_model_representation.ipynb)

<!-- ![](./01-Supervised-Learning/images/notes_univariate_linear_regression.jpg) -->

<img src="./01-Supervised-Learning/images/notes_univariate_linear_regression.jpg" width = "400">

![](./01-Supervised-Learning/images/notations_summary.png) - Univariate Linear Regression Quiz

![](./01-Supervised-Learning/images/univariate_linear_regression_quiz.png)
---

# Day 06: Cost Function:
![alt text](./01-Supervised-Learning/images/costfunction.jpg)
Visualization of cost function:
![Visualization of cost function](./01-Supervised-Learning/images/visualization_costfunction.png)

- manually reading these contour plot is not effective or correct, as the complexity increases, we need an algorithm which figures out the values w, b (parameters) to get the best fit time, minimizing cost function

[Notebook: Model Representation](./01-Supervised-Learning/code/day04_model_representation.ipynb)

*Gradient descent is an algorithm which does this task*
---

# Day 07: Gradient Descent
[Notebook: Gradient descent](./01-Supervised-Learning/code/day07_gradient-descent-code-from-scratch.ipynb)

learned the basics by assuming slope constant and with only the vertical shift.
later learned GD with both the parameters w and b.
![alt text](./01-Supervised-Learning/images/gradientdescent.png)
<!-- 
![alt text](./01-Supervised-Learning/images/implementation_of_gradient_descent.png) -->
 ![alt text](01-Supervised-Learning/images/gdnote.jpg) 
---

# Day 08: Effect of learning Rate, Cost function and Data on GD
- learning rate on GD:Affects the step size; too high can overshoot, too low can slow convergence
<img src="./01-Supervised-Learning/images/learningrate_eg1.png" width = "400">
<!-- ![alt text](./01-Supervised-Learning/images/learningrate_eg1.png) -->
<img src="./01-Supervised-Learning/images/learningrate_eg2.png" width = "400">
<!-- ![alt text](./01-Supervised-Learning/images/learningrate_eg2.png) -->
- cost function on GD:Smooth, convex functions help faster convergence; complex ones may trap in local minima

- Data on GD:Quality and scaling affect stability; more data improves gradient estimates

[Notebook: gradient descent animation 3d](./01-Supervised-Learning/code/day08_gradient-descent-animation(both-m-and-b).ipynb)
---

# Day 09: Linear Regression with multiple features, Vectorization

Predicts target using multiple features, minimizing error.
- Vectorization: Matrix operations replace loops for faster calculations.

![alt text\](image.png](01-Supervised-Learning/images/multifeatureLR.png)

[Lab1: Vectorization](./01-Supervised-Learning/code/day09_Python_Numpy_Vectorization_Soln.ipynb)
<br>
---

# Day10: Feature Scaling
[Lab2: Multiple Variable](./01-Supervised-Learning/code/day09_Lab02_Multiple_Variable_Soln.ipynb)

Today, I learned about feature scaling and how it helps improve predictions. There are multiple methods for feature scaling, including
- Min-Max Scaling
- Mean Normalization
- Z-Score Normalization
![alt text](./01-Supervised-Learning/images/notesfeature_scaling.png
)
To ensure proper convergence:
check the learning curve to confirm the loss is decreasing.
Start with a small learning rate and gradually increase to find the optimal value.

![alt text](./01-Supervised-Learning/images/gdforconvergence.png) 
![alt text](01-Supervised-Learning/images/choosinglearningrate.png)
---

# Day 11: Feature engineering and Polynomial Regression
feature engineering improves features to better predict the target.

eg If we need to predict the cost of flooring and have length and breadth of the room as features, we can use feature engineering to create a new feature, area (length × breadth), which directly impacts the flooring cost.

<img src="./01-Supervised-Learning/images/notes_featureengineering.jpg" width = "400">
![alt text](01-Supervised-Learning/images/notes_featureengineering.jpg)

explored polynomial regression that models the relationship between variables as a polynomial curve instead of a straight line

**Equation:**  
y = b₀ + b₁x + b₂x² + ... + bₙxⁿ 
It is useful for capturing nonlinear relationships in data.
![alt text](01-Supervised-Learning/images/visualizationof_polynomialfunctions.png)

![alt text](01-Supervised-Learning/images/scalingfeatures_complexfunctions.png)


[Lab1: Feature Scaling and Learning Rate](01-Supervised-Learning/code/day11_Feature_Scaling_and_Learning_Rate_Soln.ipynb) <br>
[Lab2: Feature Engineering and PolyRegression](01-Supervised-Learning/code/day11_FeatEng_PolyReg_Soln.ipynb)
---

# Day 12: Linear Regression using Scikit Learn
Had a productive session with linear regression in scikit learn. The lab helped me get a better grasp of the process, though I need more practice with tuning models. Also revisited the Scikit-Learn models ,more comfortable with them now

![](https://github.com/paudelsamir/365DaysOfData/blob/main/01-Supervised-Learning/images/LRusingscikitlearn.png)
![alt text](01-Supervised-Learning/images/LRusingscikitlearn.png)
![alt text](01-Supervised-Learning/images/scikitlearn_cleansheet1.png) 
![alt text](01-Supervised-Learning/images/scikitlearn_cleansheet2.png) <br>
[Notebook: ScikitLearn GD](01-Supervised-Learning/code/day12_Sklearn_GD_Soln.ipynb)
---

# Day 13: Classification

