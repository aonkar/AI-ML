# Course 1: Supervised Machine Learning: Regression and Classification

## Mental Map (Andrew Ng / DeepLearning.AI — ML Specialization)

---

## 🗓️ Week 1: Introduction to Machine Learning & Linear Regression with One Variable

Week 1 transitions from general machine learning definitions to building your very first predictive model: Linear Regression with a single feature (univariate).

### 1. Fundamental Definitions

- **Supervised Learning:** The algorithm learns from a dataset containing both the inputs ($X$) and the correct output labels ($Y$). It maps inputs directly to outputs.
- **Unsupervised Learning:** The algorithm finds hidden structures, groupings, or patterns in unlabeled data ($X$ only).
- **Regression vs. Classification:**
  - *Regression:* Predicts a continuous numeric value (e.g., predicting a house price of \$350,200).
  - *Classification:* Predicts a discrete category or class (e.g., classifying an email as "Spam" `[1]` or "Not Spam" `[0]`).

### 2. The Model Structure

- **The Dataset:** Composed of $m$ training examples. A single example is represented as a tuple: $(x^{(i)}, y^{(i)})$.
- **Hypothesis / Model Function:**

$$f_{w,b}(x) = wx + b$$

- $x$: Input feature (e.g., size of house).
- $w$: Weight (slope of the line).
- $b$: Bias (y-intercept).

### 3. The Objective: Cost Function

To make accurate predictions, the model needs to find parameters $w$ and $b$ that make $f_{w,b}(x)$ as close to the true value $y$ as possible. This is evaluated using the **Mean Squared Error (MSE)** cost function:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$

> 💡 **Mental Picture:** If you plot $J(w,b)$ against different values of $w$ and $b$, it forms a 3D bowl shape (a convex surface). The bottom of the bowl represents the absolute lowest error point — the global minimum.

### 4. The Mechanism: Gradient Descent

Gradient Descent is the optimization algorithm used to minimize the cost function $J(w,b)$ by iteratively adjusting the weights.

**The Algorithm** (repeat until convergence):

$$w := w - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}$$

$$b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)$$

- **Simultaneous Update:** $w$ and $b$ must be computed using their *old* values before either is updated.
- **The Learning Rate ($\alpha$):**
  - Too **small** → tiny steps, highly inefficient convergence.
  - Too **large** → overshoots the minimum, may diverge entirely.

---

## 🗓️ Week 2: Linear Regression with Multiple Input Variables

Week 2 expands the model to handle situations where real-world predictions depend on multiple distinct features instead of just one.

### 1. Multiple Features (Multivariate)

Instead of just using "size of house," we add variables like "number of bedrooms", "number of floors", and "age of home".

- **Notation:**
  - $n$: Number of features.
  - $\mathbf{x}^{(i)}$: A row vector containing all features for the $i$-th example.
  - $x_j^{(i)}$: The value of feature $j$ in the $i$-th training example.
- **New Hypothesis Function:**

$$f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

### 2. Optimization via Vectorization

Instead of writing slow `for` loops to calculate $\sum w_j x_j$, we utilize linear algebra and libraries like NumPy.

```python
f = np.dot(w, x) + b
```

**Why it matters:** Vectorization allows modern hardware (CPUs/GPUs) to parallelize computations, speeding up training by orders of magnitude.

### 3. Practical Tool: Feature Scaling

When features have vastly different numeric ranges (e.g., $x_1 = \text{bedrooms} \in [1, 5]$, $x_2 = \text{house size} \in [500, 5000]$ sq ft), the cost function contours become heavily elongated ellipses. This causes gradient descent to oscillate inefficiently.

- **Mean Normalization** — adjusts features to have zero mean:

$$x_i := \frac{x_i - \mu_i}{\max - \min}$$

- **Z-score Standardization** — scales features by standard deviation:

$$x_i := \frac{x_i - \mu_i}{\sigma_i}$$

Target range: roughly −1 ≤ x_i ≤ 1

### 4. Polynomial Regression & Feature Engineering

Linear models cannot naturally fit curves. By transforming existing features — squaring them ($x^2$), cubing them ($x^3$), or combining them ($\text{width} \times \text{length}$) — you can use the exact same linear regression mechanics to fit highly complex non-linear curves.

---

## 🗓️ Week 3: Classification via Logistic Regression

Week 3 shifts from predicting continuous numbers to predicting categorical values using the classic baseline classification algorithm.

### 1. Why Linear Regression Fails for Classification

- Adding a single far-outlier data point can radically tilt a linear regression line, shifting the decision threshold and breaking predictions.
- Linear lines output values greater than $1$ or less than $0$, which is statistically meaningless for probabilities.

### 2. The Logistic Regression Model

To fix this, we pass our linear expression through the **Sigmoid (Logistic) Function**.

- **Sigmoid Function:**

$$g(z) = \frac{1}{1 + e^{-z}}$$

Outputs a value strictly between 0 and 1.

- **Model Hypothesis:**

$$f_{\mathbf{w},b}(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$

- **Interpretation:** The output is the probability that $y = 1$ given input $\mathbf{x}$. For example, $f_{\mathbf{w},b}(\mathbf{x}) = 0.7$ means a 70% chance the example belongs to the positive class.

### 3. Decision Boundary

- If $f_{\mathbf{w},b}(\mathbf{x}) \ge 0.5$, we predict $\hat{y} = 1$.
- This threshold occurs exactly when $z = \mathbf{w} \cdot \mathbf{x} + b \ge 0$.
- The line or curve where $\mathbf{w} \cdot \mathbf{x} + b = 0$ is the **Decision Boundary** — it completely separates predicted classes in feature space.

### 4. The Logistic Loss & Cost Function

We cannot use MSE for Logistic Regression because the sigmoid makes the cost curve non-convex with many local minima. Instead we use **Binary Cross-Entropy Loss**.

- **Loss on a Single Example:**

$$L\bigl(f_{\mathbf{w},b}(\mathbf{x}^{(i)}),\, y^{(i)}\bigr) = -y^{(i)} \log\!\left(f_{\mathbf{w},b}(\mathbf{x}^{(i)})\right) - \left(1 - y^{(i)}\right) \log\!\left(1 - f_{\mathbf{w},b}(\mathbf{x}^{(i)})\right)$$

- **Total Cost over Training Set:**

$$J(\mathbf{w},b) = \frac{1}{m} \sum_{i=1}^{m} L\!\left(f_{\mathbf{w},b}(\mathbf{x}^{(i)}),\, y^{(i)}\right)$$

### 5. Overfitting vs. Underfitting (The Generalization Problem)

| | Description | Symptom |
| --- | --- | --- |
| **Underfitting (High Bias)** | Model is too simple / rigid | Fails to capture patterns even on training data |
| **Overfitting (High Variance)** | Model is too complex (too many polynomial terms) | Fits training noise perfectly; fails on unseen data |

### 6. The Solution: Regularization

Instead of manually deleting features, we use **L2 Regularization** — a penalty term added to the cost function that discourages weights $w_j$ from growing to extreme values.

- **Regularized Cost Function:**

$$J(\mathbf{w},b) = \underbrace{\left[\text{Original Cost}\right]}_{\text{Linear or Logistic}} + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

- **$\lambda$ (Lambda) — the regularization strength:**
  - Too **small** → model continues to overfit.
  - Too **large** → weights are penalized toward zero, causing underfitting.
- *Note:* The bias term $b$ is traditionally left unregularized.
