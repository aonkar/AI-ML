# Course 2: Advanced Learning Algorithms

## Mental Map (Andrew Ng / DeepLearning.AI — ML Specialization)

---

### 1. Neural Networks — Intuition & Architecture

- Inspired loosely by biological neurons; in practice, a stack of layers of computational units
- **Layer**: group of neurons/units; each unit computes `a = g(w·x + b)` on its inputs
- **Terminology**: input layer → hidden layer(s) → output layer
  - `a^[l]` = activation vector of layer `l`
  - Each layer's output feeds as input to the next
- **Forward propagation**: sequential computation from input layer to output layer
- **Why neural networks over hand-engineered polynomial features?** — they learn their own features (representation learning), scale better with complex data (images, audio, text)

### 2. Building & Training Neural Networks (TensorFlow/Keras workflow)

- Three-step recipe:
  1. **Specify the model** — `Sequential([Dense(units, activation=...), ...])`
  2. **Specify loss & compile** — e.g., `BinaryCrossentropy()`, `MeanSquaredError()`, `SparseCategoricalCrossentropy()`
  3. **Train** — `model.fit(X, y, epochs=...)` → runs gradient descent (backpropagation under the hood)
- **Forward prop from scratch**: understanding a Dense layer as `matmul(A_prev, W) + b` then activation
- **NumPy vs. TensorFlow conventions**: TF uses 2D arrays (matrices) even for single examples

### 3. Activation Functions

- **Sigmoid**: output (0,1), used historically, mostly replaced in hidden layers due to vanishing gradients
- **ReLU** (`max(0,z)`): most common for hidden layers — faster training, avoids flat-gradient regions on the positive side
- **Linear activation**: `g(z)=z`, used for regression output layer
- **Choosing activation for output layer**:
  - Binary classification → sigmoid
  - Regression (any sign) → linear
  - Regression (non-negative) → ReLU
- **Why not linear activations everywhere**: composition of linear functions collapses into a single linear function → network degenerates to linear/logistic regression, loses representational power

### 4. Multiclass Classification & Softmax

- **Softmax regression**: generalizes logistic regression to `K` classes
  `a_j = e^(z_j) / Σ e^(z_k)` for k=1..K → outputs a probability distribution over classes
- **Loss**: sparse categorical cross-entropy
- **Numerical stability**: prefer `from_logits=True` in TF (compute softmax internally in the loss rather than as a separate activation layer) to avoid rounding-error accumulation
- **Multi-label classification** (contrast with multiclass): each example can have several correct labels simultaneously — separate sigmoid output units per label

### 5. Optimization & Practical Training

- **Adam optimizer**: adapts the learning rate per-parameter automatically (faster/more robust than vanilla gradient descent) — default choice in practice
- **Convolutional layers** (brief intro): units connect only to a local region of the previous layer's inputs (not fully connected) — computationally efficient, useful for image-like/spatial data, needs less training data

### 6. Backpropagation (Conceptual)

- Computes gradients of the cost function w.r.t. every parameter via the **chain rule**, propagating error backward from output to input layer
- Computation graphs make forward and backward passes explicit
- This is what `model.fit()` executes internally every training step

### 7. Evaluating & Improving a Model — The ML Diagnostic Toolkit

- **Train/Cross-Validation(Dev)/Test split** (typical 60/20/20):
  - Train set → fit parameters
  - CV set → select model/hyperparameters (degree of polynomial, λ, architecture, etc.)
  - Test set → final, unbiased performance estimate (touch only once)
- **Bias vs. Variance diagnosis**:
  - High bias (underfit): `J_train` high, `J_cv ≈ J_train` (both high)
  - High variance (overfit): `J_train` low, `J_cv >> J_train`
  - "Just right": both low and close
- **Regularization parameter λ and bias/variance**:
  - λ too large → high bias; λ too small/zero → high variance
  - Choose λ via CV error, sweeping a range of values
- **Learning curves**: plot `J_train` and `J_cv` vs. training set size `m`
  - High bias → curves plateau close together at a high error (more data won't help much)
  - High variance → large gap between curves, gap narrows with more data (more data helps)
- **Establishing a baseline**: compare against human-level performance / competing algorithm / prior system to judge if error levels are "acceptable"

### 8. Debugging a Learning Algorithm (practical checklist)

- More training examples → fixes high variance
- Fewer features → fixes high variance
- More features / higher-degree polynomial → fixes high bias
- Decrease λ → fixes high bias
- Increase λ → fixes high variance
- Neural network specific: bigger network (more units/layers) generally reduces bias (with enough data/regularization, rarely hurts, though costs compute); more data reduces variance

### 9. Error Metrics for Skewed Datasets

- Accuracy is misleading with rare-class problems (e.g., 99% healthy patients)
- **Precision** = TP/(TP+FP) — of predicted positives, how many are correct
- **Recall** = TP/(TP+FN) — of actual positives, how many were caught
- **Precision/Recall tradeoff**: adjust classification threshold to prioritize one over the other depending on application cost of false positives vs. false negatives
- **F1 score**: harmonic mean of precision & recall — `2·P·R/(P+R)` — single number to compare models, penalizes extreme imbalance between P and R

### 10. Decision Trees

- **Structure**: root node → decision nodes (splits on a feature) → leaf nodes (predictions)
- **Splitting criteria**: choose the feature/threshold that maximizes **purity** of resulting subsets
  - **Entropy** as impurity measure: `H(p) = -p·log₂(p) - (1-p)·log₂(1-p)`
  - **Information gain**: reduction in weighted entropy after a split — pick the split that maximizes it
- **Stopping criteria**: max depth, node purity threshold, minimum examples per node, insufficient information gain
- **One-hot encoding**: used to handle categorical features with >2 values (splits historically assumed binary features)
- **Continuous features**: choose a threshold that maximizes information gain (test multiple candidate thresholds)
- **Regression trees**: variant that predicts continuous values — splits chosen to minimize variance instead of entropy

### 11. Tree Ensembles

- **Why ensembles**: single trees are highly sensitive to small data changes (high variance) → averaging over many trees stabilizes predictions
- **Random Forest**:
  - **Bagging** (bootstrap aggregation): train each tree on a random sample (with replacement) of the training set
  - **Random feature subset**: at each split, consider only a random subset of features (typically `√n`) — decorrelates the trees further
  - Prediction = majority vote (classification) / average (regression) across trees
- **Boosted trees / XGBoost**:
  - Sequential training: each new tree focuses more on examples the previous trees got wrong (weighted resampling)
  - XGBoost = optimized, regularized, fast implementation of boosted trees; default good choice in practice; has built-in train/CV split logic and regularization
- **Decision trees/ensembles vs. Neural Networks**:
  - Trees/XGBoost: excellent on **structured/tabular** data, fast to train, interpretable, don't need feature scaling
  - Neural networks: better for **unstructured** data (images, audio, text), support transfer learning, can be combined into larger learning systems, but need more compute/data and are less interpretable

---

## Cross-Cutting Themes (Course 1 → Course 2)

- Both courses build on the same skeleton: **model → cost/loss function → optimization (gradient descent)**
- Logistic regression is conceptually the "single neuron" building block of neural networks (sigmoid unit)
- Regularization (λ) appears identically in linear regression, logistic regression, and is echoed in NN training practices
- The bias/variance framework here is the systematic diagnostic lens for problems introduced informally as under/overfitting in Course 1
- Gradient descent (Course 1) generalizes to backpropagation (Course 2) — same core idea, more layers, chain rule