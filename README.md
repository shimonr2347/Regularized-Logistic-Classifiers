# LogReg-from-Scratch: Linear & Regularized Classification

A from-scratch implementation of Logistic Regression and Regularized Logistic Regression using NumPy. This project demonstrates how to build classification models to solve both linear and non-linear decision problems.

## 🚀 Overview
This repository contains a comprehensive study of Logistic Regression applied to two distinct challenges:
### University Admission Prediction: 
A linear classification problem based on exam scores.

### Microchip Quality Assurance: 
A non-linear classification problem requiring polynomial feature mapping and regularization.

## 🛠️ Key Features

### Custom Sigmoid Function: 
Implementation of the logistic hypothesis $h_\theta(x) = \frac{1}{1+e^{-\theta^T x}}$.

### Momentum Optimizer: 
An advanced optimization algorithm that uses a moving average of gradients to accelerate convergence and navigate complex loss surfaces.

### Polynomial Feature Mapping: 
Transformation of 2D input into a 28-dimensional space to capture non-linear patterns.

### L2 Regularization: 
Implementation of a penalty term ($\lambda$) to prevent overfitting in high-dimensional feature spaces.

## 📊 Project Modules
### 1. University Admission (Linear)
The model predicts the probability of admission based on two exam scores. Since the data is linearly separable, the decision boundary is a straight line.Optimization: Stochastic Gradient Descent with Momentum.
Scaling: Features are standardized (mean=0, std=1) to ensure stable convergence.

### 2. Microchip QA (Non-linear & Regularized)
Microchip test results show a circular/complex distribution that cannot be separated by a line. We apply 6th-degree polynomial mapping to fit the data.

### The Impact of Regularization ($\lambda$):
We analyze how the model's behavior changes with different regularization strengths:
* Underfitting ($\lambda = 100$): The model is too simple and fails to capture the trend.
* Optimal Fit ($\lambda = 1$): A smooth, circular boundary that generalizes well to new data.
* Overfitting ($\lambda = 0$): The model creates a complex, "wiggly" boundary to catch every outlier, failing to generalize.

## 🧬 Mathematical Implementation
### The core of the project is the Regularized Cost Function:
$$J(\theta) = \frac{1}{m} \sum_{i=1}^m \left[ -y^{(i)}\log \left( h_\theta \left(x^{(i)} \right) \right) - \left( 1 - y^{(i)} \right) \log \left( 1 - h_\theta \left( x^{(i)} \right) \right) \right] + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2$$

### And the Gradient with Momentum:
Compute gradient $\nabla J(\theta)$.
* Update Velocity: $v_t = \beta v_{t-1} + (1 - \beta)\nabla J(\theta)$.
* Update Weights: $\theta = \theta - \alpha v_t$.


## 📈 Performance Metrics & ROC Analysis
To evaluate the models beyond simple accuracy, we implement Receiver Operating Characteristic (ROC) curves and calculate the Area Under the Curve (AUC). This measures the model’s ability to distinguish between classes across all possible thresholds.

| Dataset | Training Accuracy | AUC Score | Model Type |
|---------|-------------------|-----------|------------|
| University Admission | 89.00% | 0.97 |Linear Logistic Regression |
| Microchip QA | 83.05% | 0.91 | Regularized Non-Linear ($L_2, \lambda=1$) |

## Interpretation:
* AUC = 0.97: An exceptional classifier with nearly perfect class separation.
* AUC = 0.91: An excellent classifier that effectively handles the noise and non-linearity of the microchip dataset.


## 📁 Project Structure
```
Regularized-Logistic-Classifiers/
├── data/
│   ├── data1.txt             # University scores dataset
│   └── data2.txt             # Microchip QA dataset
├── docs/                     # Generated plots (Boundaries & ROC)
│   ├── microchip_boundary.png
│   ├── microchip_roc.png
│   ├── university_boundary.png
│   └── university_roc.png
├── logistic_regression.py    # Main implementation script
└── README.md                 # Project documentation
```

## 💻 Setup & Usage
### Clone the repository:

```bash
git clone https://github.com/shimonr2347/Regularized-Logistic-Classifiers.git
```

### Install dependencies:

```bash
pip install numpy pandas matplotlib
```

### Run the script:

```bash    
python logistic_regression.py
```

## 📈 Results
*   **University Model**: ~89% Training Accuracy.
<img width="992" height="750" alt="university_boundary" src="https://github.com/user-attachments/assets/f733137d-08d2-4e2f-b436-6995abb0696d" />
*   **University Admission AUC**: 0.9735
<img width="867" height="622" alt="university_roc" src="https://github.com/user-attachments/assets/cb7e0405-c29d-4eac-a17f-fb2d65f8350d" />
*   **Microchip Model**: ~83% Training Accuracy (with $\lambda=1$).
<img width="992" height="745" alt="microchip_boundary" src="https://github.com/user-attachments/assets/09353f91-e6b9-447f-a220-e2d59c36a999" />
*   **Microchip QA AUC**: 0.9073
<img width="867" height="622" alt="microchip_roc" src="https://github.com/user-attachments/assets/f7d30eda-4353-454d-8219-fdec8d241ea1" />


---
