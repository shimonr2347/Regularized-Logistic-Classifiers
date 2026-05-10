import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def plot_linear_boundary(theta, X, y, mean, std):
    plt.figure(figsize=(8, 6))
    
    # Filter data for plotting
    pos = y == 1
    neg = y == 0
    
    # We use the raw values for the axes but the scaled values for the logic
    # To plot properly on raw scale, we reverse the scaling for the line
    plt.scatter(data1.iloc[pos, 0], data1.iloc[pos, 1], marker='o', c='b', label='Admitted')
    plt.scatter(data1.iloc[neg, 0], data1.iloc[neg, 1], marker='x', c='r', label='Not Admitted')

    # Decision Boundary: theta0 + theta1*x1 + theta2*x2 = 0
    # Solve for x2: x2 = -(theta0 + theta1*x1) / theta2
    plot_x = np.array([min(X[:, 1]), max(X[:, 1])])
    plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])
    
    # Convert scaled line back to original scale for visualization
    plot_x_raw = plot_x * std[0] + mean[0]
    plot_y_raw = plot_y * std[1] + mean[1]

    plt.plot(plot_x_raw, plot_y_raw, color='green', label='Decision Boundary')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend()
    plt.title('University Admission Decision Boundary')
    plt.show() # <--- This opens the window for the plot

def plot_nonlinear_boundary(theta, y):
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    pos = y == 1
    neg = y == 0
    plt.scatter(data2.iloc[pos, 0], data2.iloc[pos, 1], marker='o', c='b', label='Accepted')
    plt.scatter(data2.iloc[neg, 0], data2.iloc[neg, 1], marker='x', c='r', label='Rejected')

    # Create grid to evaluate the model
    u = np.linspace(-1, 1.2, 50)
    v = np.linspace(-1, 1.2, 50)
    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            # We must map the features for every single point on the grid
            point_mapped = map_feature(np.array([u[i]]), np.array([v[j]]))
            z[i, j] = (point_mapped @ theta).item() # Get scalar value from 1xN array

    # Draw the contour where z = 0 (which is where sigmoid(z) = 0.5)
    plt.contour(u, v, z.T, levels=[0], colors='green')
    
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('Microchip QA (Regularized Boundary)')
    plt.legend()
    plt.show()

def plot_roc_curve(theta, X, y, title):
    # Get probabilities from the model
    probs = sigmoid(X @ theta)
    
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []

    for t in thresholds:
        # Predict 1 if probability >= threshold
        y_pred = (probs >= t).astype(int)
        
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        tn = np.sum((y_pred == 0) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Calculate approximate AUC (Area Under Curve) using trapezoidal rule
    try:
        auc = np.trapezoid(tpr_list[::-1], fpr_list[::-1])
    except AttributeError:
        auc = np.trapz(tpr_list[::-1], fpr_list[::-1])

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    print (f"{title} AUC: {auc:.4f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def sigmoid(z):
    z = np.clip(z, -500, 500) # Prevent overflow
    return 1 / (1 + np.exp(-z))

def cost_function_reg(theta, X, y, lmbda):
    m = y.size
    h = sigmoid(X @ theta)
    
    # Logistic loss with regularization
    term1 = -y * np.log(h + 1e-15) # epsilon added for numerical stability
    term2 = (1 - y) * np.log(1 - h + 1e-15)
    cost = np.mean(term1 - term2)
    reg_term = (lmbda / (2 * m)) * np.sum(np.square(theta[1:]))
    
    # Gradient calculation
    grad = (1/m) * (X.T @ (h - y))
    grad[1:] += (lmbda / m) * theta[1:]
    
    return cost + reg_term, grad

def train_momentum(X, y, lr=0.1, beta=0.9, lmbda=1, epochs=5000):
    m, n = X.shape
    theta = np.zeros(n)
    v = np.zeros(n)
    for _ in range(epochs):
        cost, grad = cost_function_reg(theta, X, y, lmbda)
        v = (beta * v) + (1 - beta) * grad
        theta -= lr * v
    return theta

def map_feature(X1, X2, degree=6):
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.hstack((out, (X1**(i-j) * X2**j).reshape(-1, 1)))
    return out

# --- Execution ---
# Dataset 1: University Admission (Linear)
data1 = pd.read_csv("data/data1.txt", names=["Exam1", "Exam2", "Admitted"])
X1_raw = data1[["Exam1", "Exam2"]].values
y1 = data1["Admitted"].values

X1_mean = np.mean(X1_raw, axis=0)
X1_std = np.std(X1_raw, axis=0)
X1_scaled = (X1_raw - X1_mean) / X1_std

# Add Intercept
X1 = np.column_stack((np.ones(len(X1_scaled)), X1_scaled))

theta1 = train_momentum(X1, y1, lr=0.1, beta=0.9, lmbda=0, epochs=1000) # Lambda=0 for linear



# Dataset 2: Microchips (Non-linear)
data2 = pd.read_csv("data/data2.txt", names=["Test1", "Test2", "Accepted"])
X_mapped = map_feature(data2["Test1"].values, data2["Test2"].values)
y2 = data2["Accepted"].values
theta2 = train_momentum(X_mapped, y2, lr=0.01, beta=0.9, lmbda=1, epochs=10000)



print("Training Complete!")
p1 = (sigmoid(X1 @ theta1) >= 0.5).astype(int)
print(f"Dataset 1 Accuracy: {np.mean(p1 == y1) * 100:.2f}%")

p2 = (sigmoid(X_mapped @ theta2) >= 0.5).astype(int)
print(f"Dataset 2 Accuracy: {np.mean(p2 == y2) * 100:.2f}%")

# Plotting
plot_linear_boundary(theta1, X1_scaled, y1, X1_mean, X1_std)
plot_nonlinear_boundary(theta2, y2)

print("Generating ROC Curves...")

# ROC for Dataset 1
plot_roc_curve(theta1, X1, y1, "University Admission")

# ROC for Dataset 2
plot_roc_curve(theta2, X_mapped, y2, "Microchip QA")