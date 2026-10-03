import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

# A1. Perceptron Modules
def summation_unit(X, W, bias):
    """Computes the weighted sum."""
    return np.dot(X, W) + bias

def activation_unit(y, func_type='step'):
    """Applies the selected activation function."""
    if func_type == 'step':
        return np.where(y >= 0, 1, 0)
    elif func_type == 'bipolar_step':
        return np.where(y > 0, 1, np.where(y == 0, 0, -1))
    elif func_type == 'sigmoid':
        return 1 / (1 + np.exp(-y))
    elif func_type == 'tanh':
        return np.tanh(y)
    elif func_type == 'relu':
        return np.maximum(0, y)
    elif func_type == 'leaky_relu':
        return np.where(y > 0, y, 0.01 * y)

def comparator_unit(target, prediction):
    """Calculates the error."""
    return target - prediction

# A2. Perceptron Learning Implementation
def train_perceptron(X, Y, initial_w, initial_bias, lr=0.05, epochs=1000, act_func='step'):
    W = np.array(initial_w, dtype=float)
    bias = initial_bias
    errors = []

    for epoch in range(epochs):
        total_sse = 0
        for i in range(len(X)):
            # Forward pass
            y_in = summation_unit(X[i], W, bias)
            y_pred = activation_unit(y_in, act_func)
            
            # Error calculation
            error = comparator_unit(Y[i], y_pred)
            total_sse += error ** 2
            
            # Weight update
            W += lr * error * X[i]
            bias += lr * error
            
        errors.append(total_sse)
        if total_sse <= 0.002:
            break
            
    return W, bias, errors, epoch + 1

# Main Execution Blocks
if __name__ == "__main__":
    # Logic Gate Data
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_and = np.array([0, 0, 0, 1])
    
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_xor = np.array([0, 1, 1, 0])

    # A2: AND Gate with Step Activation
    w, b, errs, iters = train_perceptron(X_and, Y_and, [0.2, -0.75], 10, lr=0.05, act_func='step')
    print(f"AND Gate (Step) converged in {iters} iterations.")
    
    plt.plot(range(1, iters + 1), errs)
    plt.title("Epochs vs Sum-Square-Error (AND Gate - Step)")
    plt.xlabel("Epochs")
    plt.ylabel("SSE")
    plt.show()

    # A4: Varying Learning Rates
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    iters_needed = []
    for lr in learning_rates:
        _, _, _, it = train_perceptron(X_and, Y_and, [0.2, -0.75], 10, lr=lr, act_func='step')
        iters_needed.append(it)
        
    plt.plot(learning_rates, iters_needed, marker='o')
    plt.title("Learning Rate vs Iterations to Converge")
    plt.xlabel("Learning Rate")
    plt.ylabel("Iterations")
    plt.show()

    # A6: Customer Data Classification
    X_cust = np.array([
        [20, 6, 2], [16, 3, 6], [27, 6, 2], [19, 1, 2], [24, 4, 2],
        [22, 1, 5], [15, 4, 2], [18, 4, 2], [21, 1, 4], [16, 2, 4]
    ])
    Y_cust = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0]) # 1: High, 0: Low

    w_cust, b_cust, err_cust, iters_cust = train_perceptron(
        X_cust, Y_cust, [0.1, 0.1, 0.1], 0.5, lr=0.05, act_func='sigmoid'
    )
    print(f"Customer Perceptron converged in {iters_cust} iterations.")

    # A7: Matrix Pseudo-Inverse
    X_pinv = np.c_[np.ones(X_cust.shape[0]), X_cust]
    W_pinv = np.linalg.pinv(X_pinv).dot(Y_cust)
    print(f"Weights via Pseudo-Inverse: {W_pinv}")

    # A11 & A12: Sci-Kit MLP Classifier (AND, XOR, and Project Dataset)
    mlp_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', learning_rate_init=0.05, max_iter=1000)
    mlp_and.fit(X_and, Y_and)
    print("MLP AND Gate Predictions:", mlp_and.predict(X_and))

    mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', learning_rate_init=0.05, max_iter=1000)
    mlp_xor.fit(X_xor, Y_xor)
    print("MLP XOR Gate Predictions:", mlp_xor.predict(X_xor))

    # Project Context: FoodLinkAI Expiration Urgency with MLP
    # Re-using X and y from Lab 07 block
    X_food = np.random.uniform(1, 20, (50, 3)) 
    Y_food = np.random.choice([0, 1], 50)
    mlp_project = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', max_iter=1000)
    mlp_project.fit(X_food, Y_food)
    print("Project Data MLP Score:", mlp_project.score(X_food, Y_food))
