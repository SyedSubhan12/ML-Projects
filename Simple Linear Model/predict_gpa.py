import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from functions import compute_cost, compute_gradient, gradient_descent

# Import the dataset
data = pd.read_csv("Simple Linear Model/regression.csv")
df = pd.DataFrame(data)
X = df[['SAT']].values
y = df['GPA'].values
y = y.reshape(-1, 1)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[:, 1:])  # Skip the bias term column for scaling

# Add the bias term back
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

# Split the dataset into 80|20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize weights and bias
w_init = np.zeros((X_scaled.shape[1], 1))  # Initialize weights (including bias term)
b_init = 0  # Initialize bias

iterations = 10000
alpha = 0.0001  # Reduced learning rate

# Perform gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(X_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient)
print(f"(w, b) found by gradient descent: ({w_final.flatten()}, {b_final:8.4f})")

def predict_gpa(sat_score, w, b):
    # Scale the input SAT score
    scaled_sat_score = scaler.transform(np.array([[sat_score]]))
    # Add bias term
    X = np.hstack((np.ones((scaled_sat_score.shape[0], 1)), scaled_sat_score))
    predicted_gpa = np.dot(X, w) + b
    return predicted_gpa

def main():
    sat_score = float(input("Enter the SAT score: "))
    
    predicted_gpa = predict_gpa(sat_score, w_final, b_final)
    
    print(f"Predicted GPA for SAT score {sat_score}: {predicted_gpa[0, 0]:.2f}")

if __name__ == "__main__":
    main()

