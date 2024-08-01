import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X.dot(w) + b
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def compute_gradient(X, y, w, b):
    m = len(y)
    predictions = X.dot(w) + b
    error = predictions - y
    dw = (1 / m) * X.T.dot(error)
    db = (1 / m) * np.sum(error)
    return dw, db


def gradient_descent(X, y, w_init, b_init, alpha, iterations, compute_cost, compute_gradient):
    w = w_init
    b = b_init
    J_hist = []
    p_hist = []

    for i in range(iterations):
        cost = compute_cost(X, y, w, b)
        J_hist.append(cost)
        dw, db = compute_gradient(X, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        
        # Print cost and parameters at intervals
        if i % (iterations // 10) == 0:
            print(f"Iteration {i:4}: cost {J_hist[-1]:0.2e}, w: {w.flatten()}, b: {b:0.5e}")
    
    return w, b, J_hist, p_hist
