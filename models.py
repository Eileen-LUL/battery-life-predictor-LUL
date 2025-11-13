import numpy as np
from scipy.optimize import curve_fit

# Double-exponential model for capacity fade
def model(x, A1, k1, A2, k2, baseline):
    return A1*np.exp(-k1*x) + A2*np.exp(-k2*x) + baseline

def fit_capacity_fade(cycles, capacity):
    guess = [0.1, 0.001, 0.05, 0.0001, capacity[-1]]
    params, _ = curve_fit(model, cycles, capacity, p0=guess)
    return params

def predict_cycles_to_eol(params, threshold):
    A1, k1, A2, k2, baseline = params

    # Solve A1 e^(-k1 x) + A2 e^(-k2 x) + baseline = threshold
    for i in range(1, 50000):
        cap = A1*np.exp(-k1*i) + A2*np.exp(-k2*i) + baseline
        if cap <= threshold:
            return i
    return None
