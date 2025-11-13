import numpy as np
from scipy.optimize import curve_fit

# Double exponential model
def model(x, A1, k1, A2, k2, baseline):
    return A1 * np.exp(-k1 * x) + A2 * np.exp(-k2 * x) + baseline

def fit_capacity_fade(cycles, capacity):

    # Automatic initial guesses based on data
    A1_guess = capacity.max() - capacity.min()
    A2_guess = (capacity.max() - capacity.min()) * 0.1
    k1_guess = 0.001
    k2_guess = 0.0001
    baseline_guess = capacity[-1]

    guess = [A1_guess, k1_guess, A2_guess, k2_guess, baseline_guess]

    # Add bounds to stabilize fit
    lower_bounds = [0, 0, 0, 0, 0]
    upper_bounds = [10, 1, 10, 1, 2]  # adjust if needed

    try:
        params, _ = curve_fit(
            model, cycles, capacity,
            p0=guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000
        )
    except Exception as e:
        # If fit fails → use fallback linear model
        print("Fit failed, using fallback model:", e)
        slope, intercept = np.polyfit(cycles, capacity, 1)
        return [0, 0, 0, 0, intercept]  # baseline-only fallback

    return params
# Predict cycle life to 80% SOH (EOL)
def predict_cycles_to_eol(params, soh_threshold=0.8):
    A1, k1, A2, k2, baseline = params

    # Solve model(x) = soh_threshold * initial_capacity
    initial_capacity = A1 + A2 + baseline
    target = soh_threshold * initial_capacity

    # Search numerically for cycle where capacity drops below target
    for cycle in range(1, 5000):
        cap = A1 * np.exp(-k1 * cycle) + A2 * np.exp(-k2 * cycle) + baseline
        if cap <= target:
            return cycle

    return None  # fallback



