import numpy as np
from scipy.optimize import curve_fit

# ================================
# Double Exponential Model
# ================================
def model(x, A1, k1, A2, k2, baseline):
    return A1 * np.exp(-k1 * x) + A2 * np.exp(-k2 * x) + baseline


# ================================
# Capacity Fade Fitting
# ================================
def fit_capacity_fade(cycles, capacity):

    # Automatic initial guesses based on your uploaded data
    A1_guess = capacity.max() - capacity.min()
    A2_guess = (capacity.max() - capacity.min()) * 0.1
    k1_guess = 0.001
    k2_guess = 0.0001
    baseline_guess = capacity[-1]

    guess = [A1_guess, k1_guess, A2_guess, k2_guess, baseline_guess]

    # Add bounds to make sure curve_fit does not diverge
    lower_bounds = [0, 0, 0, 0, 0]
    upper_bounds = [10, 1, 10, 1, 2]

    try:
        params, _ = curve_fit(
            model,
            cycles,
            capacity,
            p0=guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000
        )
    except Exception as e:
        # If fitting fails → fallback to linear approximation
        print("Fit failed, using fallback model:", e)
        slope, intercept = np.polyfit(cycles, capacity, 1)
        return [0, 0, 0, 0, intercept]

    return params


# ================================
# Predict EOL Cycle (e.g., 80% SOH)
# ================================
def predict_cycles_to_eol(params, soh_threshold=0.8):

    A1, k1, A2, k2, baseline = params

    # initial capacity of the fitted model
    initial_capacity = A1 + A2 + baseline

    # target capacity at the given SOH threshold
    eol_cycle = predict_cycles_to_eol(params, soh_threshold=0.8)


    # numerical search for cycle at which model drops below threshold
    for cycle in range(1, 5000):
        cap = A1 * np.exp(-k1 * cycle) + A2 * np.exp(-k2 * cycle) + baseline
        if cap <= target:
            return cycle

    return None  # fallback if never drops below

