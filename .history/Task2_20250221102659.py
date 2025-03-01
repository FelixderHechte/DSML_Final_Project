import numpy as np
import psd  

def generate_time_series(model, T):
    """
    Generate a time series of length T using a random initial condition.
    This function assumes that your model accepts an initial condition and a length.
    """
    init_condition = np.random.randn()  # or np.random.randn(d) if multi-dimensional
    return model(init_condition, T)

def evaluate_model(model, ground_truth, sigma=20):
    """
    Generate a time series using the model and compute the power spectrum error.
    
    Parameters:
        model: a callable that takes an initial condition and T, returns a time series.
        ground_truth: numpy array of the true time series data with shape [batch, T, dimensions].
        sigma: smoothing parameter for the PSD (should match SMOOTHING_SIGMA in psd.py).
        
    Returns:
        psd_error: The computed power spectrum distance between the generated and true series.
    """
    T = ground_truth.shape[1]  # assuming shape [batch, T, dimensions]
    x_gen = generate_time_series(model, T)
    
    # If the model returns a single trajectory, you might need to add a batch dimension
    if x_gen.ndim == 2:
        x_gen = np.expand_dims(x_gen, axis=0)
    
    # Compute the power spectrum error
    psd_error = psd.power_spectrum_error(x_gen, ground_truth)
    return psd_error

# Example usage:
# Assume your_model is defined elsewhere and ground_truth is loaded with shape [batch, T, dimensions]
 
ground_truth = np.load(r"lorenz63_test.npy")
print(ground_truth.shape)
# error = evaluate_model(your_model, ground_truth)
# print("Power-Spectrum Distance Error:", error)