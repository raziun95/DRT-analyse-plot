__authors__ = 'Francesco Ciucci, Adeleke Maradesa, Baptiste Py'

__date__ = '22nd January, 2024'

## import function

import numpy as np
import sys
import matplotlib.pyplot as plt
from astropy.stats import biweight_location
from scipy.stats import ttest_1samp
from matplotlib.patches import Rectangle

#Peak analysis


def gauss_fct(p, tau, N_peaks): # N_peaks is the number of peaks in the DRT spectrum
    
    gamma_out = np.zeros_like(tau) # sum of Gaussian functions, whose parameters (the prefactor sigma_f, mean mu_log_tau, and standard deviation 1/inv_sigma for each DRT peak) are encapsulated in p
    
    for k in range(N_peaks):
        
        sigma_f, mu_log_tau, inv_sigma = p[3*k:3*k+3] 
        gaussian_out = sigma_f**2*np.exp(-inv_sigma**2/2*((np.log(tau) - mu_log_tau)**2)) # we use inv_sigma because this leads to less computational problems (no exploding gradient when sigma->0)
        gamma_out += gaussian_out 
    return gamma_out 


# detect peaks

def detect_peaks(gamma, alpha):
    # alpha is the significance level 
    # Use Biweight estimator as the measure of central tendency
    median = biweight_location(gamma)
    peaks = []

    # Iterate over the data points
    for k in range(1, len(gamma) - 1):
        # Select the current data point and its neighbors
        x = gamma[k]
        x_prev = gamma[k - 1]
        x_next = gamma[k + 1]

        # Check if the current data point is above the threshold and greater than its neighbors
        if x > alpha * median and x > x_prev and x > x_next:
            # Perform a one-sample t-test to determine if the data point is significantly larger than the median 
            t_statistic, p_value = ttest_1samp([x, x_prev, x_next], median)
            if p_value < alpha:  # alpha is the significance level 
                peaks.append(k)

    return peaks 


##
def calculate_peak_likelihood(data, peaks):
    ### Peak Scores
    peak_scores = []

    for peak_index in peaks:
        peak_height = data[peak_index]
        peak_scores.append((peak_index, peak_height))

    # Calculate the total score of all peaks
    total_score = sum(score for _, score in peak_scores)

    likelihoods = [(index, (score / total_score)) for index, score in peak_scores]
    ## print likelihood of peak occurence
    for peak_index, likelihood in likelihoods:
        print(f"Likelihood for peak occurrence at {peak_index}: {likelihood:.5f}")

    return likelihoods


def plot_gamma_with_peaks(tau,gamma, detected_peaks):
    # Plot gamma with 'o' indicating the detected peaks
    ##
    peak_likelihoods = calculate_peak_likelihood(gamma, detected_peaks)
    #gamma_fit = peak_analysis(tau,gamma, N_peaks, method = 'combine')
    ##
    fig, ax = plt.subplots()
    ax.semilogx(tau, gamma, linewidth = 3)
    
    ax.scatter(tau[detected_peaks], gamma[detected_peaks], color='red', marker='o')
    ax.semilogx(tau, gamma, color='black', linewidth=3)
    for peak_index, likelihood in peak_likelihoods:
        ax.text(tau[peak_index], gamma[peak_index] + 0.0002, f'({likelihood:.3f})', ha='center', va='bottom', fontsize=20)

    # Set the x-axis and y-axis labels
    ax.set_xlabel(r'$\tau/{\rm s}$',fontsize = 20)
    ax.set_ylabel(r'$\gamma/\Omega$',fontsize = 20)
    
    ax.set_ylim(min(gamma),round((round(max(gamma),2)+0.01/2),3))  #round(max(gamma),2)+round(max(gamma),2)/2

    # Add bar to touch each peak
    peak_tau = [tau[k] for k in detected_peaks]
    for x, y in zip(peak_tau, [gamma[k] for k in detected_peaks]):
        plt.plot([x, x], [0, y], 'g-', linewidth=4) 
    # Show the plot
    plt.show()
