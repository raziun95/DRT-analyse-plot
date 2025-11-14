# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 15:58:31 2025

@author: Raziun

Title: DRT analysis for PEMFCs
"""

# Python modules 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import pandas as pd
import os
import glob
import sys
from scipy.optimize import minimize
from scipy.signal import find_peaks
from numpy.linalg import norm
from numpy import exp, log, pi
import warnings
warnings.filterwarnings('ignore')

# Add pyDRTtools_ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyDRTtools_'))

# pyDRTtools_ modules
import pyDRTtools
import pyDRTtools.basics as basics
import pyDRTtools.runs as runs
from pyDRTtools.runs import EIS_object

# Import AIC_fun for peak fitting
import AIC_fun as aic
from AIC_fun import gauss_fct

## for nice plot
plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

####### Initialize DRT parameters
# DRT parameters
rbf_type = 'Gaussian'  # RBF type: 'Gaussian', 'C0 Matern', 'C2 Matern', etc. (capitalized!)
shape_control = 'FWHM Coefficient'  # Shape control method (exact string!)
coeff = 0.5  # FWHM coefficient
der_used = '2nd order'  # Derivative order: '1st order' or '2nd order'
data_used = 'Combined Re-Im Data'  # Data used for regularization
induct_used = 1  # Include inductance (1) or not (0)
cv_type = 'GCV'  # Cross-validation type
reg_param = 1E-3  # Initial regularization parameter

####### Load and analyze many EIS spectra concomitantly

# Get list of eisraw files from the correct folder
data_folder = '00 PUT EIS DATA HERE'
eis_files = sorted(glob.glob(os.path.join(data_folder, 'eisraw_*.csv')))
N_exp = len(eis_files)

print(f"Found {N_exp} EIS files in '{data_folder}' folder")

# Storage for all results
results_dict = {
    'simple': {'gamma': [], 'tau': [], 'Z': [], 'R_inf': [], 'L_0': [], 'lambda': [], 'params': []},
    'bayesian': {'gamma_mean': [], 'gamma_lower': [], 'gamma_upper': [], 'tau': [], 'params': []},
    'ht': {'gamma_re': [], 'gamma_im': [], 'tau': [], 'scores': [], 'params': []}
}

# Parameters used for each file
parameters_list = []

# Peak fitting parameters
N_peak = 4

# Lists to store peak fitting results
peak_fit_params_list = []
peak_areas_list = []
successful_files = []
successful_indices = []

for n in range(N_exp):
    print(f"\n{'='*60}")
    print(f"Processing file {n+1}/{N_exp}: {os.path.basename(eis_files[n])}")
    print(f"{'='*60}")
    
    try:
        # step 1: load the experimental data
        df = pd.read_csv(eis_files[n], header=None)
        df.columns = ['Freq', 'Real', 'Imag']
        
        N_freqs = df.shape[0]
        freq_vec = np.flip(df['Freq'].values)
        Z_prime = df['Real'].values
        Z_double_prime = df['Imag'].values
        Z_exp = Z_prime + 1j*Z_double_prime
        
        # Store parameters used
        file_params = {
            'filename': os.path.basename(eis_files[n]),
            'N_freqs': N_freqs,
            'freq_min': freq_vec.min(),
            'freq_max': freq_vec.max(),
            'rbf_type': rbf_type,
            'shape_control': shape_control,
            'coeff': coeff,
            'der_used': der_used,
            'data_used': data_used,
            'induct_used': induct_used,
            'cv_type': cv_type
        }
        
        # step 2: Create EIS_object
        # Note: EIS_object expects freq, Z_prime, Z_double_prime
        # It automatically sets tau = 1/freq
        eis_obj = EIS_object(freq_vec, Z_prime, Z_double_prime)
        
        # step 3: Simple DRT analysis
        print("  Running Simple DRT analysis...")
        try:
            eis_obj = runs.simple_run(eis_obj, rbf_type=rbf_type, 
                                     data_used=data_used,
                                     induct_used=induct_used,
                                     der_used=der_used,
                                     cv_type=cv_type,
                                     reg_param=reg_param,
                                     shape_control=shape_control,
                                     coeff=coeff)
            
            # Extract results
            gamma_simple = eis_obj.gamma
            tau_simple = eis_obj.out_tau_vec
            R_inf = eis_obj.R
            L_0 = eis_obj.L
            lambda_opt = eis_obj.lambda_value
            
            # Reconstruct impedance from DRT
            Z_DRT_simple = eis_obj.mu_Z_re + 1j*eis_obj.mu_Z_im
            
            file_params['lambda_optimal'] = lambda_opt
            file_params['R_inf'] = R_inf
            file_params['L_0'] = L_0
            file_params['method_simple'] = 'success'
            
            results_dict['simple']['gamma'].append(gamma_simple)
            results_dict['simple']['tau'].append(tau_simple)
            results_dict['simple']['Z'].append(Z_DRT_simple)
            results_dict['simple']['R_inf'].append(R_inf)
            results_dict['simple']['L_0'].append(L_0)
            results_dict['simple']['lambda'].append(lambda_opt)
            results_dict['simple']['params'].append(file_params.copy())
            
            print(f"    Simple DRT completed. Lambda: {lambda_opt:.2e}, R_inf: {R_inf:.4f}, L_0: {L_0:.2e}")
            
        except Exception as e:
            print(f"    Simple DRT failed: {e}")
            import traceback
            traceback.print_exc()
            file_params['method_simple'] = f'failed: {str(e)}'
            continue
        
        # step 4: Bayesian DRT analysis
        print("  Running Bayesian DRT analysis...")
        try:
            NMC_sample = 5000
            eis_obj_bayesian = runs.Bayesian_run(eis_obj, rbf_type=rbf_type,
                                                data_used=data_used,
                                                induct_used=induct_used,
                                                der_used=der_used,
                                                cv_type=cv_type,
                                                reg_param=reg_param,
                                                shape_control=shape_control,
                                                coeff=coeff,
                                                NMC_sample=NMC_sample)
            
            gamma_mean = eis_obj_bayesian.mean
            gamma_lower = eis_obj_bayesian.lower_bound
            gamma_upper = eis_obj_bayesian.upper_bound
            tau_bayesian = eis_obj_bayesian.out_tau_vec
            
            results_dict['bayesian']['gamma_mean'].append(gamma_mean)
            results_dict['bayesian']['gamma_lower'].append(gamma_lower)
            results_dict['bayesian']['gamma_upper'].append(gamma_upper)
            results_dict['bayesian']['tau'].append(tau_bayesian)
            
            bayesian_params = file_params.copy()
            bayesian_params['NMC_sample'] = NMC_sample
            bayesian_params['method_bayesian'] = 'success'
            results_dict['bayesian']['params'].append(bayesian_params)
            
            print(f"    Bayesian DRT completed with {NMC_sample} samples")
            
        except Exception as e:
            print(f"    Bayesian DRT failed: {e}")
            bayesian_params = file_params.copy()
            bayesian_params['method_bayesian'] = f'failed: {str(e)}'
            results_dict['bayesian']['params'].append(bayesian_params)
        
        # step 5: Hilbert Transform DRT analysis
        print("  Running Hilbert Transform DRT analysis...")
        try:
            # Create a new EIS_object for HT (it modifies the object)
            eis_obj_ht = EIS_object(freq_vec, Z_prime, Z_double_prime)
            eis_obj_ht = runs.BHT_run(eis_obj_ht, rbf_type=rbf_type,
                                     der_used=der_used,
                                     shape_control=shape_control,
                                     coeff=coeff)
            
            gamma_re = eis_obj_ht.mu_gamma_fine_re
            gamma_im = eis_obj_ht.mu_gamma_fine_im
            tau_ht = eis_obj_ht.out_tau_vec
            
            # Extract scores if available
            scores = {}
            if hasattr(eis_obj_ht, 'out_scores'):
                scores = {
                    's_res_re': eis_obj_ht.out_scores.get('s_res_re', None),
                    's_res_im': eis_obj_ht.out_scores.get('s_res_im', None),
                    's_mu_re': eis_obj_ht.out_scores.get('s_mu_re', None),
                    's_mu_im': eis_obj_ht.out_scores.get('s_mu_im', None),
                    's_HD_re': eis_obj_ht.out_scores.get('s_HD_re', None),
                    's_HD_im': eis_obj_ht.out_scores.get('s_HD_im', None),
                    's_JSD_re': eis_obj_ht.out_scores.get('s_JSD_re', None),
                    's_JSD_im': eis_obj_ht.out_scores.get('s_JSD_im', None)
                }
            
            results_dict['ht']['gamma_re'].append(gamma_re)
            results_dict['ht']['gamma_im'].append(gamma_im)
            results_dict['ht']['tau'].append(tau_ht)
            results_dict['ht']['scores'].append(scores)
            
            ht_params = file_params.copy()
            ht_params['method_ht'] = 'success'
            ht_params['scores'] = scores
            results_dict['ht']['params'].append(ht_params)
            
            print(f"    HT DRT completed")
            if scores:
                print(f"      EIS scores: s_res_re={scores.get('s_res_re', 'N/A')}, s_HD_re={scores.get('s_HD_re', 'N/A')}")
            
        except Exception as e:
            print(f"    HT DRT failed: {e}")
            ht_params = file_params.copy()
            ht_params['method_ht'] = f'failed: {str(e)}'
            results_dict['ht']['params'].append(ht_params)
        
        # step 6: Peak fitting on simple DRT result
        print("  Fitting peaks...")
        try:
            gamma_data = results_dict['simple']['gamma'][-1]
            tau_vec_peak = results_dict['simple']['tau'][-1]
            
            # Find initial peaks
            peaks, _ = find_peaks(gamma_data, height=1E-10)
            
            if len(peaks) == 0:
                print("    No peaks found, skipping peak fitting")
            else:
                gamma_sorted_peak = np.sort(gamma_data[peaks])
                sorted_index = peaks[np.argsort(gamma_data[peaks])]
                
                N_find_peak = len(peaks)
                p_init = np.array([])
                
                # Set up initial peak positions
                N_peak_use = min(N_peak, N_find_peak)
                for m in range(1, N_peak_use+1):
                    index_n = sorted_index[-m]
                    p_n = np.array([log(gamma_data[index_n]), log(tau_vec_peak[index_n]), log(0.3)], 
                                  dtype=np.float64)
                    p_init = np.concatenate([p_init, p_n])
                
                N_gauss = int(p_init.shape[0]/3)
                p_init = p_init.reshape(N_gauss, 3)
                p_init = p_init[p_init[:,1].argsort()]
                p_init = p_init.reshape(p_init.size, 1).T[0]
                
                bnds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)] * N_gauss
                
                # Fit peaks
                residual_fct = lambda p: norm(gauss_fct(tau_vec_peak, p) - gamma_data)**2
                res = minimize(residual_fct, p_init, method='L-BFGS-B',
                              options={'gtol': 1E-10, 'disp': False, 'maxiter': 100000},
                              bounds=bnds)
                fit_param = res.x
                
                # Calculate areas
                R_pol_vec = np.array([])
                for i in range(0, len(fit_param), 3):
                    R_0 = exp(fit_param[i])
                    tau_0 = exp(fit_param[i+1])
                    sigma = exp(fit_param[i+2])
                    gauss_gamma = R_0 * np.exp(-(log(tau_vec_peak) - log(tau_0))**2/(2*sigma)**2)
                    R_pol = np.trapezoid(gauss_gamma, log(tau_vec_peak))
                    R_pol_vec = np.append(R_pol_vec, R_pol)
                
                peak_fit_params_list.append(fit_param)
                peak_areas_list.append(R_pol_vec)
                successful_files.append(os.path.basename(eis_files[n]))
                successful_indices.append(n)
                
                print(f"    Peak fitting completed. Areas: {R_pol_vec}")
            
        except Exception as e:
            print(f"    Peak fitting failed: {e}")
        
        # Store parameters
        parameters_list.append(file_params)
        
    except Exception as e:
        print(f"  Error processing file: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print(f"Successfully processed {len(successful_indices)} out of {N_exp} files")
print(f"{'='*60}")

####### Save parameters to CSV
print("\nSaving parameters to CSV...")
params_df = pd.DataFrame(parameters_list)
os.makedirs('results', exist_ok=True)
params_df.to_csv('results/DRT_parameters_used.csv', index=False)
print("  Saved: results/DRT_parameters_used.csv")

####### Generate plots and save results
if len(successful_indices) > 0:
    # Extract time labels
    time_labels = []
    for f in successful_files:
        name = os.path.basename(f).replace('eisraw_', '').replace('.csv', '')
        time_labels.append(name)
    
    color_list = ['black', 'blue', 'magenta', 'cyan', 'green', 'yellow', 'orange', 'red']
    
    # Plot 1: Simple DRT spectra
    if len(results_dict['simple']['gamma']) > 0:
        plt.figure(figsize=(10, 6))
        for idx in range(len(results_dict['simple']['gamma'])):
            plt.semilogx(results_dict['simple']['tau'][idx], 
                        results_dict['simple']['gamma'][idx],
                        linewidth=2, color=color_list[idx % len(color_list)],
                        label=time_labels[idx] if idx < len(time_labels) else f'File {idx+1}')
        plt.xlabel(r'$\tau$ (s)', fontsize=16)
        plt.ylabel(r'$\gamma(\tau)$ ($\Omega$)', fontsize=16)
        plt.title('Simple DRT Spectra', fontsize=16)
        plt.legend(frameon=False, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs('result plots', exist_ok=True)
        plt.savefig('result plots/Simple_DRT_spectra.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: result plots/Simple_DRT_spectra.png")
    
    # Plot 2: Bayesian DRT with credible intervals
    if len(results_dict['bayesian']['gamma_mean']) > 0:
        plt.figure(figsize=(10, 6))
        for idx in range(len(results_dict['bayesian']['gamma_mean'])):
            tau = results_dict['bayesian']['tau'][idx]
            gamma_mean = results_dict['bayesian']['gamma_mean'][idx]
            gamma_lower = results_dict['bayesian']['gamma_lower'][idx]
            gamma_upper = results_dict['bayesian']['gamma_upper'][idx]
            
            color = color_list[idx % len(color_list)]
            plt.semilogx(tau, gamma_mean, linewidth=2, color=color,
                        label=time_labels[idx] if idx < len(time_labels) else f'File {idx+1}')
            plt.fill_between(tau, gamma_lower, gamma_upper, alpha=0.2, color=color)
        
        plt.xlabel(r'$\tau$ (s)', fontsize=16)
        plt.ylabel(r'$\gamma(\tau)$ ($\Omega$)', fontsize=16)
        plt.title('Bayesian DRT Spectra with 99% Credible Intervals', fontsize=16)
        plt.legend(frameon=False, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('result plots/Bayesian_DRT_spectra.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: result plots/Bayesian_DRT_spectra.png")
    
    # Plot 3: Peak areas vs time
    if len(peak_areas_list) > 0:
        plt.figure(figsize=(10, 6))
        time_hours = [int(label.replace('h', '')) for label in time_labels]
        N_peaks = len(peak_areas_list[0])
        
        for peak_idx in range(N_peaks):
            areas = [peak_areas_list[idx][peak_idx] if peak_idx < len(peak_areas_list[idx]) else 0
                    for idx in range(len(peak_areas_list))]
            plt.plot(time_hours, areas, 'o-', linewidth=2, markersize=8,
                    label=f'Peak {peak_idx+1}', color=color_list[peak_idx % len(color_list)])
        
        plt.xlabel('Time (hours)', fontsize=16)
        plt.ylabel(r'Area under peak ($\Omega$)', fontsize=16)
        plt.title('Peak Areas vs Time', fontsize=16)
        plt.legend(frameon=False, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('result plots/Peak_areas_vs_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: result plots/Peak_areas_vs_time.png")
    
    # Save DRT results to CSV
    print("\nSaving DRT results to CSV...")
    for idx, n in enumerate(successful_indices):
        time_label = time_labels[idx]
        
        # Simple DRT
        if idx < len(results_dict['simple']['gamma']):
            df_simple = pd.DataFrame({
                'tau': results_dict['simple']['tau'][idx],
                'gamma': results_dict['simple']['gamma'][idx]
            })
            df_simple.to_csv(f'results/Simple_DRT_{time_label}.csv', index=False)
            
            # Save impedance - need to get freq from the file
            try:
                df_temp = pd.read_csv(eis_files[n], header=None)
                df_temp.columns = ['Freq', 'Real', 'Imag']
                freq_save = np.flip(df_temp['Freq'].values)
            except:
                freq_save = np.arange(len(results_dict['simple']['Z'][idx]))
            
            df_Z = pd.DataFrame({
                'Freq': freq_save,
                'Real': np.real(results_dict['simple']['Z'][idx]),
                'Imag': np.imag(results_dict['simple']['Z'][idx])
            })
            df_Z.to_csv(f'results/Simple_Z_{time_label}.csv', index=False)
        
        # Bayesian DRT
        if idx < len(results_dict['bayesian']['gamma_mean']):
            df_bayesian = pd.DataFrame({
                'tau': results_dict['bayesian']['tau'][idx],
                'gamma_mean': results_dict['bayesian']['gamma_mean'][idx],
                'gamma_lower': results_dict['bayesian']['gamma_lower'][idx],
                'gamma_upper': results_dict['bayesian']['gamma_upper'][idx]
            })
            df_bayesian.to_csv(f'results/Bayesian_DRT_{time_label}.csv', index=False)
        
        # HT DRT
        if idx < len(results_dict['ht']['gamma_re']):
            df_ht = pd.DataFrame({
                'tau': results_dict['ht']['tau'][idx],
                'gamma_re': results_dict['ht']['gamma_re'][idx],
                'gamma_im': results_dict['ht']['gamma_im'][idx]
            })
            df_ht.to_csv(f'results/HT_DRT_{time_label}.csv', index=False)
    
    print("  DRT results saved to results/ folder")

print("\n" + "="*60)
print("Analysis complete!")

print("="*60)
