
__authors__ = 'Francesco Ciucci, Adeleke Maradesa, Baptiste Py, Ting Hei Wan, Mohammed Effat'

__date__ = '23rd January, 2024'

import numpy as np
import sys
import cvxopt
from cvxopt import matrix, solvers
from numpy import exp
from math import log
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from numpy.linalg import norm, cholesky
from numpy import *
from . import basics
#import basics
print(sys.path)



# Part 1: nearest positive definite   

def is_PD(A):
    
    """
       This function checks if a matrix A is positive-definite using Cholesky transform
    """
    
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

    
def nearest_PD(A):
    
    """
       This function finds the nearest positive definite matrix of a matrix A. The code is based on John D'Errico's "nearestSPD" code on Matlab [1]. More details can be found in the following two references:
         https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
         N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    
    B = (A + A.T)/2
    _, Sigma_mat, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(Sigma_mat), V))

    A_nPD = (B + H) / 2
    A_symm = (A_nPD + A_nPD.T) / 2

    k = 1
    I = np.eye(A_symm.shape[0])

    while not is_PD(A_symm): # the Matlab function chol accepts matrices with eigenvalue = 0, but numpy does not so we replace the Matlab function eps(min_eig) with the following one
        
        eps = np.spacing(np.linalg.norm(A_symm))
        min_eig = min(0, np.min(np.real(np.linalg.eigvals(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1

    return A_symm

# Part 2: Selection of the regularization parameter for ridge regression


"""

A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the Regularization Parameter in the Distribution of Relaxation Times, 
Journal of the Electrochemical Society, 170 (2023) 030502.

"""


def compute_GCV(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the score for the generalized cross-validation (GCV) approach.
       Reference: G. Wahba, A comparison of GCV and GML for choosing the smoothing parameter in the generalized spline smoothing problem, Ann. Statist. 13 (1985) 1378–1402.
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           GCV score
    """
    
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # matrix A with A_re and A_im ; see (5) in [4]
    Z = np.concatenate((Z_re, Z_im), axis = 0) # stacked impedance
    
    n_cv = Z.shape[0] # n_cv = 2*N_freqs with N_freqs the number of EIS frequencies
    
    A_agm = A.T@A + lambda_value*M # see (13) in [4]
    
    if (is_PD(A_agm)==False): # check if A_agm is positive-definite
        A_agm = nearest_PD(A_agm) 
        
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    A_GCV = A@inv_A_agm@A.T  # see (13) in [4]
    
    # GCV score; see (13) in [4]
    GCV_num = 1/n_cv*norm((np.eye(n_cv)-A_GCV)@Z)**2 # numerator
    GCV_dom = (1/n_cv*np.trace(np.eye(n_cv)-A_GCV))**2 # denominator
    
    GCV_score = GCV_num/GCV_dom
    
    return GCV_score


def compute_mGCV(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the score for the modified generalized cross validation (mGCV) approach.
       Reference: Y.J. Kim, C. Gu, Smoothing spline Gaussian regression: More scalable computation via efficient approximation, J. Royal Statist. Soc. 66 (2004) 337–356.
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           mGCV score
    """
    
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # see (5) in [4]
    Z = np.concatenate((Z_re, Z_im), axis = 0)
    
    n_cv = Z.shape[0] # 2*number of frequencies
    
    A_agm = A.T@A + lambda_value*M # see (13) in [4]

    if (is_PD(A_agm)==False):
        A_agm = nearest_PD(A_agm)
    
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    A_GCV = A@inv_A_agm@A.T # see (13) in [4]
    
    # the stabilization parameter, rho, is computed as described by Kim et al.
    rho = 2 # see (15) in [4]
    
    # mGCV score ; see (14) in [4]
    mGCV_num = 1/n_cv*norm((np.eye(n_cv)-A_GCV)@Z)**2 # numerator
    mGCV_dom = ((1/n_cv)*(np.trace(np.eye(n_cv)-rho*A_GCV)))**2 # denominator
    mGCV_score = mGCV_num/mGCV_dom
    
    return mGCV_score


def compute_rGCV(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the score for the robust generalized cross-validation (rGCV) approach.
       Reference: M. A. Lukas, F. R. de Hoog, R. S. Anderssen, Practical use of robust GCV and modified GCV for spline smoothing, Comput. Statist. 31 (2016) 269–289.   
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           rGCV score    
    """
     
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # see (5) in [4]
    Z = np.concatenate((Z_re, Z_im), axis = 0)
    
    n_cv = Z.shape[0] # 2*number of frequencies
    
    A_agm = A.T@A + lambda_value*M # see (13) in [4]

    if (is_PD(A_agm)==False):
        A_agm = nearest_PD(A_agm)
    
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    A_GCV = A@inv_A_agm@A.T # see (13) in [4]
    
    # GCV score ; see (13) in [4]
    rGCV_num = 1/n_cv*norm((np.eye(n_cv)-A_GCV)@Z)**2
    rGCV_dom = ((1/n_cv)*(np.trace(np.eye(n_cv)-A_GCV)))**2
    rGCV = rGCV_num/rGCV_dom
    
    # the robust parameter, xsi, is computed as described in Lukas et al.
    xi = 0.3 # see (16) in [4]
    
    # mu_2 parameter ; see (16) in [4]
    mu_2 = (1/n_cv)*np.trace(A_GCV.T@A_GCV)
    
    # rGCV score ; see (16) in [4]
    rGCV_score = (xi + (1-xi)*mu_2)*rGCV
        
    return rGCV_score

def compute_re_im_cv(log_lambda, A_re, A_im, Z_re, Z_im, M):
    """
    This function computes the re-im score using CVXOPT to minimize the quadratic problem.
    Inputs:
        log_lambda: regularization parameter
        A_re: discretization matrix for the real part of the impedance
        A_im: discretization matrix for the imaginary part of the impedance
        Z_re: vector of the real parts of the impedance
        Z_im: vector of the imaginary parts of the impedance
        M: differentiation matrix
    Output:
        re-im score
    """

    lambda_value = exp(log_lambda)

    # Obtain H and c matrices for both real and imaginary part
    H_re, c_re = basics.quad_format_separate(A_re, Z_re, M, lambda_value)
    H_im, c_im = basics.quad_format_separate(A_im, Z_im, M, lambda_value)

    lb = np.zeros([Z_re.shape[0] + 1])  # + 1 if a resistor or an inductor is included in the DRT model
    bound_mat = np.eye(lb.shape[0])

    args_re = [cvxopt.matrix(H_re),cvxopt.matrix(c_re), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]
    args_im = [cvxopt.matrix(H_im), cvxopt.matrix(c_im), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]

    # Solve the quadratic programming problems
    sol_re = cvxopt.solvers.qp(*args_re)
    sol_im = cvxopt.solvers.qp(*args_im)

    if 'optimal' not in sol_re['status'] or 'optimal' not in sol_im['status']:
        return None
    # obtain gamma vector for real and imagianry parts of the impedance
    gamma_ridge_re = np.array(sol_re['x']).flatten()
    gamma_ridge_im = np.array(sol_im['x']).flatten()

    # Stacking the resistance R and inductance L on top of gamma_ridge_im and gamma_ridge_re, respectively
    gamma_ridge_re_cv = np.concatenate((np.array([0, gamma_ridge_re[1]]), gamma_ridge_im[2:]))
    gamma_ridge_im_cv = np.concatenate((np.array([gamma_ridge_im[0], 0]), gamma_ridge_re[2:]))
    
    # Re-im score; see (13) in [2] and (17) in [4]
    re_im_cv_score = norm(Z_re - A_re @ gamma_ridge_re_cv) ** 2 + norm(Z_im - A_im @ gamma_ridge_im_cv) ** 2

    return re_im_cv_score




def compute_kf_cv(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the k-fold (kf) score.
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           kf score
    """

    lambda_value = exp(log_lambda)
    
    # non-negativity constraint on the DRT gamma
    lb = np.zeros([Z_re.shape[0]+1])
    bound_mat = np.eye(lb.shape[0])
    
    # parameters for kf
    N_splits = 5 # N_splits=N_freq correspond to leave-one-out cross-validation
    random_state = 34054 + compute_kf_cv.counter*100  # change random state for each experiment
    kf = KFold(n_splits = N_splits, shuffle = True, random_state = random_state)                
    kf_cv = 0
    
    # train and test 
    for train_index, test_index in kf.split(Z_re):
        
        # step 1: preparation of the train and test sets
        print("TRAIN:", train_index, "TEST:", test_index)
        A_re_train, A_re_test = A_re[train_index,:], A_re[test_index,:]
        A_im_train, A_im_test = A_im[train_index,:], A_im[test_index,:]        
        Z_re_train, Z_re_test = Z_re[train_index], Z_re[test_index]
        Z_im_train, Z_im_test = Z_im[train_index], Z_im[test_index]
        
        # step 2: qudratic programming to obtain the DRT
        H_combined, c_combined = basics.quad_format_combined(A_re_train, A_im_train, Z_re_train, Z_im_train, M, lambda_value)
        args = [cvxopt.matrix(H_combined),cvxopt.matrix(c_combined), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]
        # Solve the quadratic programming problems
        sol = cvxopt.solvers.qp(*args)
        if 'optimal' not in sol['status']:
            return None
        
        #solve for gamma
        gamma_ridge = np.array(sol['x']).flatten()
        # step 3: update of the kf scores    
        kf_cv += 1/Z_re_test.shape[0]*(norm(Z_re_test-A_re_test@gamma_ridge)**2 + norm(Z_im_test-A_im_test@gamma_ridge)**2)
    
    # kf score ; see section 1.2 in the SI of [4]
    kf_cv_score = kf_cv/N_splits
    
    return kf_cv_score
compute_kf_cv.counter = 0


def compute_LC(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the score for L curve (LC)
       Reference: P.C. Hansen, D.P. O’Leary, The use of the L-curve in the regularization of discrete ill-posed problems, SIAM J. Sci. Comput. 14 (1993) 1487–1503.
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           LC score
    """
    
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # matrix A with A_re and A_im; # see (5) in [4]
    Z = np.concatenate((Z_re, Z_im), axis = 0) # stacked impedance
    
    # numerator eta_num of the first derivative of eta = log(||Z_exp - Ax||^2)
    A_agm = A.T@A + lambda_value*M # see (13) in [4]
    if (is_PD(A_agm)==False):
        A_agm = nearest_PD(A_agm)
           
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    A_LC = A@((inv_A_agm.T@inv_A_agm)@inv_A_agm)@A.T
    eta_num = Z.T@A_LC@Z 

    # denominator eta_denom of the first derivative of eta
    A_agm_d = A@A.T + lambda_value*np.eye(A.shape[0])
    if (is_PD(A_agm_d)==False):
        A_agm_d = nearest_PD(A_agm_d)
    
    L_agm_d = cholesky(A_agm_d) # Cholesky transform to inverse A_agm_d
    inv_L_agm_d = np.linalg.inv(L_agm_d)
    inv_A_agm_d = inv_L_agm_d.T@inv_L_agm_d
    eta_denom = lambda_value*Z.T@(inv_A_agm_d.T@inv_A_agm_d)@Z
    
    # derivative of eta
    eta_prime = eta_num/eta_denom
    
    # numerator theta_num of the first derivative of theta = log(lambda*||Lx||^2)
    theta_num  = eta_num
    
    # denominator theta_denom of the first derivative of theta
    A_LC_d = A@(inv_A_agm.T@inv_A_agm)@A.T
    theta_denom = Z.T@A_LC_d@Z
    
    # derivative of theta 
    theta_prime = -(theta_num)/theta_denom
    
    # numerator LC_num of the LC score in (19) in [4]
    a_sq = (eta_num/(eta_denom*theta_denom))**2
    p = (Z.T@(inv_A_agm_d.T@inv_A_agm_d)@Z)*theta_denom
    m = (2*lambda_value*Z.T@((inv_A_agm_d.T@inv_A_agm_d)@inv_A_agm_d)@Z)*theta_denom
    q = (2*lambda_value*Z.T@(inv_A_agm_d.T@inv_A_agm_d)@Z)*eta_num 
    LC_num = a_sq*(p+m-q)

    # denominator LC_denom of the LC score
    LC_denom = ((eta_prime)**2 + (theta_prime)**2)**(3/2)
    
    # LC score ; see (19) in [4]
    LC_score = LC_num/LC_denom
    
    return -LC_score 


def optimal_lambda(A_re, A_im, Z_re, Z_im, M, log_lambda_0, cv_type):
    
    """
       This function returns the regularization parameter given an initial guess and a regularization method. For constrained minimization, we use the scipy function sequential least squares programming (SLSQP).
       Inputs: 
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
           log_lambda0: initial guess for the regularization parameter
           cv_type: regularization method
       Output:
           optimized regularization parameter given the regularization method chosen
    """
    
    # credible for the lambda values
    bnds = [(log(10**-7),log(10**0))] 
    
    # GCV method
    if cv_type == 'GCV': 
        res = minimize(compute_GCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('GCV')
    
    # mGCV method
    elif cv_type == 'mGCV': 
        res = minimize(compute_mGCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('mGCV')
        
    # rGCV method
    elif cv_type == 'rGCV': 
        res = minimize(compute_rGCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('rGCV')  
    
    # L-curve method
    elif cv_type == 'LC':
        res = minimize(compute_LC, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('LC')
    
    # re-im discrepancy
    elif cv_type == 're-im':
        res = minimize(compute_re_im_cv, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('re-im')
        
    # k-fold 
    elif cv_type == 'kf':  
        res = minimize(compute_kf_cv, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('kf') 
    
    # custom value
    else:
        lambda_value = exp(log_lambda_0)
        print('custom')
        return lambda_value

    lambda_value = exp(res.x)

    return lambda_value

