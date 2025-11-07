import numpy as np
from numpy import exp
from numpy.linalg import norm, cholesky
from numpy import inf, sin, cos, cosh, pi, exp, log10, log
from math import sqrt
from scipy import integrate
from scipy.optimize import fsolve, minimize, LinearConstraint, Bounds
from scipy.linalg import toeplitz, hankel
#import cvxopt

"""
this file store all the functions that are shared by all the three DRT method, i.e., simple, Bayesian, and BHT
"""

def g_i(freq_n, tau_m, epsilon, rbf_type):
    """
        this function generate the elements of A_re    
    """
    alpha = 2*pi*freq_n*tau_m  
    
    rbf_switch = {
                'gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0_matern': lambda x: exp(-abs(epsilon*x)),
                'C2_matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4_matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6_matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                'inverse_quadratic': lambda x: 1/(1+(epsilon*x)**2)
                }
    
    rbf = rbf_switch.get(rbf_type)
    integrand_g_i = lambda x: 1./(1.+(alpha**2)*exp(2.*x))*rbf(x)
    out_val = integrate.quad(integrand_g_i, -50, 50, epsabs=1E-9, epsrel=1E-9)
    
    return out_val[0]


def g_ii(freq_n, tau_m, epsilon, rbf_type):
    """
       this function generate the elements of A_im 
    """ 
    alpha = 2*pi*freq_n*tau_m  
    
    rbf_switch = {
                'gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0_matern': lambda x: exp(-abs(epsilon*x)),
                'C2_matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4_matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6_matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                'inverse_quadratic': lambda x: 1/(1+(epsilon*x)**2)
                }
    
    rbf = rbf_switch.get(rbf_type)
    integrand_g_ii = lambda x: alpha/(1./exp(x)+(alpha**2)*exp(x))*rbf(x)    
    out_val = integrate.quad(integrand_g_ii, -50, 50, epsabs=1E-9, epsrel=1E-9)
    
    return out_val[0]


def compute_epsilon(freq, coeff, rbf_type, shape_control): 
    """
        this function is used to compute epsilon, i.e., the shape factor of
        the rbf used for discretization. user can directly set the shape factor
        by selecting 'shape' for the shape_control. alternatively, 
        when 'FWHM_coeff' is selected, the shape factor is such that 
        the full width half maximum (FWHM) of the rbf equals to the average 
        relaxation time spacing in log space over coeff, i.e., FWHM = delta(ln tau)/coeff
    """ 
   
    N_freq = freq.shape[0]
    
    if rbf_type == 'pwl':
        return 0
    
    rbf_switch = {
                'gaussian': lambda x: exp(-(x)**2)-0.5,
                'C0_matern': lambda x: exp(-abs(x))-0.5,
                'C2_matern': lambda x: exp(-abs(x))*(1+abs(x))-0.5,
                'C4_matern': lambda x: 1/3*exp(-abs(x))*(3+3*abs(x)+abs(x)**2)-0.5,
                'C6_matern': lambda x: 1/15*exp(-abs(x))*(15+15*abs(x)+6*abs(x)**2+abs(x)**3)-0.5,
                'inverse_quadratic': lambda x: 1/(1+(x)**2)-0.5
                }

    rbf = rbf_switch.get(rbf_type)
    
    if shape_control == 'FWHM_coeff':
        #equivalent as the 'FWHM Coefficient' option in matlab code
        FWHM_coeff = 2*fsolve(rbf,1)
        delta = np.mean(np.diff(np.log(1/freq.reshape(N_freq))))
        epsilon = coeff*FWHM_coeff/delta
        
    else:
        #equivalent as the 'Shape Factor' option in matlab code
        epsilon = coeff
    
    return epsilon[0]
    

def inner_prod_rbf_1(freq_n, freq_m, epsilon, rbf_type):
    """  
        this function output the inner product of the first derivative of the
        rbf with respect to log(1/freq_n) and log(1/freq_m)
    """  
    a = epsilon*log(freq_n/freq_m)

    rbf_switch = {
                'gaussian': -epsilon*(-1+a**2)*exp(-(a**2/2))*sqrt(pi/2),
                'C0_matern': epsilon*(1-abs(a))*exp(-abs(a)),
                'C2_matern': epsilon/6*(3+3*abs(a)-abs(a)**3)*exp(-abs(a)),
                'C4_matern': epsilon/30*(105+105*abs(a)+30*abs(a)**2-5*abs(a)**3-5*abs(a)**4-abs(a)**5)*exp(-abs(a)),
                'C6_matern': epsilon/140*(10395 +10395*abs(a)+3780*abs(a)**2+315*abs(a)**3-210*abs(a)**4-84*abs(a)**5-14*abs(a)**6-abs(a)**7)*exp(-abs(a)),
                'inverse_quadratic': 4*epsilon*(4-3*a**2)*pi/((4+a**2)**3)
                }
    
    return rbf_switch.get(rbf_type)


def inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type):
    """  
        this function output the inner product of the second derivative of the
        rbf with respect to log(1/freq_n) and log(1/freq_m)
    """  
    a = epsilon*log(freq_n/freq_m)

    rbf_switch = {
                'gaussian': epsilon**3*(3-6*a**2+a**4)*exp(-(a**2/2))*sqrt(pi/2),
                'C0_matern': epsilon**3*(1+abs(a))*exp(-abs(a)),
                'C2_matern': epsilon**3/6*(3 +3*abs(a)-6*abs(a)**2+abs(a)**3)*exp(-abs(a)),
                'C4_matern': epsilon**3/30*(45 +45*abs(a)-15*abs(a)**3-5*abs(a)**4+abs(a)**5)*exp(-abs(a)),
                'C6_matern': epsilon**3/140*(2835 +2835*abs(a)+630*abs(a)**2-315*abs(a)**3-210*abs(a)**4-42*abs(a)**5+abs(a)**7)*exp(-abs(a)),
                'inverse_quadratic': 48*(16 +5*a**2*(-8 + a**2))*pi*epsilon**3/((4 + a**2)**5)
                }
    
    return rbf_switch.get(rbf_type)


def gamma_to_x(gamma_vec, tau_vec, epsilon, rbf_type): ## double check this to see if the function is correct
    """  
        this function map the gamma_vec back to the x vector
        for piecewise linear, x = gamma
    """  
    if rbf_type == 'pwl':
        x_vec = gamma_vec
        
    else:
        rbf_switch = {
                    'gaussian': lambda x: exp(-(epsilon*x)**2),
                    'C0_matern': lambda x: exp(-abs(epsilon*x)),
                    'C2_matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                    'C4_matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                    'C6_matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                    'inverse_quadratic': lambda x: 1/(1+(epsilon*x)**2)
                    }
        
        rbf = rbf_switch.get(rbf_type)
        
        N_taus = tau_vec.size
        B = np.zeros([N_taus, N_taus])
        
        for p in range(0, N_taus):
            for q in range(0, N_taus):
                delta_log_tau = log(tau_vec[p])-log(tau_vec[q])
                B[p,q] = rbf(delta_log_tau)
                
        B = 0.5*(B+B.T)
                
        x_vec = np.linalg.solve(B, gamma_vec)
            
    return x_vec


def x_to_gamma(x_vec, tau_map_vec, tau_vec, epsilon, rbf_type): ## double check this to see if the function is correct
    
    if rbf_type == 'pwl':
        gamma_vec = x_vec
        
    else:
        rbf_switch = {
                    'gaussian': lambda x: exp(-(epsilon*x)**2),
                    'C0_matern': lambda x: exp(-abs(epsilon*x)),
                    'C2_matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                    'C4_matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                    'C6_matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                    'inverse_quadratic': lambda x: 1/(1+(epsilon*x)**2)
                    }
        
        rbf = rbf_switch.get(rbf_type)
        
        N_taus = tau_vec.size
        N_tau_map = tau_map_vec.size
        gamma_vec = np.zeros([N_tau_map, 1])
#        rbf_vec = np.zeros([N_taus,1])
        B = np.zeros([N_tau_map, N_taus])
        
        for p in range(0, N_tau_map):
            for q in range(0, N_taus):
                delta_log_tau = log(tau_map_vec[p])-log(tau_vec[q])
                B[p,q] = rbf(delta_log_tau)              
#        B = 0.5*(B+B.T)              
        gamma_vec = B@x_vec
        
        
    return gamma_vec



# This function is to check A_re_assemble
def A_re_fct(freq_vec, tau_vec):
    
    omega_vec = 2.*pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

    out_A_re = np.zeros((N_freqs, N_taus+1))
    out_A_re[:,0] = 1.
    
    for p in range(0, N_freqs):
        for q in range(0, N_taus):

            if q == 0:
                out_A_re[p, q+1] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
            elif q == N_taus-1:
                out_A_re[p, q+1] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
            else:
                out_A_re[p, q+1] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])
 
    return out_A_re

# This function is to check A_im_assemble
def A_im_fct(freq_vec, tau_vec):
    
    omega_vec = 2.*pi*freq_vec

    N_taus = tau_vec.size
    N_freqs = freq_vec.size

    out_A_im = np.zeros((N_freqs, N_taus+1))
    out_A_im[:,0] = omega_vec
    
    for p in range(0, N_freqs):
        for q in range(0, N_taus):
            if q == 0:
                out_A_im[p, q+1] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
            elif q == N_taus-1:
                out_A_im[p, q+1] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
            else:
                out_A_im[p, q+1] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])

    return out_A_im

def assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type):
    """
        this function assemble the A_re matrix
    """    
#   compute number of frequency, tau and omega
    omega_vec = 2.*pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

#   define the A_re output matrix
    out_A_re_temp = np.zeros((N_freqs, N_taus))
    out_A_re = np.zeros((N_freqs, N_taus+2))
    
#   check if the frequencies are sufficiently log spaced
    std_diff_freq = np.std(np.diff(np.log(1/freq_vec)))
    mean_diff_freq = np.mean(np.diff(np.log(1/freq_vec)))

#   check if the frequencies are sufficiently log spaced and that N_freqs = N_taus
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01 and N_freqs == N_taus 

    if toeplitz_trick and rbf_type != 'pwl':
        # use toeplitz trick
        R = np.zeros(N_taus)
        C = np.zeros(N_freqs)
        
        for p in range(0, N_freqs):
            
            C[p] = g_i(freq_vec[p], tau_vec[0], epsilon, rbf_type)
        
        for q in range(0, N_taus):
            
            R[q] = g_i(freq_vec[0], tau_vec[q], epsilon, rbf_type)        
                        
        out_A_re_temp = toeplitz(C,R) 

    else:
        # use brute force
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
            
                if rbf_type == 'pwl':
                    if q == 0:
                        out_A_re_temp[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                    elif q == N_taus-1:
                        out_A_re_temp[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_re_temp[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])                    
                
                else:
                    out_A_re_temp[p, q]= g_i(freq_vec[p], tau_vec[q], epsilon, rbf_type)
        
    out_A_re[:,2:] = out_A_re_temp

    return out_A_re




def assemble_A_im(freq_vec, tau_vec, epsilon, rbf_type):
    """
        This function assemble the A_im matrix
    """        
#   compute number of frequency, tau and omega
    omega_vec = 2.*pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

#   define the A_re output matrix
    out_A_im_temp = np.zeros((N_freqs, N_taus))
    out_A_im = np.zeros((N_freqs, N_taus+2))
    
#   check if the frequencies are sufficiently log spaced
    std_diff_freq = np.std(np.diff(np.log(1/freq_vec)))
    mean_diff_freq = np.mean(np.diff(np.log(1/freq_vec)))

#   check if the frequencies are sufficiently log spaced and that N_freqs = N_taus
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01 and N_freqs == N_taus 
    
    if toeplitz_trick and rbf_type != 'pwl':
        # use toeplitz trick
        R = np.zeros(N_taus)
        C = np.zeros(N_freqs)
        
        for p in range(0, N_freqs):
            
            C[p] = - g_ii(freq_vec[p], tau_vec[0], epsilon, rbf_type)
        
        for q in range(0, N_taus):
            
            R[q] = - g_ii(freq_vec[0], tau_vec[q], epsilon, rbf_type)        
                        
        out_A_im_temp = toeplitz(C,R) 

    else:
        # use brute force
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
            
                if rbf_type == 'pwl':                
                    if q == 0:
                        out_A_im_temp[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                        
                    elif q == N_taus-1:
                        out_A_im_temp[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_im_temp[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])                    
                
                else:
                    out_A_im_temp[p, q]= - g_ii(freq_vec[p], tau_vec[q], epsilon, rbf_type)
        
    out_A_im[:,2:] = out_A_im_temp    
    
    return out_A_im


def assemble_M_1(tau_vec, epsilon, rbf_type):
    """
        this function assembles the M matrix which contains the 
        the inner products of 1st-derivative of the discretization rbfs
        size of M matrix depends on the number of collocation points, i.e. tau vector
    """
    freq_vec = 1/tau_vec   
    # first get number of collocation points
    N_taus = tau_vec.size
    N_freq = freq_vec.size
    # define the M output matrix
    out_M_temp = np.zeros([N_taus, N_taus])
    out_M = np.zeros([N_taus+2, N_taus+2])
    
    #check if the collocation points are sufficiently log spaced
    std_diff_freq = np.std(np.diff(np.log(tau_vec)));
    mean_diff_freq = np.mean(np.diff(np.log(tau_vec)));
    
    #If they are we apply the toeplitz trick   
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01
    
    if toeplitz_trick and rbf_type != 'pwl':
        #Apply the toeplitz trick to compute the M matrix 
        R = np.zeros(N_taus)
        C = np.zeros(N_taus)
        
        for n in range(0,N_taus):
            C[n] = inner_prod_rbf_1(freq_vec[0], freq_vec[n], epsilon, rbf_type)# may be use tau instead of freq
            
        for m in range(0,N_taus):
            R[m] = inner_prod_rbf_1(freq_vec[m], freq_vec[0], epsilon, rbf_type)    
        
        out_M_temp = toeplitz(C,R) 
         
    elif rbf_type == 'pwl':
        #If piecewise linear discretization
        out_L_temp = np.zeros([N_freq-1, N_freq])
        
        for iter_freq_n in range(0,N_freq-1):
            delta_loc = log((1/freq_vec[iter_freq_n+1])/(1/freq_vec[iter_freq_n]))
            out_L_temp[iter_freq_n,iter_freq_n] = -1/delta_loc
            out_L_temp[iter_freq_n,iter_freq_n+1] = 1/delta_loc

        out_M_temp = out_L_temp.T@out_L_temp
    
    else:
        #compute rbf with brute force
        for n in range(0, N_taus):
            for m in range(0, N_taus):            
                out_M_temp[n,m] = inner_prod_rbf_1(freq_vec[n], freq_vec[m], epsilon, rbf_type)
        
    out_M[2:,2:] = out_M_temp
    
    return out_M


def assemble_M_2(tau_vec, epsilon, rbf_type):
    """
        this function assembles the M matrix which contains the 
        the inner products of 2nd-derivative of the discretization rbfs
        size of M matrix depends on the number of collocation points, i.e. tau vector
    """ 
    freq_vec = 1/tau_vec            
    # first get number of collocation points
    N_freqs = freq_vec.size
    N_taus = tau_vec.size
    
    # define the M output matrix
    out_M_temp = np.zeros([N_taus, N_taus])
    out_M = np.zeros([N_taus+2, N_taus+2])
    
    #check if the collocation points are sufficiently log spaced
    std_diff_freq = np.std(np.diff(np.log(tau_vec)));
    mean_diff_freq = np.mean(np.diff(np.log(tau_vec)));
    
    #If they are we apply the toeplitz trick   
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01
    
    if toeplitz_trick and rbf_type != 'pwl':
        #Apply the toeplitz trick to compute the M matrix 
        R = np.zeros(N_taus)
        C = np.zeros(N_taus)
        
        for n in range(0,N_taus):
            C[n] = inner_prod_rbf_2(freq_vec[0], freq_vec[n], epsilon, rbf_type)# later, we shall use tau instead of freq
            
        for m in range(0,N_taus):
            R[m] = inner_prod_rbf_2(freq_vec[m], freq_vec[0], epsilon, rbf_type)# later, we shall use tau instead of freq
        
        out_M_temp = toeplitz(C,R) 
         
    elif rbf_type == 'pwl':
        #Piecewise linear discretization
        out_L_temp = np.zeros((N_taus-2, N_taus))
    
        for p in range(0, N_taus-2):
            delta_loc = log(tau_vec[p+1]/tau_vec[p])
            
            if p == 0 or p == N_taus-3:
                out_L_temp[p,p] = 2./(delta_loc**2)
                out_L_temp[p,p+1] = -4./(delta_loc**2)
                out_L_temp[p,p+2] = 2./(delta_loc**2)
            else:
                out_L_temp[p,p] = 1./(delta_loc**2)
                out_L_temp[p,p+1] = -2./(delta_loc**2)
                out_L_temp[p,p+2] = 1./(delta_loc**2)
                
        out_M_temp = out_L_temp.T@out_L_temp
    
    else:
        #compute rbf with brute force
        for n in range(0, N_taus):
            for m in range(0, N_taus):            
                out_M_temp[n,m] = inner_prod_rbf_2(freq_vec[n], freq_vec[m], epsilon, rbf_type)
        
    out_M[2:,2:] = out_M_temp
    
    return out_M


def quad_format(A,b,M,lambda_value):
    """
        this function reformats the DRT regression 
        as a quadratic program - this uses either re or im
    """
    H = 2*(A.T@A+lambda_value*M)
    H = (H.T+H)/2
    c = -2*b.T@A
    
    return H,c


def quad_format_combined(A_re,A_im,b_re,b_im,M,lambda_value): 
    """
        this function reformats the DRT regression 
        as a quadratic program - this uses both re and im
    """
    H = 2*((A_re.T@A_re+A_im.T@A_im)+lambda_value*M)
    H = (H.T+H)/2
    c = -2*(b_im.T@A_im+b_re.T@A_re)

    return H,c


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
        this function formats numpy matrix to cvxopt matrix 
        it then conduct the quadratic programming with cvxopt and
        output the optimum in numpy array format
    """
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        
    cvxopt.solvers.options['abstol'] = 1e-15
    cvxopt.solvers.options['reltol'] = 1e-15
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    
    return np.array(sol['x']).reshape((P.shape[1],))


def pretty_plot(width=8, height=None, plt=None, dpi=None,
                color_cycle=("qualitative", "Set1_9")):
    """
    This code is bollowed from pymatgen to produce high quality figures. This needs further polishing later
    Args:
        width (float): Width of plot in inches. Defaults to 8in.
        height (float): Height of plot in inches. Defaults to width * golden
            ratio.
        plt (matplotlib.pyplot): If plt is supplied, changes will be made to an
            existing plot. Otherwise, a new plot will be created.
        dpi (int): Sets dot per inch for figure. Defaults to 300.
        color_cycle (tuple): Set the color cycle for new plots to one of the
            color sets in palettable. Defaults to a qualitative Set1_9.
    Returns:
        Matplotlib plot object with properly sized fonts.
    """
    ticksize = int(width * 2.5)

    golden_ratio = (sqrt(5) - 1) / 2

    if not height:
        height = int(width * golden_ratio)

    if plt is None:
        import matplotlib.pyplot as plt
        import importlib
        mod = importlib.import_module("palettable.colorbrewer.%s" %
                                      color_cycle[0])
        colors = getattr(mod, color_cycle[1]).mpl_colors
#        from cycler import cycler

        plt.figure(figsize=(width, height), facecolor="w", dpi=dpi)
        ax = plt.gca()
#        ax.set_prop_cycle(cycler('color', colors))
    else:
        
        fig = plt.gcf()
        fig.set_size_inches(width, height)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    ax = plt.gca()
    ax.set_title(ax.get_title(), size=width * 4)

    labelsize = int(width * 3)

    ax.set_xlabel(ax.get_xlabel(), size=labelsize)
    ax.set_ylabel(ax.get_ylabel(), size=labelsize)

    return plt


def gauss_fct_ref (x,p): return p[0]*exp(-(log(x)-p[1])**2/(2*p[2]))


def gauss_fct(tau_vec, p):
    
    gamma = np.zeros_like(tau_vec)
    for i in range(0, len(p), 3):
        R_0 = exp(p[i])
        tau_0 = exp(p[i+1])
        sigma = exp(p[i+2])
        gamma = gamma + R_0 * np.exp(-(log(tau_vec) - log(tau_0))**2/(2*sigma)**2)
        
    return gamma

