# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:19:15 2024

@author: Jannis
"""

import numpy as np
from landscapes import *
from multi_species import MyMultiSpecies as msp
import pandas as pd
from scipy.stats import skew

def interroot(xl,xr,yl,yr):
    '''
    Compute the root of an affine function interpolating two points.

    Parameters
    ----------
    xl : float
        x-value on the left.
    xr : float
        x-value on the right.
    yl : float
        function value on the left.
    yr : float
        function value on the right.

    Returns
    -------
    float
        Root of linear interpolation.
    '''
    return xl-yl*(xr-xl)/(yr-yl)
# Landscape properties
N = 32 # number of nodes in x- and y-direction
length = 2*np.pi # length of the periodic square
W = complete_spacePeriodic(N, N, 2*np.pi) # Weight matrix for morans I

# Number of traits
n_traits  = 100
# Simulation parameters
d = 1e-1 # Diffusion
n_prey = 1 # Number of prey
a = 1.7 # Maximal attack rate
s = 0.1 # Trade-off parameter
gamma = 0.3 # Unused, conversion coefficient for predation if predator is modelled dynamically
eff = 0.5 # Effectiveness of crypsis
resp = 1.4 # Unused, respiration of predator if predator is modelled dynamically
g = 1.2 # Growth coefficient
params = resp,g,a,s,d,eff,gamma

# DataFrame for storing experiment values
df = pd.DataFrame(columns=["Landscape","Morans I","Variation","Skewness","Singular strategy","ESS","GED","Sympatric branching","Parapatric branching"])

# Setup ecological model
R = np.array([0.]) # initial trait
u0 = np.ones((N*N,1))*4.5 # initial prey
v0 = np.ones((N*N,1))  # predator
Res = 4.5*np.ones((N*N,1)) # Resource map
L = np.ones((N*N,1)) # Colour landscape
eco = msp(n_prey,N,length,L,Res,u0,v0,R,params,pred_constant=True) # Solver instance


for counter in range(15000):
    n_ess,n_ged, n_parap,n_sympatr = 0,0,0,0
    # Initial values for simulation
    eco.prey = u0
    eco.pred = v0
    
    L = perlin(N, 3, 9) # Set up landscape
    
    # Normalize landscape
    L-= np.min(L)
    L/= np.max(L)
    
    # Compute attributes
    variance = np.var(L)
    mi = morans_i(L,W)
    skw = skew(L.flatten())
    
    # Reshape into form for solver
    eco.L = L.reshape((N*N,1))
    
    # Slopes for storing selection gradients
    Slopes = []
    
    U = eco.find_eq(1e-4,10,maxiter=100,x0=eco.initial_guess_flat(0)) # Solve initially
    for r in np.linspace(0,1,n_traits):
        eco.R = np.array([r]) # Update trait value

        conv = eco.find_eq(1e-4,0)
        while not conv: # Catch if non-convergent
            conv = eco.find_eq(1e-4,np.random.uniform(low=0,high=10),maxiter=100)

        sl = eco.Grad(r,eco.prey) # Compute selection gradient
        Slopes.append(sl) # Append
        
        if r > 0:
            if Slopes[-1]*Slopes[-2] < 0: # Check for singular strategy
                r_star = interroot(r-1/(n_traits-1), r, Slopes[-2], Slopes[-1]) # Find root
                eco.R = np.array([r_star]) # Update trait value
                conv = eco.find_eq(1e-9,0)
                while not conv:
                    conv = eco.find_eq(1e-9,np.random.uniform(low=0,high=10),maxiter=100)
                    
                sl2 = eco.Grad2(r_star,eco.prey) # Second derivative
                
                r_left = r_star - 1/(n_traits-1)
                r_right = r_star + 1 /(n_traits-1)
                # Compute fitness left and right to approximate second derivative
                lam_center, x0 = eco.inv_fitness(r_star,eco.prey,100,1e-3)
                lam_left,vec = eco.inv_fitness(r_left, x0, 100, 1e-3)
                lam_right,vec = eco.inv_fitness(r_right,x0,100,1e-3)
                
                if (lam_left+lam_right-2*lam_center) <0: # Fitness maximum
        
                    if Slopes[-2] > 0:
                        n_ess += 1 
                    else :
                        n_ged +=1
                else: # fitness minimum
                    if sl2 > 0:
                        n_sympatr += 1
                    else : # 
                        n_parap  += 1
         
    df.loc[counter] = [L,mi,variance,skw,n_sympatr+n_parap+n_ged+n_ess,n_ess,n_ged,n_sympatr,n_parap] # Append experiment data to DataFrame

    
    
      
df.to_pickle("ExperimentA.pkl") # Save as .pkl file
