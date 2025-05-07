# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:19:15 2024

@author: Jannis
"""

import numpy as np
import matplotlib.pyplot as plt
from landscapes import weights,regrid
from multi_species import MyMultiSpecies as msp
import pandas as pd
'''
Here, we visualize the densities of resident prey at branching points and 
the fitness in space of mutants on both sides of the resident trait.
'''

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
w = weights(32, 100) # weights for rescaling to higher resolution for visibility

# Number of traits
n_traits  = 100
# Simulation parameters
d = 1e-1
n_prey = 1
a = 1.7 #1
s = 0.1
gamma = 0.3
eff = 0.5
resp = 1.4
g = 1.2
params = resp,g,a,s,d,eff,gamma

# Load experiment results
df = pd.read_pickle("ExperimentAstar.pkl")
df['SkewMorans'] = df['Skewness']*df['Morans I']
df_success = df[df['Parapatric branching']>=2] # Choose landscape from this one

R = np.array([0.]) # initial trait
u0 = np.ones((N*N,1))*4.5 # initial prey
v0 = np.ones((N*N,1))  # predator


# Choose landscape and ressource map
row = df_success.sample()
L = row.Landscape.item().reshape((32*32,1))
o = 4
Res =  o*(L-np.mean(L)+4.5) + (1-o)*4.5 # ressource map (with opportunity cost)

Slopes = [] # Store selection gradients
eco = msp(n_prey,N,length,L,Res,u0,v0,R,params,pred_constant=True)

# Store the resident densities and fitness in the landscape
Resident = []
Mutant_left = []
Mutant_right = []
Eigenvals = []
for r in np.linspace(0,1,n_traits):
    eco.R = np.array([r]) # update trait value

    conv = eco.find_eq(1e-4,0) # compute steady state
    while not conv: # Catch if not convergant
        conv = eco.find_eq(1e-4,np.random.uniform(low=0,high=10),x0=eco.initial_guess_flat(0),maxiter=100,update=True)

    sl = eco.Grad(r,eco.prey) # compute selection gradient

    Slopes.append(sl) 
    
    if r > 0:
        if Slopes[-1]*Slopes[-2] < 0:
            r_star = interroot(r-1/(n_traits-1), r, Slopes[-2], Slopes[-1]) # find root
            eco.R = np.array([r_star]) # update trait value
            conv = eco.find_eq(1e-9,0,x0=eco.initial_guess_flat(0),update=True,maxiter=60) # compute steady state
            while not conv: # Catch if not convergant
                conv = eco.find_eq(1e-9,np.random.uniform(low=0,high=10),x0=eco.initial_guess_flat(0),maxiter=100,update=True)
                
            r_left = r_star - 1e-2 # Trait mutant left
            r_right = r_star + 1e-2 # Trait mutant right
            lam_left,eig_left = eco.inv_fitness(r_left, eco.prey, 100, 1e-3) # Eigenvector left
            lam_right,eig_right = eco.inv_fitness(r_right,eco.prey,100,1e-3) # eigenvector right
            
            
            # Gather the arrays
            prey_density = regrid(eco.prey.reshape((32,32)), w)/Res.max()
            mutant_left = regrid(eig_left.reshape((32,32)), w)/np.max(eig_left)*lam_left*1e3 # Scaling
            mutant_right = regrid(eig_right.reshape((32,32)), w)/np.max(eig_right)*lam_right*1e3 # Scaling
            # Append 
            Resident.append(prey_density)
            Mutant_left.append(mutant_left)
            Mutant_right.append(mutant_right)
            Eigenvals.append(lam_left)
            Eigenvals.append(lam_right)

            
# Graphic 
fig,axs = plt.subplots(3,3,sharey=True,sharex=True,figsize=(12,6))
fig.subplots_adjust(wspace=0.3, hspace=0.3)

for i in range(3):
    pcm = axs[i,0].pcolormesh(Resident[i],vmin=0.3,vmax=0.85,cmap='summer')
    cf_lm = axs[i,1].contourf(Mutant_left[i])
    cf_rm = axs[i,2].contourf(Mutant_right[i])
    axs[i,0].set_yticks([])
    axs[-1,i].set_xticks([])
    cbar_rm = fig.colorbar(cf_lm,ax=axs[i,2],location='right')
    cbar_lm = fig.colorbar(cf_rm,ax=axs[i,1],location='right')
cbar = fig.colorbar(pcm,ax=axs[:,0],location='left',label="Resident density over maximal carrying capacity")

axs[0,0].annotate("",xy=(1.15,0.85),xytext=(1.15,-2.45),xycoords='axes fraction',arrowprops=dict(
    arrowstyle='-', 
    lw=2,                  # Line width
    color='black'          # Arrow color
))


axs[0,0].set_title(f"Resident densities \n at branching points",wrap=True)
axs[0,1].set_title(f"Fitness of mutants \n on the left",wrap=True)
axs[0,2].set_title(f"Fitness of mutants \n on the right",wrap=True)


          