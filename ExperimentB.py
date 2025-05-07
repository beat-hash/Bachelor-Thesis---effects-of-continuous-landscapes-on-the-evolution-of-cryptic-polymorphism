# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:18:57 2025

@author: Jannis
"""

import numpy as np
import matplotlib.pyplot as plt
from multi_species import MyMultiSpecies as msp
import pandas as pd
import seaborn as sns


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

# Load landscapes
df_load = pd.read_pickle("ExperimentA.pkl")
row= df_load[df_load['Singular strategy'] == 3].sample() # Choose a row, change to 3 if you want a succesfull landscape

sns.set_theme(style='whitegrid')
palette = sns.color_palette('mako',as_cmap=False)

# Setup ecological model
R = np.array([0.]) # initial trait
u0 = np.ones((N*N,1))*4.5 # initial prey
v0 = np.ones((N*N,1))  # predator
Res = 4.5*np.ones((N*N,1)) # Resource map
L = row.Landscape.item().reshape((N*N,1)) # Colour landscape
eco = msp(n_prey,N,length,L,Res,u0,v0,R,params,pred_constant=True) # Solver instance


# Vary d
Ds = np.exp(np.linspace(np.log(1e-2),0,30))
arr = np.zeros((n_traits,len(Ds))) # Store the slopes

ESS, Parap, Sympatr = np.empty((2,1)),np.empty((2,1)),np.empty((2,1)) # Store the position of singular strategies in trait-eco space

counter = 0
for d in Ds:
    eco.d = d # Update diffusion coefficient
    
    # Slopes for storing selection gradients
    Slopes = []
    
    U = eco.find_eq(1e-4,10,maxiter=100,x0=eco.initial_guess_flat(0)) # Solve initially
    for r in np.linspace(0,1,n_traits):
        eco.R = np.array([r]) # Update trait value

        conv = eco.find_eq(1e-4,0)
        while not conv: # Catch if non-convergent
            conv = eco.find_eq(1e-4,np.random.uniform(low=0,high=10),maxiter=100)

        sl = eco.Grad(r,eco.prey) # Compute selection gradient
        Slopes.append(sl[0]) # Append
        
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
                
                if (lam_left+lam_right-2*lam_center) <0:
                    
                    if Slopes[-2] > 0:
                        ESS = np.append(ESS,[[d],[r_star]],axis=1)
             
                    else :
          
                        None # GEDs not relevant
                else: 
                    if sl2 > 0:
      
                        Sympatr = np.append(Sympatr,[[d],[r_star]],axis=1)
                    else :
                
                        Parap = np.append(Parap,[[d],[r_star]],axis=1)
    arr[:,counter] = Slopes 
    counter+= 1

# Append to big list
arr_list = [arr.copy()]
ESS_list = [ESS.copy()]
Parap_list = [Parap.copy()]
Sympatr_list = [Sympatr.copy()]
# Vary attack rate
As = np.linspace(0.1,3,24)
eco.d = 1e-1 # Reset diffusion coefficient
counter = 0
arr = np.zeros((n_traits,len(As)))
ESS, Parap, Sympatr = np.empty((2,1)),np.empty((2,1)),np.empty((2,1))

for a in As:
    eco.a = a # Update attack rate
    # Slopes for storing selection gradients
    Slopes = []
    
    U = eco.find_eq(1e-4,10,maxiter=100,x0=eco.initial_guess_flat(0)) # Solve initially
    for r in np.linspace(0,1,n_traits):
        eco.R = np.array([r]) # Update trait value

        conv = eco.find_eq(1e-4,0)
        while not conv: # Catch if non-convergent
            conv = eco.find_eq(1e-4,np.random.uniform(low=0,high=10),maxiter=100)

        sl = eco.Grad(r,eco.prey) # Compute selection gradient
        Slopes.append(sl[0]) # Append
        
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
                if (lam_left+lam_right-2*lam_center) <0:
                    
                    if Slopes[-2] > 0:
                        ESS = np.append(ESS,[[a],[r_star]],axis=1)

                    else :
                        None 
                else: 
                    if sl2 > 0:
                        Sympatr = np.append(Sympatr,[[a],[r_star]],axis=1)

                    else :
                        Parap = np.append(Parap,[[a],[r_star]],axis=1)

    arr[:,counter] = Slopes 

    counter+= 1

arr_list.append(arr), ESS_list.append(ESS),Parap_list.append(Parap),Sympatr_list.append(Sympatr) # Append to big list

Ss = np.linspace(0.05,0.4,24) # Vary trade-off parameter
arr = np.zeros((n_traits,len(Ss)))
eco.a = 1.7 # Reset attack rate
counter = 0
ESS, Parap, Sympatr = np.empty((2,1)),np.empty((2,1)),np.empty((2,1))

for s in Ss:
    eco.s = s # Update trade-off parameter

    # Slopes for storing selection gradients
    Slopes = []
    
    U = eco.find_eq(1e-4,10,maxiter=100,x0=eco.initial_guess_flat(0)) # Solve initially
    for r in np.linspace(0,1,n_traits):
        eco.R = np.array([r]) # Update trait value

        conv = eco.find_eq(1e-4,0)
        while not conv: # Catch if non-convergent
            conv = eco.find_eq(1e-4,np.random.uniform(low=0,high=10),maxiter=100)

        sl = eco.Grad(r,eco.prey) # Compute selection gradient
        Slopes.append(sl[0]) # Append
        
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
                
                if (lam_left+lam_right-2*lam_center) <0:
                    
                    if Slopes[-2] > 0:
                        ESS = np.append(ESS,[[s],[r_star]],axis=1)

                    else :
                        None
                else: 
                    if sl2 > 0:
                        Sympatr = np.append(Sympatr,[[s],[r_star]],axis=1)

                    else :
                        Parap = np.append(Parap,[[s],[r_star]],axis=1)

    arr[:,counter] = Slopes 

    counter+= 1


arr_list.append(arr), ESS_list.append(ESS),Parap_list.append(Parap),Sympatr_list.append(Sympatr) # append to lists 

Os = np.linspace(0,2,24) # Vary oppcost
eco.s = 0.1 # Reset trade-off parameter
arr = np.zeros((n_traits,len(Os)))
counter = 0
ESS, Parap, Sympatr = np.empty((2,1)),np.empty((2,1)),np.empty((2,1))
for o in Os:
    Res_shift = 4 + o*L + (1-o)*0.5
    eco.Res = Res_shift # Update resource map
    
    # Slopes for storing selection gradients
    Slopes = []
    
    U = eco.find_eq(1e-4,10,maxiter=100,x0=eco.initial_guess_flat(0)) # Solve initially
    for r in np.linspace(0,1,n_traits):
        eco.R = np.array([r]) # Update trait value

        conv = eco.find_eq(1e-4,0)
        while not conv: # Catch if non-convergent
            conv = eco.find_eq(1e-4,np.random.uniform(low=0,high=10),maxiter=100)

        sl = eco.Grad(r,eco.prey) # Compute selection gradient
        Slopes.append(sl[0]) # Append
        
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
                
                if (lam_left+lam_right-2*lam_center) <0:
                    
                    if Slopes[-2] > 0:
                        ESS = np.append(ESS,[[o],[r_star]],axis=1)

                    else :
                        None
                else: 
                    if sl2 > 0:
                        Sympatr = np.append(Sympatr,[[o],[r_star]],axis=1)

                    else :
                        Parap = np.append(Parap,[[o],[r_star]],axis=1)

    arr[:,counter] = Slopes 

    counter+= 1


arr_list.append(arr), ESS_list.append(ESS),Parap_list.append(Parap),Sympatr_list.append(Sympatr) # Append to lists

# Make the figure
fig, axs = plt.subplots(ncols=4,nrows=1,sharey=True,figsize=(12,3))
xd,yd = np.meshgrid(Ds,np.linspace(0,1,n_traits))
xa,ya = np.meshgrid(As,np.linspace(0,1,n_traits))
xs,ys = np.meshgrid(Ss,np.linspace(0,1,n_traits))
xo,yo = np.meshgrid(Os,np.linspace(0,1,n_traits))
x,y = [xd,xa,xs,xo],[yd,ya,ys,yo]
for i in range(4):
    y_direc = arr_list[i][::15,::3] / np.max(np.abs(arr_list[i][::15,::3]))
    axs[i].quiver(x[i][::15,::3],y[i][::15,::3],np.zeros(np.shape(arr_list[i]))[::15,::3],y_direc/np.abs(y_direc),alpha=np.abs(y_direc))
    axs[i].contour(x[i],y[i],arr_list[i],[0])
    axs[i].scatter(Sympatr_list[i][0,1:],Sympatr_list[i][1,1:],c=palette[0],label="Sympatric branching",s=50,marker='s')
    axs[i].scatter(Parap_list[i][0,1:],Parap_list[i][1,1:],c=palette[2],label="Parapatric branching",s=50,marker='D')
    axs[i].scatter(ESS_list[i][0,1:],ESS_list[i][1,1:],c=palette[4],label="ESS",s=100,marker='*')
    axs[i].set_xlim(x[i][0,0],x[i][0,-1])
    axs[i].set_ylim(0,1)
    
axs[0].set_xscale('log')
axs[0].set_ylabel("Trait value")
axs[2].legend(loc='upper center',bbox_to_anchor=(-0.15,1.15),ncols=4)
axs[0].set_xlabel("Diffusion coefficient")
axs[1].set_xlabel("Attack rate")
axs[2].set_xlabel("Trade-off")
axs[3].set_xlabel("Opportunity cost")
        
