# -*- coding: utf-8 -*-
"""
Two-patch model 
"""


import numpy as np
import matplotlib.pyplot as plt
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


def alpha(r,L,s,eff):
    return (1-eff*np.exp(-(r-L)**2/s**2))

def alphaprime(r,L,s,eff):
    return 2*(r-L)/s**2*eff*np.exp(-(r-L)**2/s**2)

def w1(x,r,params,Holling = 1):
    '''
    Describers non-linear growth and predation in first patch.

    Parameters
    ----------
    x : float
        Patch population density.
    r : float
        Population trait value.
    params : tuple
        Parameters.

    Returns
    -------
    float
        Growth in the patch.

    '''
    g,K,s,P,ratio = params
    # Return values for different Holling functionals
    if Holling == 1:
        return ratio*g*(1-x/K/ratio) - alpha(r,1,s,0.5)*P # Holling type I
    elif Holling == 2:
        return (g*ratio)*(1-x/K) - (1-np.exp(-(1-r)**2/s**2))*P/(1+x**2) # Holling type II
    else :
        return (g*ratio)*(1-x/K) - (1-np.exp(-(1-r)**2/s**2))*P*x/(1+x**2) # Holling type III
    

def w2(x,r,params,Holling = 1):
    '''
    Describers non-linear growth and predation in second patch.

    Parameters
    ----------
    x : float
        Patch population density.
    r : float
        Population trait value.
    params : tuple
        Parameters.

    Returns
    -------
    float
        Growth in the patch.

    '''
    g,K,s,P,ratio = params
    # Return values for different Holling functionals

    if Holling == 1:
        return g*(1-x/K*ratio)/ratio - alpha(r,0,s,0.5)*P # Holling type I
    elif Holling == 2:
        return g*(1-x/K*ratio)/ratio - (1-np.exp(-(r)**2/s**2))*P/(1+x**2) # Holling type II
    else :
        return g*(1-x/K*ratio)/ratio - (1-np.exp(-(r)**2/s**2))*P*x/(1+x**2) # Holling type III
    

def lam(u,r,params,D,Holling = 1):
    '''
    Returns dominant eigenvalue of the systems matrix.

    Parameters
    ----------
    u : 1D Array
        State of the system.
    r : float
        Trait value of the population.
    params : tuple
        Parameters.
    D : float
        Diffusion constant.

    Returns
    -------
    float
        Dominant eigenvalue.

    '''
    x = u[0]
    y = u[1]
    w_1 = w1(x,r,params,Holling=Holling)
    w_2 = w2(y,r,params,Holling=Holling)
    
    return (w_1+w_2)/2 + np.sqrt( D**2 + (w_1-w_2)**2/4) - D

def f(u,params,r,D,Holling=1):
    '''
    Right hand side of the ODE system.

    Parameters
    ----------
    u : 1D Array
        State of the system.
    r : float
        Trait value of the population.
    params : tuple
        Parameters.
    D : float
        Diffusion constant.

    Returns
    -------
    1D Array
        Output of the right hand side.

    '''
    x = u[0]
    y = u[1]
    return np.array([(w1(x,r,params,Holling=Holling)-D)*x + D*y,(w2(y,r,params,Holling=Holling)-D)*y + D*x ])
    
def Jacobian(f,x0):
    '''
    Numerical Jacobian.

    Parameters
    ----------
    f : Callable
        Function of which Jacobian is to be computed.
    x0 : 1D Array
        Point at which Jacobian is to be computed.

    Returns
    -------
    NabF : 2D Array
        Jacobian.

    '''
    f0 = f(x0)
    
    NabF = np.zeros((len(f0),len(x0)))
    for i in range(len(x0)):
        x1 = x0.copy()
        x1[i] = x1[i]-1e-9
        NabF[:,i] = (f0-f(x1))/1e-9
    
    return NabF

def Df(u,params,r,D):
    '''
    Jacobian of the rhs of our two-patch model.

    Parameters
    ----------
    u : 1D Array
        State of the system.
    r : float
        Trait value of the population.
    params : tuple
        Parameters.
    D : float
        Diffusion constant.


    Returns
    -------
    2D Array
        Jacobian.

    '''
    return Jacobian(lambda x : f(x,params,r,D),u)


def Newton(f,Df,x0,maxiter,alpha,tol):
    '''
    Newton method for small nonlinear systems.

    Parameters
    ----------
    f : Callable
        Function to be set to zero.
    Df : Callable
        Jacobian of f.
    x0 : 1D Array
        Initial guess.
    maxiter : int
        Maximum iterations.
    alpha : float
        Stepsize ofr iteration.
    tol : float
        Desired accuracy.

    Returns
    -------
    1D array
        Approximation to root of f.

    '''
    x_old = x0
    for i in range(maxiter):
        fx = f(x_old)
        if np.linalg.norm(fx) < tol:
            return x_old
        Dfx = Df(x_old)
        
        delta_x = np.linalg.solve(Dfx,-fx)

        x_new = x_old + delta_x*alpha
        
        x_old = x_new 
    
    return x_new


# Colourmap for plots
palette = sns.color_palette('mako',as_cmap=False)
# Arrays with parameters to vary
acc = 100  # Number of varied parameters
Preds = np.linspace(0.5,2,acc)
Ss = np.linspace(0.2,1,acc)
Ds = np.linspace(0.4,2.4,acc)
Rs = np.linspace(0,.999,100)
Ratios = np.linspace(.9,1.1,acc)


counterx = 0 # Counter for values stored on the horizontal axis
sol = np.ones(2)*10 # Initial guess to Newton method

# Arrays for creating the plots
biomass = np.zeros((100,acc)) # Biomass of the resident population
arr = np.zeros((100,acc)) # Will contain the slopes of the invasion fitness
arr2 = np.zeros((100,acc))
# Not-varied parameters
P = 1.5
ratio = 1
g = 1
K = 1
s = .8
D = 1
ess,bra = [[100,100]],[[100,100]]
for D in Ds: # Select here what you want to vary
    
    countery = 0 # Counter ofr vertical array-elemtns
    for r in np.linspace(0,1,100):
        params = g,K,s,P,np.sqrt(ratio)
        sol = Newton(lambda u : f(u,params,r,D,Holling=3),lambda u : Df(u,params,r,D),sol,50,1,1e-12)
        laml = lam(sol,r-1e-4,params,D) #Invasion fitness left of the resident
        lamr = lam(sol,r+1e-4,params,D) # Invasion fitness right of the resident
        grad = (lamr-laml)/1e-4/2 # Slope of the invasion fitness
        arr[countery,counterx] = grad
        if r > 0:
            if arr[countery-1,counterx]*arr[countery,counterx] < 0:
                r_star = interroot( r-1/99, r,arr[countery-1,counterx], arr[countery,counterx])
                laml = lam(sol,r_star-1e-4,params,D) #Invasion fitness left of the resident
                lamr = lam(sol,r_star+1e-4,params,D) # Invasion fitness right of the resident
                lamc = lam(sol,r_star,params,D)
                grad2 = (lamr+laml-2*lamc)
                if grad2 < 0:
                    ess.append([D,r_star])
                else :
                    bra.append([D,r_star])
        countery += 1
    counterx += 1
Arr = [arr.copy()]
ESS = [np.array(ess)]
Bra = [np.array(bra)]
counterx = 0
D = 1
ess,bra = [[100,100]],[[100,100]]
for P in Preds: # Select here what you want to vary
    
    countery = 0 # Counter ofr vertical array-elemtns
    for r in np.linspace(0,1,100):
        params = g,K,s,P,np.sqrt(ratio)
        sol = Newton(lambda u : f(u,params,r,D,Holling=3),lambda u : Df(u,params,r,D),sol,50,1,1e-12)
        laml = lam(sol,r-1e-4,params,D) #Invasion fitness left of the resident
        lamr = lam(sol,r+1e-4,params,D) # Invasion fitness right of the resident
        grad = (lamr-laml)/1e-4/2 # Slope of the invasion fitness
        arr[countery,counterx] = grad
        if r > 0:
            if arr[countery-1,counterx]*arr[countery,counterx] < 0:
                r_star = interroot( r-1/99, r,arr[countery-1,counterx], arr[countery,counterx])
                laml = lam(sol,r_star-1e-4,params,D) #Invasion fitness left of the resident
                lamr = lam(sol,r_star+1e-4,params,D) # Invasion fitness right of the resident
                lamc = lam(sol,r_star,params,D)
                grad2 = (lamr+laml-2*lamc)/1e-8
                if grad2 < 0:
                    ess.append([P,r_star])
                else :
                    bra.append([P,r_star])
        countery += 1
    counterx += 1
Arr.append(arr.copy())
ESS.append(np.array(ess))
Bra.append(np.array(bra))
counterx = 0
P=1.5
ess,bra = [[100,100]],[[100,100]]
for s in Ss: # Select here what you want to vary
    
    countery = 0 # Counter ofr vertical array-elemtns
    for r in np.linspace(0,1,100):
        params = g,K,s,P,np.sqrt(ratio)
        sol = Newton(lambda u : f(u,params,r,D,Holling=3),lambda u : Df(u,params,r,D),sol,50,1,1e-12)
        laml = lam(sol,r-1e-4,params,D) #Invasion fitness left of the resident
        lamr = lam(sol,r+1e-4,params,D) # Invasion fitness right of the resident
        grad = (lamr-laml)/1e-4/2 # Slope of the invasion fitness
        arr[countery,counterx] = grad
        if r > 0:
            if arr[countery-1,counterx]*arr[countery,counterx] < 0:
                r_star = interroot( r-1/99, r,arr[countery-1,counterx], arr[countery,counterx])
                laml = lam(sol,r_star-1e-4,params,D) #Invasion fitness left of the resident
                lamr = lam(sol,r_star+1e-4,params,D) # Invasion fitness right of the resident
                lamc = lam(sol,r_star,params,D)
                grad2 = (lamr+laml-2*lamc)/1e-8
                if grad2 < 0:
                    ess.append([s,r_star])
                else :
                    bra.append([s,r_star])
        countery += 1
    counterx += 1
Arr.append(arr.copy())
ESS.append(np.array(ess))
Bra.append(np.array(bra))
counterx = 0
s = .8
P = 1.5
ess,bra = [[100,100]],[[100,100]]
for ratio in Ratios: # Select here what you want to vary
    
    countery = 0 # Counter ofr vertical array-elemtns
    for r in np.linspace(0,1,100):
        params = g,K,s,P,np.sqrt(ratio)
        sol = Newton(lambda u : f(u,params,r,D,Holling=3),lambda u : Df(u,params,r,D),sol,50,1,1e-12)
        laml = lam(sol,r-1e-4,params,D) #Invasion fitness left of the resident
        lamr = lam(sol,r+1e-4,params,D) # Invasion fitness right of the resident
        grad = (lamr-laml)/1e-4/2 # Slope of the invasion fitness
        arr[countery,counterx] = grad
        if r > 0:
            if arr[countery-1,counterx]*arr[countery,counterx] < 0:
                r_star = interroot( r-1/99, r,arr[countery-1,counterx], arr[countery,counterx])
                laml = lam(sol,r_star-1e-4,params,D) #Invasion fitness left of the resident
                lamr = lam(sol,r_star+1e-4,params,D) # Invasion fitness right of the resident
                lamc = lam(sol,r_star,params,D)
                grad2 = (lamr+laml-2*lamc)/1e-8
                if grad2 < 0:
                    ess.append([ratio,r_star])
                else :
                    bra.append([ratio,r_star])
        countery += 1
    counterx += 1
Arr.append(arr.copy())
ESS.append(np.array(ess))
Bra.append(np.array(bra))
# Plots
fig,axs = plt.subplots(1,4,figsize=(12,3),sharey=True)
X = [Ds,Preds,Ss,Ratios]
for i in range(4):
    axs[i].contour(X[i],Rs,Arr[i],[0])
    axs[i].scatter(ESS[i][:,0],ESS[i][:,1],c=palette[-2],marker='*',label='ESS')
    axs[i].scatter(Bra[i][:,0],Bra[i][:,1],c=palette[0],marker='s',label='Branching point')
    y_direc = Arr[i][::10,::5] 
    axs[i].quiver(X[i][::5],Rs[::10],np.zeros_like(y_direc),y_direc/np.abs(y_direc),alpha=np.abs(y_direc)/np.max(np.abs(y_direc)))
    axs[i].set_xlim(X[i][0],X[i][-1])
    axs[i].set_ylim(0,1)
axs[2].legend(loc='upper center',bbox_to_anchor=(-0.15,1.15),ncols=2)
axs[0].set_ylabel("Trait value")
axs[0].set_xlabel("Dispersal")
axs[1].set_xlabel("Attack rate")
axs[2].set_xlabel("Trade-off")
axs[3].set_xlabel("Opportunity cost")

