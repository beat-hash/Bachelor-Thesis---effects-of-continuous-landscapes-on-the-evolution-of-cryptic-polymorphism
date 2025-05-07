"""
Created on Mon Jan 20 13:23:02 2025

@author: Jannis
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
import matplotlib.colors as mcolors
from multi_species import MyMultiSpecies as msp
import scipy.interpolate as scip 
from landscapes import *

def slope(M):
    n = np.shape(M)[0]
    M2 = np.zeros(np.shape(M)) # Array will contain Laplacian
    frequ = np.fft.fftfreq(n,d = 2*np.pi/n)*2*np.pi # Frequencies
    for i in range(n):
        V = M[:,i] + M[i,:]*1j  # Row and column for index
        V_hat = np.fft.fft(V) 
        V_hat_frequ = 1j* frequ*V_hat # Applying formula for 2nd derivatives
        V_back = np.fft.ifft(V_hat_frequ)
        M2[:,i]+= np.real(V_back)**2 # Laplacian in column direction
        M2[i,:]+= np.imag(V_back)**2 # Laplacian in row direction
    
    return M2


# Graphic: trade-off curves
f1 = lambda x,s : np.abs(x-0.5)**s 
f2 = lambda x,s : (1-0.5*np.exp(-(x-0.5)**2/s**2))
R = np.linspace(0,1,201)

fig,axs = plt.subplots(1,2,sharey=True,sharex=True,figsize=(8,4))
sns.set_theme(style='whitegrid')
for s in np.linspace(0.1,0.5,3):
    sns.lineplot(x=R,y=f2(R,s),label=f"{np.round(s,decimals=2)}",ax=axs[1],color='blue',alpha=s*2)
for s in np.linspace(0.5,1.5,3):
    sns.lineplot(x=R,y=f1(R,s),label=f"{np.round(s,decimals=2)}",ax=axs[0],color='blue',alpha=s/1.5)
axs[0].legend(title="Trade-off")
axs[1].legend(title="Trade-off")
axs[0].set_ylabel("Attack rate")
axs[0].set_xlabel("Trait value")
axs[1].set_xlabel("Trait value")
plt.show()

# Load Datasets for visualization
df_oppcost = pd.read_pickle("ExperimentAstarFixed.pkl") # DataFrame, experiment with opportunity cost
df = pd.read_pickle("MyExperimentA.pkl") # DataFrame, experiment A 
W = complete_spacePeriodic(32, 32, 2*np.pi)
df['Local Morans'] = df['Landscape'].apply(lambda l : morans_i(np.log(l+1e-6).reshape((32,32)), W))


# Figure 3.3 - spline interpolation map
from mpl_toolkits.axes_grid1 import make_axes_locatable

palette=sns.color_palette('ch:',as_cmap=False)

fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(12,6.5),sharex=True)

x = df['Morans I'].to_numpy()
y = df['Variation'].to_numpy()
z = df['Sympatric branching']

sns.set_theme(style='whitegrid')
tck = scip.bisplrep(x, y, z)


x_mesh,y_mesh = np.meshgrid(x,y)
tol = 1e-2*9 # Used to cut the image at the edges: At each visualization point there should be 50 datapoints within a ball of radius tol
Indices = np.zeros((len(y),len(x))) == 1
i = 0
sx,sy = (x.max()-x.min()),y.max()-y.min()
for x_i in x:
    j = 0
    for y_i in y:
        A = (x-x_i)**2/sx**2
        B = (y - y_i)**2/sy**2
        if np.sort(A.flatten()+ B.flatten())[50] < tol**2: # Criteria to be in cluded in visualization 
            Indices[j,i] = True 
        j+= 1 
    i+=1 

z_new = scip.bisplev(x, y, tck)
palette=sns.color_palette('ch:',as_cmap=False)
scplt = axs[0,0].scatter(x_mesh.flatten()[Indices.flatten()],y_mesh.flatten()[Indices.flatten()],c=z_new.T.flatten()[Indices.flatten()])

divider = make_axes_locatable(axs[0,0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(scplt, cax=cax, orientation='vertical')

# Parapatric branching
z = df['Parapatric branching']
tck = scip.bisplrep(x, y, z)
z_new = scip.bisplev(x, y, tck)
scplt = axs[0,1].scatter(x_mesh.flatten()[Indices.flatten()],y_mesh.flatten()[Indices.flatten()],c=z_new.T.flatten()[Indices.flatten()])

divider = make_axes_locatable(axs[0,1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(scplt, cax=cax, orientation='vertical')

# ESS 
z = df['ESS']
tck = scip.bisplrep(x, y, z)
z_new = scip.bisplev(x, y, tck)
scplt = axs[0,2].scatter(x_mesh.flatten()[Indices.flatten()],y_mesh.flatten()[Indices.flatten()],c=z_new.T.flatten()[Indices.flatten()])

divider = make_axes_locatable(axs[0,2])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(scplt, cax=cax, orientation='vertical')

# With Skewness - sympatric branching
x = df['Morans I'].to_numpy()
y = df['Skewness'].to_numpy()
z = df['Sympatric branching']

x_mesh,y_mesh = np.meshgrid(x,y)

Indices = np.zeros((len(y),len(x))) == 1
i = 0
sx,sy = (x.max()-x.min()),y.max()-y.min()
for x_i in x:
    j = 0
    for y_i in y:
        A = (x-x_i)**2/sx**2
        B = (y - y_i)**2/sy**2
        if np.sort(A.flatten()+ B.flatten())[50] < tol**2:
            Indices[j,i] = True 
        j+= 1 
    i+=1 


tck = scip.bisplrep(x, y, z)

z_new = scip.bisplev(x, y, tck)

scplt = axs[1,0].scatter(x_mesh.flatten()[Indices.flatten()],y_mesh.flatten()[Indices.flatten()],c=z_new.T.flatten()[Indices.flatten()])
divider = make_axes_locatable(axs[1,0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(scplt, cax=cax, orientation='vertical')


# Parapatric branching
z = df['Parapatric branching']

sns.set_theme(style='whitegrid')
tck = scip.bisplrep(x, y, z)
z_new = scip.bisplev(x, y, tck)
scplt = axs[1,1].scatter(x_mesh.flatten()[Indices.flatten()],y_mesh.flatten()[Indices.flatten()],c=z_new.T.flatten()[Indices.flatten()])

divider = make_axes_locatable(axs[1,1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(scplt, cax=cax, orientation='vertical')

# ESS 
z = df['ESS']

sns.set_theme(style='whitegrid')
tck = scip.bisplrep(x, y, z)
z_new = scip.bisplev(x, y, tck)
scplt = axs[1,2].scatter(x_mesh.flatten()[Indices.flatten()],y_mesh.flatten()[Indices.flatten()],c=z_new.T.flatten()[Indices.flatten()])

divider = make_axes_locatable(axs[1,2])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(scplt, cax=cax, orientation='vertical')


axs[0,0].set_title("Sympatric branchings",wrap=True,fontsize=18)
axs[0,1].set_title("Parapatric branchings",wrap=True,fontsize=18)
axs[0,2].set_title("ESS",wrap=True,fontsize=18)

for i in range(3):
    axs[1,i].set_xlabel("Morans I")
axs[0,0].set_ylabel("Variation")
axs[1,0].set_ylabel("Skewness")
for i in range(2):
    for j in range(2):
        axs[i,j+1].set_yticks([])
plt.show()


# Lineplots 
plt.rcParams['text.usetex'] = False
df['Branching'] = df['Sympatric branching'] + df['Parapatric branching']
df['High Morans'] = df['Morans I'].apply(lambda x : x > 1/2)
df['High Variation'] = df['Variation'].apply(lambda x : x > 0.035)

df_oppcost['Branching'] = df_oppcost['Sympatric branching'] + df_oppcost['Parapatric branching']
df_oppcost['High Skewness'] = df_oppcost['Skewness'].apply(lambda x : x**2 > df_oppcost['Skewness'].quantile(3/4))
df_oppcost['High Morans'] = df_oppcost['Morans I'].apply(lambda x : x > df_oppcost['Morans I'].quantile(0.75))
df_oppcost['High Variation'] = df_oppcost['Variation'].apply(lambda x : x > df_oppcost['Variation'].quantile(0.75))


# Branching - Morans I
fig,axs = plt.subplots(1,3,figsize=(12,3),sharey=True)
sns.regplot(data=df[df['High Variation']==True],x='Morans I',y='Branching',scatter=False,order=1,color=palette[2],ax=axs[0],label=r'Variation $\geq 0.035$')
sns.regplot(data=df[df['High Variation']==False],x='Morans I',y='Branching',scatter=False,order=1,color=palette[-1],ax=axs[0],label=r'Variation $\leq 0.035$')
plt.rcParams['text.usetex'] = True
axs[0].legend()
plt.rcParams['text.usetex'] = False
# Branching - Variation
sns.regplot(data=df[df['High Morans']==True],x='Variation',y='Branching',scatter=False,color=palette[2],ax=axs[1],label=r'Morans I $\geq 0.5$')
sns.regplot(data=df[df['High Morans']==False],x='Variation',y='Branching',scatter=False,color=palette[-1],ax=axs[1],label=r'Morans I $\leq 0.5$')
plt.rcParams['text.usetex'] = True
axs[1].legend()
plt.rcParams['text.usetex'] = False
# Branching - Skewness 
sns.regplot(data=df,x='Skewness',y='Branching',scatter=False,order=2,color='black',ax=axs[2])
plt.show()

# ESS - Morans I
sns.regplot(data=df[df.Skewness **2 > 0.2],x='Morans I',y='ESS',scatter=False,color='black',order=1)
plt.legend()
plt.show()

# Branching - Skewness*Morans I
fig,axs = plt.subplots(1,2,figsize=(8,3),sharey=True)
df['SkewnessMorans'] = df['Skewness']*df['Morans I']
df_oppcost['SkewnessVariation'] = df_oppcost['Skewness']*df_oppcost['Variation']
df['SkewnessVariation'] = df['Skewness']*df['Variation']
df_oppcost['SkewnessMorans'] = df_oppcost['Skewness']*df_oppcost['Morans I']

sns.regplot(data=df_oppcost,x='SkewnessMorans',y='Branching',scatter=False,color='black',ax=axs[0],line_kws={'linestyle':'dashed'})
sns.regplot(data=df,x='SkewnessMorans',y='Branching',scatter=False,color='black',ax=axs[0],order=2)
axs[0].set_xlabel("Skewness * Morans I")

sns.regplot(data=df_oppcost,x='SkewnessVariation',y='Branching',scatter=False,color='black',ax=axs[1],line_kws={'linestyle':'dashed'},label='Opportunity cost')
sns.regplot(data=df,x='SkewnessVariation',y='Branching',scatter=False,color='black',ax=axs[1],order=2,label='No cost')
axs[1].set_xlabel("Skewness * Variation")
axs[1].set_ylabel("")
axs[1].legend()
plt.show()


def Laplace_Fourier(M): # Compute discrete Laplacian of 2D array M
    n = np.shape(M)[0]
    M2 = np.zeros(np.shape(M)) # Array will contain Laplacian
    frequ = np.fft.fftfreq(n,d = 2*np.pi/n)*2*np.pi # Frequencies
    for i in range(n):
        V = M[:,i] + M[i,:]*1j  # Row and column for index
        V_hat = np.fft.fft(V) 
        V_hat_frequ = - frequ**2*V_hat # Applying formula for 2nd derivatives
        V_back = np.fft.ifft(V_hat_frequ)
        M2[:,i]+= np.real(V_back) # Laplacian in column direction
        M2[i,:]+= np.imag(V_back) # Laplacian in row direction
    return M2
        
def weights(N,M): # Compute weights for Baycentric formula
    w = np.zeros((M,N))
    X_prev = np.linspace(0,2*np.pi*(1-1/N),N) # Previous discretized grid
    X = np.linspace(0,2*np.pi*(1-1/M),M) # New discretized grid
    alternating = (-1)**(np.arange(N))
    for i in range(M):
        x = X[i]
        if (i*N) % M == 0: # Catch singular expression when x equals grid node
            w[i,int(N/M*(i+1))] = 1 
        else :
            w[i,:] = 1 / np.tan((x-X_prev)/2)*alternating # Baycentric formula
            w[i,:]/= np.sum(w[i,:])
    return w
            
def regrid1D(F,weights): # Interpolate 1D array F
    return weights@F 

def regrid2D(F,weights): # Interpolate 2D array F
    F_inty = weights@F # Interpolate in y direction first
    F_intx = F_inty@ weights.transpose() # Interpolate x direction
    return F_intx




# Show order of convergence for two different functions
fig,axs = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(12,6))
N = 500 # Number of grid points in row and column of the fine grid
sns.set_theme(style='whitegrid')
x,y = np.meshgrid(np.linspace(0,2*np.pi*(1-1/N),N),np.linspace(0,2*np.pi*(1-1/N),N)) # Domain

# First f is smooth function
f = lambda x,y: np.exp(np.sin(x)+np.sin(y))
f2 = lambda x,y : f(x,y)*(-np.sin(x)-np.sin(y)+np.cos(x)**2+np.cos(y)**2) # Derivative
F2 = f2(x,y)
E = [] # Error
I = [] # Stepsize
for i in range(30):
    n = 2*(i+1) # Discretization
    I.append(n)
    w = weights(n,N)
    x_n,y_n = np.meshgrid(np.linspace(0,2*np.pi*(1-1/n),n),np.linspace(0,2*np.pi*(1-1/n),n))
    F_n = f(x_n,y_n) # Evaluate F on coarse grid
    Df_n = Laplace_Fourier(F_n) # compute discrete laplacian
    Df_nup = regrid2D(Df_n,w) # Interpolate
    E.append(np.linalg.norm(Df_nup-F2,ord=np.inf)) # Compute error on fine grid

# Plot the error:
sns.lineplot(x=I,y=E,ax=axs[0])
sns.scatterplot(x=I,y=E,ax=axs[0])
sns.lineplot(x=I,y=np.exp(-np.array(I)),ax=axs[0],color='black')

axs[0].set_ylim(1e-12,1e4)
axs[0].set_xlabel("N")
axs[0].set_title("Smooth function",fontsize=18)

# Next f is less smooth, we choose a polynomial with jump discontinuity at the ends of the domain in the 3rd derivative
# Polynomial parameters:
N = 500
a = 5+1/3 
b = -10-2/3
c = a
# Functions:
f = lambda x,y: (a*(x/2/np.pi)**4 + b*(x/2/np.pi) **3 + c * (x/2/np.pi)**2)+(a*(y/2/np.pi)**4 + b*(y/2/np.pi) **3 + c * (y/2/np.pi)**2)
f2 = lambda x,y : ((12*a*(x/2/np.pi)**2 + 6*b*(x/2/np.pi) + 2*c)+(12*a*(y/2/np.pi)**2 + 6*b*(y/2/np.pi) + 2*c))/(np.pi*2)**2
x,y = np.meshgrid(np.linspace(0,2*np.pi*(1-1/N),N),np.linspace(0,2*np.pi*(1-1/N),N))
F2 = f2(x,y)
E = [] # Error 
I = [] # Step size
for i in range(25):
    n = 8*(i+1)+100 # Discretization
    I.append(n)
    w = weights(n,N)
    x_n,y_n = np.meshgrid(np.linspace(0,2*np.pi*(1-1/n),n),np.linspace(0,2*np.pi*(1-1/n),n))
    F_n = f(x_n,y_n) # Evaluate f on coarse grid
    Df_n = Laplace_Fourier(F_n) # Compute discrete laplacian
    Df_nup = regrid2D(Df_n,w) # Interpolate to fine grid
    E.append(np.linalg.norm(Df_nup-F2,ord=np.inf)) # Compute error with target function

# Make the plots
sns.lineplot(x=I,y=E,ax=axs[1])
sns.lineplot(x=I,y=2/np.array(I),ax=axs[1],color='black')
sns.scatterplot(x=I,y=E,ax=axs[1])
#plt.xscale('log')

axs[1].set_xlabel("N")
axs[1].set_yscale('log')
axs[1].set_title("Polynomial",fontsize=18)

plt.show()


# Plot: Visualize Fourier interpolation in 1D and 2D
from mpl_toolkits.mplot3d import Axes3D

F1D = np.random.rand(6) # Random points on line to be interpolated
F2D = np.random.rand(6,6) # Random points in space to be interpolated
w = weights(6,50) # Weights for barycentric interpolation
F1D_int = regrid1D(F1D,w) # Array holding interpolation
F2D_int = regrid2D(F2D,w) # Array holding interpolation

X_coarse = np.linspace(0,2*np.pi*(1-1/6),6) 
X_fine = np.linspace(0,2*np.pi*(1-1/50),50)

# Make the plot
sns.set_theme(style='white')
cmap = sns.color_palette('mako',as_cmap=True)
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121)
ax1.plot(X_fine,F1D_int,c='b')
ax1.scatter(X_coarse,F1D,c='b',s=100)

ax2 = fig.add_subplot(122,projection='3d')
x_coarse,y_coarse = np.meshgrid(X_coarse,X_coarse)
x_fine,y_fine = np.meshgrid(X_fine,X_fine)
ax2.plot_surface(x_fine,y_fine,F2D_int,cmap=cmap,alpha=0.8,edgecolor='b')
ax2.scatter(x_coarse,y_coarse,F2D,s=100,c='yellow',edgecolor='black')
ax2.view_init(elev=60, azim=-45)
    
        