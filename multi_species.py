# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg as sclg
from scipy import sparse as sps
import scipy.integrate as sci  
class MyMultiSpecies():
    '''
    This class initiates objects describing the ecology of the model in my thesis and lets us compute the steady state, selection gradient and invasion fitness in the current state.  It allows for more than one prey species the model, but in my thesis this was never used.
    '''
    def __init__(self,n_prey,N,length,Landscape, Res,prey,pred,R,params,pred_constant = False,N_mini=None):

        self.resp,self.g,self._a,self._s,self._d,self.eff,self.gamma = params 
        self.length = length # Length of a side of the periodic square
        self.n_prey = n_prey # Number of prey. The number of predators is always 1.
        self.N = N # Discretization
        self.dx = length/N # Step length in space
        self.L = Landscape # Colour landscape
        self.prey = prey # Initial prey densities. Must be an array of shape (N*N,n_prey)
        self.pred = pred # Initial predator density. Must be an array of shape (N*N,1)
        self.pred_constant = pred_constant # Bool. If True, the predator is kept constant in the model.
        self.Delta = MyMultiSpecies.D2(N,self.dx) # Dsicrete Laplacian
        self._R = R # Prey trait values.
        self.attackL = np.hstack([MyMultiSpecies.attack(Landscape, R[i],self.a,self.s,self.eff) for i in range(n_prey)]) # How the attack rate varies in the landscape for each prey. 
        self._Res = Res # Ressource landscape
        self.lin = self.linear_part() # Assemble linear term of right-hand-side.
        
    # Getters and setters
    @property # prey trait values
    def R(self):
        return self._R 
    @R.setter 
    def R(self,new_R):
        self._R = new_R
        self.attackL = np.hstack([MyMultiSpecies.attack(self.L, new_R[i],self.a,self.s,self.eff) for i in range(self.n_prey)]) # attack landscape is updated
        self.lin = self.linear_part() # linear part is updated

    
    @property # maximal attack rate
    def a(self):
        return self._a 
    @a.setter 
    def a(self,new_a):
        self._a = new_a 
        self.attackL = np.hstack([MyMultiSpecies.attack(self.L, self.R[i],new_a,self.s,self.eff) for i in range(self.n_prey)]) # update attack landscape
        self.lin = self.linear_part() # update linear part

    
    @property # diffusion constant
    def d(self):
        return self._d 
    @d.setter 
    def d(self,new_d):
        self._d = new_d 
        self.lin = self.linear_part() # update linear part

    @property 
    def Res(self): # ressource landscape
        return self._Res
    @Res.setter 
    def Res(self,new_Res):
        self._Res =new_Res
        self.lin = self.linear_part() # update linear part
    
    @property 
    def s(self): # trade-off parameter 
        return self._s 
    @s.setter 
    def s(self,new_s):
        self._s = new_s 
        self.attackL = np.hstack([MyMultiSpecies.attack(self.L, self.R[i],self.a,new_s,self.eff) for i in range(self.n_prey)]) # update attack landscape
        self.lin = self.linear_part() # update linear part

    
    def flat_Matrix(A):
        '''
        For a matrix A of dimension (n,n) this computes a matrix A* of dimension (n*n,n*n) such that 
        A* flatten(u) = flatten(Au + uA.T)

        Parameters
        ----------
        A : 2D array
            Matrix.

        Returns
        -------
        2D array
            The "flattened" matrix.

        '''
        n = np.shape(A)[0]
        diag_ones = sps.diags(np.ones(n))
        A_tilde = sps.vstack([sps.hstack([A[i,j]*diag_ones for j in range(n)]) for i in range(n)])
        A_tildet = sps.kron(diag_ones,A)
        return  A_tilde + A_tildet
            
    def linear_part(self): # Assembles linear matrix describing the linear term of the PDE right hand side
        pred_constant = self.pred_constant 
        n_prey = self.n_prey
        N = self.N
        Diff = self.Delta*self.d
        diag_diff = Diff  # blocks on the diagonal
        attackL  = self.attackL
        arates = [sps.diags((attackL[:,i]) ) for i in range(n_prey)]
        
        diag_growth = sps.diags(np.ndarray.flatten(self.Res))*self.g# ressource growth
        
        if pred_constant == False:
            lin = sps.kron(np.eye(n_prey+1), diag_diff)  + sps.block_diag([diag_growth]*n_prey+[-np.eye(N*N,N*N)*self.resp]) 
        else :
            pred = self.pred.flatten()
            predation = sps.block_diag([-attack_matrix@ sps.diags(pred) for attack_matrix in arates])
            lin = sps.block_diag([diag_diff+diag_growth]*n_prey) + predation
        return lin
    
    def system(self,U): # PDE right hand side and Jacobian of the right hand side
        pred_constant = self.pred_constant 
        n_prey = self.n_prey
        N = self.N
        gamma = self.gamma
        arates = [sps.diags((self.attackL[:,i]) ) for i in range(n_prey)]
        lin = self.lin 
        
        # Quadratic part for competition and predation
        if pred_constant == False:
            predation = sps.bmat([[None,-sps.vstack(arates)],[sps.hstack(arates)*gamma,None]])
            comp_kron = np.zeros((n_prey+1,n_prey+1))
            comp_kron[:n_prey,:n_prey] = np.ones((n_prey,n_prey))
            comp = sps.kron(comp_kron,-sps.eye(N*N,N*N)*self.g)
            quad = sps.diags((comp+predation)@U)
        else :
            predation = sps.block_diag([-attack_matrix@ sps.diags(self.pred.flatten()) for attack_matrix in arates])
            comp_kron = np.ones((n_prey,n_prey ))
            comp = sps.kron(comp_kron,-sps.eye(N*N,N*N)*self.g)
            quad = sps.diags((comp)@U)
        
        
        return (lin+ quad)@U, lin + quad + sps.diags(U)@ (comp + predation) # F and Jacobian of F

    
    
    def find_eq(self,tol,alpha,maxiter=20,x0=None,update=False):
        '''
        Computes the steady state of the system using AIIE. If alpha is set to 0, this is the standard Newton method.

        Parameters
        ----------
        tol : float
            Tolerance.
        alpha : float
            AIIE parameter.
        maxiter : int, optional
            Maximum iterations. The default is 20.
        x0 : 1D array, optional
            Initial guess. If None, the current state is used as initial guess.
        update : bool, optional
            If True, a message is printed at every AIIE iteration. The default is False.

        Returns
        -------
        check : bool
            Convergence result.

        '''
        f = lambda u : self.system(u) # System right hand side and Jacobian
        if x0 is None:
            if self.pred_constant:    
                x0 = self.prey.flatten(order='F')
            else : 
                x0 = np.hstack([self.prey.flatten(order='F'),self.pred.flatten()])
        n_prey = self.n_prey 
        x_new,check = MyMultiSpecies.AIIE(f,x0,maxiter,tol,alpha,update=update) # Call AIIE
        
        N  = self.N
        if check :
            if self.pred_constant == False:
                self.prey = np.hstack([x_new[i*N*N:(i+1)*N*N].reshape((N*N,1)) for i in range(n_prey)])
                self.pred = x_new[n_prey*N*N:].reshape((N*N,1))
            else : 
                self.prey = np.hstack([x_new[i*N*N:(i+1)*N*N].reshape((N*N,1)) for i in range(n_prey)])
        return check 
    
    
    def AIIE(F,x0,maxiter,tol,alpha0,update=False):
        x_old = x0    
        I = sps.eye(len(x0))
        fx,Dfx = F(x_old)
        if update : # Update about convergence if desired
            print("Current function evaluation is ",np.linalg.norm(fx))
        for i in range(maxiter):
            alpha = alpha0/ (i + 1) # Compute alpha 
            
            Mat = Dfx-alpha*I 
            
            if not alpha0 == 0:
                guess = -fx/alpha0**2*alpha 
            else :
                guess = np.zeros_like(fx)
            
            
            delta_x,conv = sps.linalg.lgmres(Mat,-fx,x0=guess,maxiter=15)
            if conv > 0:
                delta_x = sps.linalg.spsolve(Mat,-fx)
            
            
            
            x_new = x_old + delta_x
            x_old = x_new 
            fx,Dfx = F(x_old)
            if update : # Update about convergence if desired
                print("Current function evaluation is ",np.linalg.norm(fx))
            if (np.linalg.norm(fx) < tol) &( np.linalg.norm(x_old) > 1e-2):
                return x_old,True # Break if already converged
            
            
        return x_new, False
    
    def initial_guess_flat(self,dist):
        if self.pred_constant :
            aL_bar = np.mean(self.attackL*self.pred)
            Res_bar = np.mean(self.Res)
            return (self.g*Res_bar-aL_bar)/self.g*np.ones(self.N**2) + np.ones(self.N**2)*dist
        else :
            aL_bar = np.mean(self.attackL)
            Res_bar = np.mean(self.Res)
            return np.vstack([self.resp/aL_bar/self.gamma*np.ones((self.N**2,1)),(Res_bar-self.resp/aL_bar/self.gamma)*self.g/aL_bar*np.ones((self.N**2,1))]).flatten() + np.ones(self.N**2*2)*dist
        
    def D2(N,dx): # Discrete Laplacian 
        I = np.arange(1,N)
        row = np.ones(N)*(-np.pi**2*2/dx**2-1)/6
        row[1:] = (-1)**(1+I)/np.sin(I*dx/2)**2/2
        D2 = sclg.circulant(row)
        return MyMultiSpecies.flat_Matrix(D2)
    
    def attack(L,r,a,s,eff): # Computes the attack landscape for colour landscape L, trade-off s, effect strength eff, trait value r, maximal attack rate a.
        return a*(1-eff*np.exp(-( np.add(L,-r) )**2/s**2))
    
    def Grad(self,r,u): # Selection gradient
        pred = self.pred
        L = self.L
        a = self.a
        eff = self.eff
        s = self.s
        def aux(r,L,s,a,eff):
            return  2*eff*(r-L)/s**2*a*np.exp(-(r-L)**2/s**2)
        predated = - pred *aux(r,L,s,a,eff)
        D = np.dot(u.flatten(),predated*u)/np.dot(u.flatten(),u.flatten())
        return D
        
    def Grad2(self,r,u): # Second derivative of selection on resident population
        a = self.a 
        eff = self.eff 
        pred = self.pred 
        L = self.L 
        s = self.s 
        def aux(r,L,s,a,eff): 
            return 2*eff*a/s**2*np.exp(-(L-r)**2/s**2)* (1-2*(r-L)**2/s**2)
        D = np.dot(u.flatten(), (-aux(r,L,s,a,eff)*pred*u).flatten() )/ np.dot(u.flatten(),u.flatten())
        return D
    
    def inv_fitness(self,r_inv,x0,maxiter,tol): # Invasion fitness (dominant eigenvector of linearization around the zero solution)
        n_prey = self.n_prey
        L = self.L 
        prey = self.prey
        pred = self.pred 
        Res = self.Res
        Delta = self.Delta
        a= self.a
        eff = self.eff 
        d = self.d
        s = self.s
        g = self.g
        attackL_inv = MyMultiSpecies.attack(L,r_inv,a,s,eff)
        compDynamics = - prey @np.ones((n_prey,1)) # Competition with resident 
        predated = - pred*attackL_inv # Predation
        g = g*(Res + compDynamics) + predated # Reaction term
        F = Delta*d + sps.diags(np.ndarray.flatten(g)) # Reaction + diffusion
        
        l,eig = sps.linalg.eigs(F,k=1,which='LR',v0=x0,maxiter=maxiter) # Dominant eigenvalue
        return np.real(l[0]),np.real(eig)
     
    def integrate(self,T,dt,evol_steps):
        Evolution = []
        t = 0.
        if self.pred_constant : 
            U = self.prey.flatten(order='F') 
    
        while t <= T:
            solution = sci.solve_ivp(lambda t,U : self.system(U)[0],[t,t+dt],U,method='LSODA',jac = lambda t,U : self.system(U)[1].toarray())
            U = solution.y[:,-1]
            prey = np.hstack([U[i*self.N**2:(i+1)*self.N**2].reshape((self.N**2,1)) for i in range(self.n_prey)])
            self.prey = prey 
            traits = self.R.copy() 
            for i in range(self.n_prey):
                selection = self.Grad(traits[i], prey[:,i:i+1])
                traits[i]+= selection*dt*1e-7 # Factor for separating time scales
            Evolution.append(traits)
            self.R = traits
            t+= dt 
        return np.array(Evolution)
    
    def add_prey(self,prey_new,r_new):
        self.n_prey+= 1 
        preys_new = np.zeros((self.N**2,self.n_prey))
        preys_new[:,:self.n_prey-1] = self.prey 
        preys_new[:,-1] = prey_new
        self.prey = preys_new 
        
        traits_new = np.zeros(self.n_prey)
        traits_new[:self.n_prey-1] = self.R 
        traits_new[-1] = r_new 
        self.R = traits_new 
        self.lin = self.linear_part()
    
    
    