# -*- coding: utf-8 -*-
import numpy as np
def morans_i(M,W):
    '''
    Compute Morans index
    
    Parameters:
        M: 2D numpy array 
            The data array for which Morans I will be computed
        W: 2D numpy array 
            The weight matrix for the caluclation
    
    Returns:
        float
        Morans index
    
    '''
    
    x = np.ndarray.flatten(M) # Flatten the data
    N = len(x) 
    x_mean = np.mean(x)
    
    v = x - np.ones(N)*x_mean 
    v_t = np.transpose(v)
    I = (v_t@W@v)/(v_t@v) / (np.mean(W)*N)
    return I
    

def periodic_rook(row,col):
    '''
    Compute weight matrix for periodic rook contingency
    
    Parameters:
        row: integer
            number of rows
        col: integer
            number of columns
            
    Returns:
        2D numpy array
        Weight matrix for periodic rook contingency
    '''
    vec1 = np.ones(col-1) # Right side neighbour
    vec2 = np.ones(col) # Neighbour up on top
    vec3 = [1] # Neighbour left on side
    vec4 = np.ones(col*(row-1)) # Neighbour down
    
    for k in range(row-1):
        vec1 = np.append(vec1,0)
        vec1 = np.append(vec1,np.ones(col-1))
        
        vec3 = np.append(vec3,np.zeros(col-1))
        vec3 = np.append(vec3,1)
        
    M = np.diag(vec1,1) + np.diag(vec2,col*(row-1)) + np.diag(vec3,col-1) + np.diag(vec4,col)
    M = M + np.transpose(M)
    return M
    
def complete_spacePeriodic(row,col,L):
    '''
    Compute the weight matrix using 1/(shortest distance) as weights
    Parameters:
        row: integer
            number of rows
        col: integer
            number of columns
        L : float
            length of the periodic square
    
    Returns:
        2D numpy array
        weight matrix
    '''
    N = row*col
    h,k = L/row, L / col # Step sizes in x,y direction
    row_half, col_half = row//2,col//2
    W = np.zeros((N,N))
    dist_vert = np.zeros(row) # The distances in straight y direction
    dist_hor = np.zeros(col) # The distances in straight x direction
    
    for i in range(row_half):
        dist_vert[i] = i*h
    for i in range(row_half):
        dist_vert[i+row_half] = (row_half-i)*h
    for j in range(col_half):
        dist_hor[j] = j*k
    for j in range(col_half):
        dist_hor[j+col_half] = (col_half-j)*k

    def dist_sq(pos1,pos2,dist_vert,dist_hor):
        # Compute the inverse square distance of two points in the periodic square using the Pythagorean theorem
        i1,j1,i2,j2 = pos1[0],pos1[1],pos2[0],pos2[1]
        d = (dist_vert[max(i1,i2)-min(i1,i2)]**2 + dist_hor[max(j1,j2)-min(j1,j2)]**2)
        return 1/d
    
    Indices = [] # Stores Indices for each gridpoint as list
    for i in range(row):
        for j in range(col):
            Indices.append((i,j))
            
    for I in range(N):
        for J in range(N-I-1):
            W[I,J+I+1] = dist_sq(Indices[I],Indices[J+I+1],dist_vert,dist_hor) # Computes upper triangular part of weight matrix W
            
    W = W + np.transpose(W) # W is symmetric, here I combine the upper and lower triangular part
    return W
    

class RandomVector():
    
    def __init__(self,length):
        '''
        RandomVector object

        Parameters
        ----------
        length : float
            Length of the vector.

        Returns
        -------
        None.

        '''
        self.length = length
        theta = 2*np.pi*np.random.rand()
        self.vec = length*np.array([np.cos(theta),np.sin(theta)])
        
    def dot(self,dx,dy):
        '''
        Compute dot product with another vector.

        Parameters
        ----------
        dx : float
            x coordinate.
        dy : float
            y coordinate.

        Returns
        -------
        float
            dot product.

        '''
        disp = np.array([dx,dy])
        
        return np.dot(self.vec,disp)
    
    def interpolate(vec0,vec1,vec2,vec3,p):
        '''
        Interpolate perlin noise at position p with 4 vectors at each surrounding lattice.
        
        Parameters
        ----------
        vec0 : RandomVector
            Down left.
        vec1 : RandomVector
            Down right.
        vec2 : RandomVector
            Top left.
        vec3 : RandomVector
            Top right.
        p : Numpy array
            Position of the point p.

        Returns
        -------
        float
            Interpolation.

        '''
        def spline3(a,b,x): # Cubic spline interpolation
            return a + x**2*(3*(b-a) + 2*(a-b)*x)
        
        x,y = p[0],p[1] # Point coordinates
        x0,y0 = int(x),int(y) # Point coordinates rounded
        dx,dy = x-x0,y-y0 # Displacement to bottom left gridpoint
        
        n0,n1 = vec0.dot(dx,dy),vec1.dot(dx-1,dy) # Dotproduct on vertical axis down
        i0 = spline3(n0,n1,dx) # Interpolation on bottom line
        n2,n3 = vec2.dot(dx,dy-1),vec3.dot(dx-1,dy-1) # Dot product on vertical axis upper line
        i1 = spline3(n2,n3,dx) # Interpolation on upper line
        
        return spline3(i0,i1,dy) # Interpolation between the two previous interpolation points
    
    
    
def periodic_Noise(M,N,amp):
    '''
    Computes periodic Perlin noise for MxM lattices on NxN grid with amplitude amp.

    Parameters
    ----------
    M : int
        Number of lattices in a row and column.
    N : int
        Number of gridpoints in row and column.
    amp : float
        Ammplitude of the noise.

    Returns
    -------
    Noise_landscape : 2D array
        Perlin noise.

    '''
    V = np.array([RandomVector(amp) for i in range(M*M)]) # List of random vectors for each lattice
    V_matrix = V.reshape((M,M)) # Reshape into grid-format
    V_extend = np.pad(V_matrix,pad_width=1,mode='wrap') # Pad such that corresponding border vectors become neighbours
    Vec_index = [int(i*M/N) for i in range(N)] # Indices for bottom left lattice vector corresponding to each point of the new NxN grid
    
    Noise_landscape = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            iv,jv = Vec_index[i],Vec_index[j]
            vec0 = V_extend[iv,jv]
            vec1 = V_extend[iv+1,jv]
            vec2 = V_extend[iv,jv+1]
            vec3 = V_extend[iv+1,jv+1]
            p = np.array([i/N*M,j/N*M])
            Noise_landscape[i,j] = RandomVector.interpolate(vec0, vec1, vec2, vec3, p)
            
    return Noise_landscape #,V_extend[:-1,:-1]

      
def perlin(N,start,layers):
    '''
    Creates an array by summing several times periodic Perlin noise with decreasing amplitude and increasing frequency.

    Parameters
    ----------
    N : int
        Number of rows and columns of desired array.
    start : int
        Frequency of layer with leading amplitude.
    layers : int
        Number of layers.

    Returns
    -------
    my_map : 2D array
        Sum of the layers.

    '''
    my_map = periodic_Noise(start,N,1)
    for i in range(layers-1):
        my_map += periodic_Noise(start+i+1,N,2**(-i-1))

    return my_map 

def weights(N,M):
    '''
    Compute the weights for the barycentric interpolation formula for the trigonometric interpolant

    Parameters
    ----------
    N : int
        Initial grid discretization.
    M : int
        Target grid discretization.

    Returns
    -------
    w : 2D array
        Barycentric weights.

    '''
    w = np.zeros((M,N))
    X_prev = np.linspace(0,2*np.pi*(1-1/N),N) # Previous discretized grid
    X = np.linspace(0,2*np.pi*(1-1/M),M) # New discretized grid
    alternating = (-1)**(np.arange(N))
    for i in range(M):
        x = X[i]
        if (i*N) % M == 0: # Catch singular expression when x equals grid node
            w[i,int(N/M*(i))] = 1 
        else :
            w[i,:] = 1 / np.tan((x-X_prev)/2)*alternating # Baycentric formula
            w[i,:]/= np.sum(w[i,:])
    return w

def regrid(F,weights):
    '''
    Interpolates square Array to another grid with weights from some Barycentric formula. Grid nodes are assumed to be equispaced.

    Parameters
    ----------
    F : 2D square array
        Interpolation points.
    weights : 2D array
        Weights for barycentric formula for the trigonometric interpolant.

    Returns
    -------
    F_inty : 2D square array
        Newly computed interpolation.

    '''
    F_intx = weights@F 
    F_inty = F_intx@ weights.transpose()
    return F_inty   
import matplotlib.pyplot as plt



