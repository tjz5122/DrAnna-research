
'''
work:
1. derive SABR boundary 0 with respect to s iNtead of traNformation of x, also test SABR boundary online
2. Error calculation for Heston
   Nx = 16   Nt = c*16**2, if not pass multiply constant c1
   Nx = 32   Nt = c*32**2, if not pass multiply constant c2 go back to 16 again where M = 16**2 * c2
   ... 
   2**n  closest to max capacity


'''
import numpy as np
import matplotlib.pyplot as plt




def solver(N, Nt, c):
    min_v = 0.2
    max_v = 2
    min_price = 1 #since the uniform mesh, then dv = dx
    max_price = 40
    rho = 0.5
    theta = 1  #mean min vol and max vol
    kappa = 2  # the degree of mean reversion
    r = 0.2   #0.02 #risk-free rate
    sigma = 1 #volvol
    K = 20    #strike price 
    
    T = 1
    L = 1
    
    ds = (max_price - min_price)/N #1
    dv = (max_v - min_v)/N #1
    dt = T/Nt
    
    u = np.zeros((Nt+1, N+1, N+1))
    
    #check if stability condition holds
    if dt <= ((dv**2) * (ds**2))/(r*(dv**2)*(ds**2) + max_v*(sigma**2)*(ds**2) + max_v*(max_price**2)*(dv**2)):  #stability condition
    #dt*(r*(dv**2)*(ds**2) + max_v*(sigma**2)*(ds**2) + max_v*(max_price**2)*(dv**2)) <= (dv**2) * (ds**2):
            
        #initial condition applied
        for i in range(0, N):
            for j in range(0, N):
                u[0,i,j] = max((min_price+(i*ds)-K),0)  
                
        #loop through the matrix to insert value
        for t in range(0, Nt):
            for i in range(0, N):
                si = min_price + i*ds
                u[t+1,i,N] = si
                
                if i != N:
                    u[t+1,i,0] = u[t,i,0] - (r*si*dt/ds)*(u[t,i+1,0] - u[t,i,0]) - (kappa*theta*dt/dv)*(u[t,i,1] - u[t,i,0]) + dt*r*u[t,i,0]
                    
                for j in range(0, N):  
                    vj = min_v + j*dv
                    
                    #new boundary condition
                    if j != 0: 
                        u[t+1,0,j] = 0 
                    
                    if ((i != 0) or (i != N) or (j != 0) or (j != N)):
                        #iteration
                        a1 = 1 - (r*dt) - (dt*vj*(si**2))/(ds**2) - (dt*vj*(sigma**2) / (dv**2))
                        a2 = (dt/2)*si*(vj*si/(ds**2) + (r/ds))
                        a3 = (dt/2)*si*(vj*si/(ds**2) - (r/ds))
                        a4 = (dt/2)*((sigma**2)*vj/(dv**2) + kappa*(theta - vj)/dv)
                        a5 = (dt/2)*((sigma**2)*vj/(dv**2) - kappa*(theta - vj)/dv)
                        a6 = (dt*rho*sigma*vj*si)/(4*ds*dv)
                        u[t+1,i,j] = a1*u[t,i,j] + a2*u[t,i+1,j] + a3*u[t,i-1,j] + a4*u[t,i,j+1] + a5*u[t,i,j-1] + a6*(u[t,i+1,j+1] - u[t,i+1,j-1] - u[t,i-1,j+1] + u[t,i-1,j-1])
                       
                    if i == (N-1):
                        u[t+1,N,j] = ds + u[t+1,N-1,j]
                        
        return u[-1]
    else:
        return np.zeros(1)



def RelativeErrorMatrix(result1, result2):
    #print(len(result1), len(result1[0]), len(result2), len(result2[0]))
    abserror = np.amax(abs(result2 - result1))
    relative_error = np.amax(abs(result2 - result1)/ abs(result1))
    return relative_error



def process(base_N, max_N, c):
    '''
    Nx = 16   Nt = c*16**2, if not pass multiply constant c1
    Nx = 32   Nt = c*32**2, if not pass multiply constant c2 go back to 16 again where M = 16**2 * c2
    .
    .
    .
    2**n  closest to max capacity
    '''
    N = base_N
    error_list = []
    pre_result = solver(N, c*(N**2), c)
    print(pre_result)
    if pre_result.any() != False:
        N *= 2
    else:
        process(N, max_N, c*2)
        
    while (N <= max_N):

        Nt = c*(N**2)
        new_result = solver(N, Nt, c)
        if new_result.any() != False:
            #conditon 1:
            comparable_result = np.array([[new_result[i][j] for i in range(0,len(new_result),2)] for j in range(0,len(new_result[0]),2)])
            #conditon 2:
            #comparable_result = np.array([[new_result[i][j] for i in range(0,len(new_result),N/base_N)] for j in range(0,len(new_result[0]),N/base_N)])
            
            relative_error = RelativeErrorMatrix(comparable_result, pre_result)
            
            error_list.append(relative_error)
            
            #conditon 1:
            pre_result = new_result
            #conditon 2:
            #pre_result = comparable_result

        else:
            process(N, max_N, c*2)
    
        #increase the N to next level
        N *= 2
    
    return error_list
        
    
#all parameters large than 0
max_N = 100
N = 16
c = 1

print(process(N, max_N, c))
