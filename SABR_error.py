import numpy as np
import matplotlib.pyplot as plt


def SABR_solver(N, Nt, c):

    #all parameters large than 0
    K = 40    #strike price 
    kappa = 2/3  # the degree of mean reversion
    min_v = 0 #minimum underlying vol
    max_v = 5 #maximum underlying vol
    
    min_s = 2 #minimum underlying stock price
    max_s = 80  #maximum underlying stock price
    
    sigma = 0.1 #volvol
    rho = 0.5#10 coeelation of the browian motion
    theta = 3#1 #mean of vol, should be in the middle of both
    
    T = 1    #length of total time
    L = 1    #length of total stock pirce and vol
    Ns = Nv = 36 #space or stock price nodes
    Nt = 4000 #time nodes
    
    dv = (max_v - min_v)/Nv  #step length of violatility
    ds = (max_s - min_s)/Ns  #step length of stock price
    dt = T/Nt  #step length of time
    
    u = np.zeros((Nt+1, Ns+1, Nv+1))
    
    if (dt <= ((ds*dv)**2)/(((max_s*dv)**2 + (sigma*ds)**2)*max_v**2)) is True:  #check the stability condition
        #initial condition applied
        for i in range(0, Ns+1):
            for j in range(0, Nv+1):
                u[0,i,j] = max((min_s+(i*ds)-K),0)  #initial condition same as black scholes
        
        
        for t in range(0, Nt):
            #u[t+1,0,:] = u[t+1,:,0] = u[t+1,Ns-1,:] = u[t+1,:,Nv-1] = 0 #boundary condition applied
            
            for i in range(1, Ns):
                si = min_s + i*ds  #calculate each price in the node of the mesh
                u[t+1,i,Nv] = si    #the boundary condition for right (max vol, increasing stock price)
                
                if i != Ns:  #the boundary condition for left (min vol, increasing stock price)
                    u[t+1,i,0] = u[t,i,0] + dt*kappa*theta*(u[t,i,1] - u[t,i,0])/dv
                    
                for j in range(1, Nv):  
                    vj = min_v + j*dv  #calculate each vol in the node of the mesh
                    
                    if j != 0:   #the boundary condition for bottom (min stock price, increasing vol)
                        u[t+1,0,j] = 0 
    
                    if ((i != 0) or (i != Ns) or (j != 0) or (j != Nv)): # if the node is not in the corner, then we calculate
                        
                        a1 = 1 - (dt*(vj*si)**2)/(ds**2) - (dt*(vj*sigma)**2)/(dv**2)
                        a2 = (dt*(vj*sigma)**2)/(2*dv**2) + dt*kappa*(theta - vj)/(2*dv)
                        a3 = (dt*(vj*sigma)**2)/(2*dv**2) - dt*kappa*(theta - vj)/(2*dv)
                        a4 = (dt*(vj*si)**2)/(2*ds**2)
                        a5 = (dt*(vj*si)**2)/(2*ds**2)
                        a6 = (sigma*rho*dt*si*vj**2)/(4*ds*dv)
                        u[t+1,i,j] = a1*u[t,i,j] + a2*u[t,i,j+1] + a3*u[t,i,j-1] + a4*u[t,i+1,j] + a5*u[t,i-1,j] + a6*(u[t,i+1,j+1]-u[t,i+1,j-1]-u[t,i-1,j+1]+u[t,i-1,j-1])
    
                    if i == Ns-1:  #the boundary condition for top (max stock price, increasing vol)
                        u[t+1,Ns,j] = ds + u[t+1,Ns-1,j]
                        
        u[-1, -1, 0] = u[-1, -1, 1]
        u[-1, -1, -1] = max_s
        
        return u[-1]
    else:
        return np.zeros(1)
    
    
    
def RelativeError(result1, result2):
    #print(len(result1), len(result1[0]), len(result2), len(result2[0]))
    abserror = np.amax(abs(result2 - result1))
    relative_error = np.amax(abs(result2 - result1)/ abs(result1))
    return abserror, relative_error



def ErrorCalculation(base_N, max_N, c, count):
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
    final_c = c
    final_count = count
    final_N = base_N
    pre_result = SABR_solver(N, c*(N**2), c)
    if pre_result.any() != False:
        N *= 2
    else:
        count += 1
        return ErrorCalculation(N, max_N, c*2, count)
        
    while (N <= max_N):

        Nt = c*(N**2)
        new_result = SABR_solver(N, Nt, c)
        if new_result.any() != False:
            comparable_result = np.array([[new_result[i][j] for i in range(0,len(new_result),2)] for j in range(0,len(new_result[0]),2)])

            abs_error = RelativeError(comparable_result, pre_result)[0]
            
            error_list.append(abs_error)
            
            pre_result = new_result


        else:
            count += 1
            return ErrorCalculation(N, max_N, c*2, count)
    
        #increase the N to next level
        N *= 2
    
    return error_list, final_c, final_count, final_N
        
    
    

#all parameters large than 0
max_N = 150
N = 16
c = 1
count = 1;
#e1,e2,e3,e4 #abs

#log2(e2/e1)  close to 1
#log2(e3/e2)    close to 1
#log2(e4/e3)    close to 1

result = ErrorCalculation(N, max_N, c, count)
f = open("SABR_error", 'w')
f.write("error_list = {}\n".format(result[0]))
f.write("final_c = {}\n".format(result[1]))
f.write("final_count = {}\n".format(result[2]))
f.write("final_N = {}\n".format(result[3]))
f.close()
