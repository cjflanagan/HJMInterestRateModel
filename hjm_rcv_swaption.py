# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import time
import numpy as np
import random
#import matplotlib.pyplot as plt


def hjm_rcv_swaption(y,sigma,K,j,k,n,trials,seed):
    start = time.clock() # start cpu time count
    
    L = 100 # notional amount 
    # Calculate # price to convert cash payments into today's dollars
    true_price = [0] * n 
    true_price[0] = np.exp(-y[0])
    for i in range(1,n):
        true_price[i] = true_price[i-1] * np.exp(-y[i])
        
    path_price_mean = [0] * n # I'll calculate mean path prices to check my code

    price = [0] * trials # allocate array to store swaption prices
    
    delta_price = [] # store data that will be used to calculate delta         
    delta_rate = []              
    delta_aux = [] # list of tuples              

    alpha = [0] * n  # "no-arbitrage" drifts 
    for i in range(0,n):           
        if i == 0:
            alpha[i] = (sigma[i])**2 / 2  # "no-arbitrage" drifts 
        else:
            alpha[i] = (sum(sigma[:i+1]))**2 / 2 - sum(alpha[:i])  # "no-arbitrage" drifts 
    print '\nAlpha = ', alpha
    print '\n'
        
    random.seed(seed) # Initialize internal state of the random number generator
    for tr in range(0,trials): # make 1 trial
        z = [0] * n  # Random Normal Shocks
        for i in range(0,n):           
            z[i] = random.gauss(0,1) # Random Normal Shocks

        # Calculate iF = f_1(i) - 1-year forward rates at year i
        g = [0] * n # auxiliary cumulative sum z_i*sigma_i+alpha_i
        iF = [0] * n
        iF[0] = y[0]
        path_price = [0] * n # I'll calculate mean path prices to check my code
        path_price[0] = true_price[0]
        for i in range(1,n):
            g[i] = z[i]*sigma[i] + alpha[i] + g[i-1]
            iF[i] = y[i] + g[i]
            path_price[i] = path_price[i-1] * np.exp(-iF[i])
            path_price_mean[i] += path_price[i]
            
        # Calculate the swaption price
        count = 0 # count payments
        for i in range(j,j+k): # k payoffs after the swaption matures at year j
            if K > iF[i]:   # payment is only made if (fixed rate)-(floating rate)>0
                price[tr] += (K - iF[i]) * true_price[i]
                count += 1
        if price[tr] > 0 and count == k:
            delta_aux.append( (price[tr],np.mean(iF[j:j+k])) ) # tuple
        if np.mod(tr,10000) == 0:
            print 'Trial', tr
        
    price_mean = np.mean(price) * L
    price_std = np.std(price) * L
    delta_mean = 0
    delta_std = 0
    gamma_mean = 0
    gamma_std = 0
        
    path_price_mean[0] = true_price[0]
    print '\nmean path prices:\n%.4f' % path_price_mean[0] 
    for i in range(1,n):
        path_price_mean[i] /= trials # I'll calculate mean path prices to check my code
        print '%.4f' % path_price_mean[i] 
    print '\n'
    
    # Find delta and gamma by replacing derivatives with finite differences
    for a,b in sorted(delta_aux): # sort by forward rate
        delta_rate.append(b)  # mean forward rate (variable)
        delta_price.append(a) # price
    """    
    plt.plot(delta_rate,delta_price,'r.')
    plt.xlabel('Forward rate')
    plt.ylabel('Swaption price')
    plt.show()
    """
    
    delta_x = []
    delta_y = []
    m = len( delta_rate ) # number of good data points found
    for i in range(0,m-1):
        if delta_rate[i] != delta_rate[i+1]: # find 1st derivative
            delta_x.append( (delta_rate[i+1] + delta_rate[i])/2 ) # midpoint
            delta_y.append( (delta_price[i+1] - delta_price[i]) / (delta_rate[i+1] - delta_rate[i]) )
                
    # find 2nd derivative
    gamma = []
    m = len( delta_x ) # number of good data points found
    for i in range(0,m-1):
        gamma.append( (delta_y[i+1] - delta_y[i]) / (delta_x[i+1] - delta_x[i]) )
            
    delta_mean = np.mean(delta_y)
    delta_std  = np.std(delta_y)
    gamma_mean = np.mean(gamma)
    gamma_std  = np.std(gamma)
                      
    elapsed = (time.clock() - start) # finish cpu time count
    return(price_mean,price_std,delta_mean,delta_std,gamma_mean,gamma_std,elapsed)


"""
(b) n = 10
"""
n = 10
y = [0.05] * (n+1)   # the yield of a zero-coupon bond maturing at time i years
j = 3 # the number of years until the maturity of the swaption (the swaption matures at year j)
k = 5 # determines the tenor of the swap: the last payment of the swap is at year j + k

"""
(c) n = 30
"""
#n = 30
#y = [0.05] * (n+1)   # the yield of a zero-coupon bond maturing at time i years
#y[0] = 0.045
#y[1] = 0.0475
#j = 5 # the number of years until the maturity of the swaption (the swaption matures at year j)
#k = 25 # determines the tenor of the swap: the last payment of the swap is at year j + k

"""
the parameters below are the same for (b) and (c)
"""
sigma = [0.01] * n   # vector with n elements representing the HJM annual volatility factor
K = 0.05 # the strike of the receiver swaption (with simple compounding). 
trials = 100000  # number of simulation trials
seed = 5667 # random number seed

(price_mean,price_std,delta_mean,delta_std,gamma_mean,gamma_std,elapsed) = hjm_rcv_swaption(y,sigma,K,j,k,n,trials,seed)
print 'Price = $%.4f' % price_mean, '+/- %.4f' % price_std
print 'Delta = %.4f' % delta_mean, '+/- %.4f' % delta_std
print 'Gamma = %.4f' % gamma_mean, '+/- %.4f' % gamma_std
print '\nTime elapsed:', elapsed, 'sec'

