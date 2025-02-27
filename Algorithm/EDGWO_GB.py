import math, numpy as np 
from Setting import rnd 

def gradient(fitness_function,position, Dim ):
    epsilon = 10**(-6)
    grad = np.zeros(Dim)
    temp = np.zeros(Dim)
    for i in range(Dim):
        temp[i-1] = 0
        temp[i] = epsilon
        grad[i] = (fitness_function(position+temp)-fitness_function(position-temp))/(2*epsilon)
    return grad

def edgwo_gb (fitness_function, tmax, N , Dim , LB , UB ):
    EDGWO_T = 0.8               #經過多久比例的時間後執行梯度下降改
    INITIAL_LEARNING_RATE = 1
    LEARNING_RATE_REWARD = 1.3
    LEARNING_RATE_PUNISH = 0.6

    PlotYs_EDGWO6 = []
    print(f"START EDGWO6, N:{N} tmax:{tmax} Dim:{Dim}")
    Xs = np.zeros((N,Dim))
    for i in range(N):
        for j in range(Dim):
            Xs[i,j] = LB+rnd.random()*(UB-LB)
    Fa = Fb = Fc = float("inf")
    Xa = Xb = Xc = np.zeros(Dim)
    ia = ib = ic = -1
    learning_rate = INITIAL_LEARNING_RATE
    for i in range(N):
        fitness = fitness_function(Xs[i])
        if fitness < Fa:
            ic,Fc,Xc = ib,Fb,Xb.copy()
            ib,Fb,Xb = ia,Fa,Xa.copy()
            ia,Fa,Xa = i,fitness,Xs[i].copy()
        elif fitness < Fb:
            ic,Fc,Xc = ib,Fb,Xb.copy()
            ib,Fb,Xb = i,fitness,Xs[i].copy()
        elif fitness < Fc:
            ic,Fc,Xc = i,fitness,Xs[i].copy()
    best = fitness_function(Xa)
    count = 0 
    for t in range(int(tmax*EDGWO_T)):
        PlotYs_EDGWO6.append(fitness_function(Xa))
        Xm = (np.sum(Xs,axis=0))/N
        a = 2-t*2/tmax
        Fre_Xa , Fre_Fa = Xa , Fa 
        for i in range(N):
            A1, A2, A3 = 2*a*rnd.random()-a, 2*a*rnd.random()-a, 2*a*rnd.random()-a
            C1, C2, C3 = 2*rnd.random(), 2*rnd.random(), 2*rnd.random()
            p = rnd.random()
            X1 = X2 = X3 = object()
            if(abs(A1)<(0.2+0.5*(t/tmax))):
                Da = abs(C1*Xa-Xs[i])
                X1 = Xa-A1*Da
            else:
                X1 = (Xa-Xm)-rnd.random()*(LB+rnd.random()*(UB-LB))
            
            if(abs(A2)<(0.2+0.5*(t/tmax))):
                Db = abs(C2*Xb-Xs[i])
                X2 = Xb-A2*Db
            else:
                X2 = (Xb-Xm)-rnd.random()*(LB+rnd.random()*(UB-LB))

            if(abs(A3)<(0.2+0.5*(t/tmax))):
                Dc = abs(C3*Xc-Xs[i])
                X3 = Xc-A3*Dc
            else:
                X3 = (Xc-Xm)-rnd.random()*(LB+rnd.random()*(UB-LB))

            if(p<0.2+(0.4*(t/tmax))):
                if(i==ia or i==ib or i==ic):
                    Xnew = X1*(3/6)+X2*(2/6)+X3*(1/6)
                    if(fitness_function(Xs[i])>fitness_function(Xnew)):
                        Xs[i] = Xnew
                else:
                    Xs[i] = X1*(3/6)+X2*(2/6)+X3*(1/6)
            else:
                if(i==ia or i==ib or i==ic):
                    l = -1+2*rnd.random()
                    Xnew = Xa + abs(Xa-Xs[i])*math.exp(l)*math.cos(2*math.pi*l)
                    if(fitness_function(Xs[i])>fitness_function(Xnew)):
                        Xs[i] = Xnew
                else:
                    l = -1+2*rnd.random()
                    Xs[i] = Xa + abs(Xa-Xs[i])*math.exp(l)*math.cos(2*math.pi*l)
        for i in range(N):
            fitness = fitness_function(Xs[i])
            if fitness < Fa:
                Fc,Xc = Fb,Xb.copy()
                Fb,Xb = Fa,Xa.copy()
                Fa,Xa = fitness,Xs[i].copy()
            elif fitness < Fb:
                Fc,Xc = Fb,Xb.copy()
                Fb,Xb = fitness,Xs[i].copy()
            elif fitness < Fc:
                Fc,Xc = fitness,Xs[i].copy()
        
        best = min(best,Fa)    

        if( Fa == Fre_Fa ):
            count += 1   
        
        if( count == 5 ):
            grad = gradient(fitness_function,Xa , Dim)
            new_position = Xa - learning_rate*grad
            new_F = fitness_function(new_position)
            if(best > new_F):
                Fa = new_F
                Xa = new_position
                learning_rate *= (2 - 0.5 * t/tmax)
            else:
                learning_rate *= (0.1 + 0.4 * t/tmax)
            count = 0         

    # EDGWO above
    # Gradient descent below
    def Gradient_descent(position , learning_rate ):
        GD_tmax = tmax-int(tmax * EDGWO_T)    
        learning_rate = 1 
        for t in range(GD_tmax):
            grad = gradient(fitness_function,position,Dim)
            new_position = position - learning_rate * grad
            if(fitness_function(new_position) < fitness_function(position)):
                position = new_position
                learning_rate *= LEARNING_RATE_REWARD
            else:
                learning_rate *= LEARNING_RATE_PUNISH
            PlotYs_EDGWO6.append(fitness_function(position))
        return position
    
    Xa = Gradient_descent(Xa, learning_rate)
    print(f"BEST {best}")
    print(f"LAST {Xa}")

    return PlotYs_EDGWO6
