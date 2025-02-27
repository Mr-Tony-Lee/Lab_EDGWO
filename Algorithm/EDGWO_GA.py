import math, numpy as np , random
from Setting import rnd 

def edgwo_ga (fitness_function, tmax, N , Dim , LB , UB ):
    dim = Dim    
    UB = 100        
    LB = -100
    slope = 10e-5
    INITIAL_LEARNING_RATE = 1
    LEARNING_RATE_REWARD = 1.5
    LEARNING_RATE_PUNISH = 0.5

    class Data:
        def __init__(self):
            self.Fa = self.Fb = self.Fc = float("inf")
            self.Xa = self.Xb = self.Xc = np.zeros(dim)
            self.ia = self.ib = self.ic = -1
            self.PlotYs = []
            self.Xs = np.zeros((N,dim))
            self.best = float("inf")
            self.to_gradient = 0
            self.learning_rate = INITIAL_LEARNING_RATE

            for i in range(N):
                for j in range(dim):
                    self.Xs[i,j] = LB+rnd.random()*(UB-LB)

        def fitness_func(self, func):
            for i in range(N):
                fitness = func(self.Xs[i])
                if fitness < self.Fa:
                    self.ic, self.Fc, self.Xc = self.ib, self.Fb, self.Xb.copy()
                    self.ib, self.Fb, self.Xb = self.ia, self.Fa, self.Xa.copy()
                    self.ia, self.Fa, self.Xa = i, fitness, self.Xs[i].copy()
                elif fitness < self.Fb:
                    self.ic, self.Fc, self.Xc = self.ib, self.Fb, self.Xb.copy()
                    self.ib, self.Fb, self.Xb = i, fitness, self.Xs[i].copy()
                elif fitness < self.Fc:
                    self.ic, self.Fc, self.Xc = i, fitness, self.Xs[i].copy()
            self.best = func(self.Xa)

        def algorithm_func(self, t, func):
            self.PlotYs.append(func(self.Xa))
            Xm = (np.sum(self.Xs,axis=0))/N
            a = 2-t*2/tmax
            for i in range(N):
                A1, A2, A3 = 2*a*rnd.random()-a, 2*a*rnd.random()-a, 2*a*rnd.random()-a
                C1, C2, C3 = 2*rnd.random(), 2*rnd.random(), 2*rnd.random()
                p = rnd.random()
                X1 = X2 = X3 = object()
                if(abs(A1)<(0.2+0.5*(t/tmax))):
                    Da = abs(C1*self.Xa-self.Xs[i])
                    X1 = self.Xa-A1*Da
                else:
                    X1 = (self.Xa-Xm)-rnd.random()*(LB+rnd.random()*(UB-LB))
                
                if(abs(A2)<(0.2+0.5*(t/tmax))):
                    Db = abs(C2*self.Xb-self.Xs[i])
                    X2 = self.Xb-A2*Db
                else:
                    X2 = (self.Xb-Xm)-rnd.random()*(LB+rnd.random()*(UB-LB))

                if(abs(A3)<(0.2+0.5*(t/tmax))):
                    Dc = abs(C3*self.Xc-self.Xs[i])
                    X3 = self.Xc-A3*Dc
                else:
                    X3 = (self.Xc-Xm)-rnd.random()*(LB+rnd.random()*(UB-LB))

                if(p<0.2+(0.4*(t/tmax))):
                    if(i==self.ia or i==self.ib or i==self.ic):
                        Xnew = X1*(3/6)+X2*(2/6)+X3*(1/6)
                        if(func(self.Xs[i])>func(Xnew)):
                            self.Xs[i] = Xnew
                    else:
                        self.Xs[i] = X1*(3/6)+X2*(2/6)+X3*(1/6)
                else:
                    if(i==self.ia or i==self.ib or i==self.ic):
                        l = -1+2*rnd.random()
                        Xnew = self.Xa + abs(self.Xa-self.Xs[i])*math.exp(l)*math.cos(2*math.pi*l)
                        if(func(self.Xs[i])>func(Xnew)):
                            self.Xs[i] = Xnew
                    else:
                        l = -1+2*rnd.random()
                        self.Xs[i] = self.Xa + abs(self.Xa-self.Xs[i])*math.exp(l)*math.cos(2*math.pi*l)

            self.fitness_func(func)
            if(t >= 200 and abs(self.PlotYs[t]-self.PlotYs[t-20]) <= slope):
                self.to_gradient = 1
        
        def gradient(self, position, func):
            epsilon = 10**(-6)
            grad = np.zeros(dim)
            temp = np.zeros(dim)
            for i in range(dim):
                temp[i-1] = 0
                temp[i] = epsilon
                grad[i] = (func(position+temp)-func(position-temp))/(2*epsilon)
            return grad
        
        def Gradient_descent(self, position, func):
            grad = self.gradient(position, func)
            new_position = position - self.learning_rate*grad
            if(func(new_position)<func(position)):
                self.best = func(new_position)
                position = new_position
                self.learning_rate *= LEARNING_RATE_REWARD
            else:
                self.learning_rate *= LEARNING_RATE_PUNISH
            self.PlotYs.append(func(position))
            self.Xa = position

    def EDGWO(func, N, tmax, Dim):
        t = 0
        reInit = 0
        data = []
        best_now = []
        best = float("inf")

        while(t < tmax):
            data.append(Data())
            data[reInit].fitness_func(func)

            if(reInit != 0):
                data[reInit].PlotYs = data[reInit-1].PlotYs.copy()

            while(data[reInit].to_gradient == 0 and t < tmax):
                for i in range(reInit+1):
                    if(data[i].to_gradient == 0):
                        data[i].algorithm_func(t, func)
                    else:
                        data[i].Gradient_descent(data[i].Xa, func)
                    best = min(best, data[i].best)
                best_now.append(best)
                t += 1
            reInit += 1

        return best_now

    return EDGWO(fitness_function, N, tmax, Dim)
