import random 
import numpy as np  


def gwo(fitness_function , tmax , N , Dim , LB , UB ):
    PlotYs_GWO = []
    Xs = np.zeros((N,Dim))
    for i in range(N):
        for j in range(Dim):
            Xs[i,j] = LB+random.Random().random()*(UB-LB)
    Fa = Fb = Fc = float("inf")
    Xa = Xb = Xc = np.zeros(Dim)
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
    best = fitness_function(Xa)
    for t in range(tmax):
        PlotYs_GWO.append(fitness_function(Xa))
        a = 2-t*2/tmax
        for i in range(N):
            A1, A2, A3 = 2*a*random.Random().random()-a, 2*a*random.Random().random()-a, 2*a*random.Random().random()-a
            C1, C2, C3 = 2*random.Random().random(), 2*random.Random().random(), 2*random.Random().random()
            Da = abs(C1*Xa-Xs[i])
            Db = abs(C2*Xb-Xs[i])
            Dc = abs(C3*Xc-Xs[i])
            X1 = Xa-A1*Da
            X2 = Xb-A2*Db
            X3 = Xc-A3*Dc
            Xs[i] = (X1+X2+X3)/3
            
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
        best = min(best,fitness_function(Xa))
    print(f"Gwo: Fitness {best}")
    print(f"Gwo: Position {Xa}")
    return PlotYs_GWO
