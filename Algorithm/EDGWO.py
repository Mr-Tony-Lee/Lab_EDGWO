import numpy as np 
import copy , math , random 


# Wolf class
class Wolf:
    def __init__(self, fitness, dim, LB, UB):
        # self.position = np.random.uniform(LB, UB, dim)
        self.position = np.zeros(dim)
        for i in range(dim):
            self.position[i] = LB + (random.random() * (UB-LB))
        self.fitness = fitness(self.position)  # current fitness (評分)

#--------------------------------------------------------------------------------------------------
# Elite-driven grey wolf optimization (EDGWO)
def edgwo(fitness, max_iter, n, dim, LB, UB):
    fitness_history = []
    np.random.seed(); random.seed()
    population = [Wolf(fitness, dim, LB, UB) for _ in range(n)]
    alpha_wolf , beta_wolf , delta_wolf = population[:3]
    
    for i in range(n):
        if(alpha_wolf.fitness > population[i].fitness):
            delta_wolf , beta_wolf , alpha_wolf = copy.copy(beta_wolf), copy.copy(alpha_wolf) , copy.copy(population[i])
        elif (beta_wolf.fitness > population[i].fitness):
            delta_wolf , beta_wolf = copy.copy(beta_wolf) , copy.copy(population[i])
        elif (delta_wolf.fitness > population[i].fitness ):
            delta_wolf = copy.copy(population[i])       

    for i in range(max_iter):
        fitness_history.append(alpha_wolf.fitness)
        a = 2 - i * (2 / max_iter)
        Xm = np.mean([wolf.position for wolf in population], axis=0)

        for i in range(n):
            A1, A2, A3 = (2 * a * random.random() - a ), (2 * a * random.random() - a ), (2 * a * random.random() - a )
            C1, C2, C3 = 2 * random.random(), 2 * random.random(), 2*random.random() 
            X1 , X2 , X3  = [np.zeros(dim) for _ in range(3)]
            p = np.random.uniform(0,1)

            if abs(A1) < 1:
                X1 = alpha_wolf.position - A1 * abs(C1 * alpha_wolf.position - population[i].position)
            else:    
                X1 = (alpha_wolf.position - Xm) - random.random() * (LB + random.random() * (UB - LB))

            if abs(A2) < 1:
                X2 = beta_wolf.position - A2 * abs(C2 * beta_wolf.position - population[i].position)
            else:
                X2 = (beta_wolf.position - Xm) - random.random() * (LB + random.random() * (UB - LB))

            if abs(A3) < 1:
                X3 = delta_wolf.position - A3 * abs(C3 * delta_wolf.position - population[i].position)
            else:
                X3 = (delta_wolf.position - Xm) - random.random() * (LB + random.random() * (UB - LB))

            if p < 0.5:
                population[i].position = (X1 + X2 + X3) / 3.0
            else:
                l = -1 + 2 * random.random()
                population[i].position = alpha_wolf.position + ((abs(alpha_wolf.position - population[i].position) * math.cos(2 * math.pi * l) * math.exp(l)))

        for i in range(n):
            population[i].fitness = fitness(population[i].position)    
            
            if(alpha_wolf.fitness > population[i].fitness):
                delta_wolf , beta_wolf , alpha_wolf = copy.copy(beta_wolf) , copy.copy(alpha_wolf) , copy.copy(population[i])
            elif (beta_wolf.fitness > population[i].fitness):
                delta_wolf , beta_wolf = copy.copy(beta_wolf) , copy.copy(population[i])
            elif (delta_wolf.fitness > population[i].fitness ):
                delta_wolf = copy.copy(population[i])       
    
    print(f"EDGWO_Lee: Fitness {alpha_wolf.fitness}")
    print(f"EDGWO_Lee: Position {alpha_wolf.position}")
    return fitness_history 
#------------------------------------------------------------------------------------------------------------------------
