import random , copy , sys 
import numpy as np 


def PSO(fitness ,tmax , N ,Dim , LB , UB ):
    PlotYs_PSO = []
    class Particle:
        def __init__(self):
            self.rnd = random.Random()

            # initialize position of the particle with 0.0 value
            self.position = [0.0 for i in range(Dim)]

            # initialize velocity of the particle with 0.0 value
            self.velocity = [0.0 for i in range(Dim)]

            # initialize best particle position of the particle with 0.0 value
            self.best_part_pos = [0.0 for i in range(Dim)]

            # loop Dim times to calculate random position and velocity
            # range of position and velocity is [LB, max]
            for i in range(Dim):
                self.position[i] = ((UB - LB) *
                self.rnd.random() + LB)
                self.velocity[i] = ((UB - LB) *
                self.rnd.random() + LB)

            # compute fitness of particle
            self.fitness = fitness(np.array(self.position)) # curr fitness

            # initialize best position and fitness of this particle
            self.best_part_pos = copy.copy(self.position) 
            self.best_part_fitnessVal = self.fitness # best fitness

        # particle swarm optimization function
    # def pso(fitness, tmax, N, Dim):
    # hyper parameters
    w = 0.729 # inertia
    c1 = 1.49445 # cognitive (particle)
    c2 = 1.49445 # social (swarm)

    rnd = random.Random()

    # create n random particles
    swarm = [Particle() for i in range(N)] 

    # compute the value of best_position and best_fitness in swarm
    best_swarm_pos = [0.0 for i in range(Dim)]
    best_swarm_fitnessVal = sys.float_info.max # swarm best

    # computer best particle of swarm and it's fitness
    for i in range(N): # check each particle
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position) 

    # main loop of pso
    Iter = 0
    while Iter < tmax:
        # print(f"Iter {Iter}, fitness={best_swarm_fitnessVal}")
        PlotYs_PSO.append(best_swarm_fitnessVal)

        for i in range(N): # process each particle
            # compute new velocity of curr particle
            for k in range(Dim): 
                r1 = rnd.random() # randomizations
                r2 = rnd.random()
            
                swarm[i].velocity[k] = ( 
                                        (w * swarm[i].velocity[k]) +
                                        (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) +
                                        (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k])) 
                                    ) 


                # if velocity[k] is not in [LB, max]
                # then clip it 
                if swarm[i].velocity[k] < LB:
                    swarm[i].velocity[k] = LB
                elif swarm[i].velocity[k] > UB:
                    swarm[i].velocity[k] = UB


            # compute new position using new velocity
            for k in range(Dim): 
                swarm[i].position[k] += swarm[i].velocity[k]

            # compute fitness of new position
            swarm[i].fitness = fitness(np.array(swarm[i].position))

            # is new position a new best for the particle?
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # is new position a new best overall?
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)
        
        # for-each particle
        Iter += 1
    #end_while
    print(f"Pso: Fitness {best_swarm_fitnessVal}")
    print(f"Pso: Position {best_swarm_pos}")
    return PlotYs_PSO
    # end pso