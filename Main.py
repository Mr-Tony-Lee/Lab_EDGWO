#-------------------------------------------Algorithm area------------------------------------------------
from Algorithm.EDGWO import edgwo
from Algorithm.EDGWO_GA import edgwo_ga
from Algorithm.EDGWO_GB import edgwo_gb
from Algorithm.EDGWO_G import edgwo_g
from Algorithm.PSO import PSO
from Algorithm.Gwo import gwo
#-------------------------------------------Algorithm area------------------------------------------------

#-------------------------------------------import area------------------------------------------------
import Setting as Setting , numpy as np , matplotlib.pyplot as plt , os , CEC2022 
#-------------------------------------------import area------------------------------------------------

#-------------------------------------------CSC2022 Function area------------------------------------------------
def CEC2022_f1(pos):    return CEC2022.Zakharov(pos + Setting.Shift_Val_X)
def CEC2022_f2(pos):    return CEC2022.Rosenbrock(pos + Setting.Shift_Val_X)
def CEC2022_f3(pos):    return CEC2022.Expanded_Schaffer(pos + Setting.Shift_Val_X)
def CEC2022_f4(pos):    return CEC2022.Rastrigin(pos + Setting.Shift_Val_X)
def CEC2022_f5(pos):    return CEC2022.Levy(pos + Setting.Shift_Val_X)
def CEC2022_f6(pos):    return CEC2022.Bent_Cigar(pos + Setting.Shift_Val_X)
def CEC2022_f7(pos):    return CEC2022.High_Conditioned_Elliptic(pos + Setting.Shift_Val_X)
def CEC2022_f8(pos):    return CEC2022.HGBat(pos + Setting.Shift_Val_X)
def CEC2022_f9(pos):    return CEC2022.Katsuura(pos + Setting.Shift_Val_X)
def CEC2022_f10(pos):   return CEC2022.Happycat(pos + Setting.Shift_Val_X)
#-------------------------------------------CSC2022 area------------------------------------------------

#-------------------------------------------Execute area------------------------------------------------
import concurrent.futures
def execute_single_run(fitness_function, max_iter, N, dim, LB, UB, run_index):
    print(f"----------------------------------Now Execution Times : {run_index+1}----------------------------------")
    all_PlotYs = []
    print("\nGwo Start...\n")
    all_PlotYs.append(gwo(fitness_function, max_iter, N , dim , LB , UB))

    print("\nedgwo Start...\n")
    all_PlotYs.append(edgwo(fitness_function, max_iter, N , dim , LB , UB))

    print("\nPSO Start...\n")
    all_PlotYs.append(PSO(fitness_function, max_iter, N , dim , LB , UB))

    print("\nedgwo_g Start...\n")
    all_PlotYs.append(edgwo_g(fitness_function, max_iter, N , dim , LB , UB))

    print("\nedgwo_ga Start...\n")
    all_PlotYs.append(edgwo_ga(fitness_function, max_iter, N , dim , LB , UB))

    print("\nedgwo_gb Start...\n")
    all_PlotYs.append(edgwo_gb(fitness_function, max_iter, N , dim , LB , UB))

    all_PlotYs = np.array(all_PlotYs).reshape(len(os.listdir("Algorithm"))-1,Setting.max_iter)
    return all_PlotYs

def execute(fitness_function, max_iter, N, dim, LB, UB):
    all_PlotYs = np.zeros((len(os.listdir("Algorithm"))-1,Setting.max_iter))
    print(f"------------------------------Now {fitness_function.__name__} Start ------------------------------")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(execute_single_run, fitness_function, max_iter, N, dim, LB, UB, i)
            for i in range(Setting.exe_Val)
        ]
        for future in concurrent.futures.as_completed(futures):
            Py = future.result()  # 確保每個執行緒完成
            all_PlotYs += Py
    all_PlotYs /= Setting.exe_Val
    draw(fitness_function, all_PlotYs)
    print(f"------------------------------Now {fitness_function.__name__} End ------------------------------")
    # input("Press Enter to Go next Function...")
#-------------------------------------------Execute area------------------------------------------------

#-------------------------------------------Draw area------------------------------------------------
def draw(fitness_function, all_PlotY):
    plt.figure(figsize = (10.8, 10.8), dpi = 100) # 1080 * 1080
    plt.title(fitness_function.__name__)
    plt.xlabel("Number of iteration")  
    plt.ylabel("Average Function Values (log10)")  
    PlotXs = np.arange(0,Setting.max_iter)
    line1, = plt.plot(PlotXs, np.log10(all_PlotY[1]) , color = 'red' , linewidth = 1, label = 'edgwo')   
    line2, = plt.plot(PlotXs, np.log10(all_PlotY[3]) , color = 'purple' , linewidth = 1, label = 'edgwo_g')          
    line3, = plt.plot(PlotXs, np.log10(all_PlotY[4]) , color = 'black'   , linewidth = 1, label = 'edgwo_ga')    
    line4, = plt.plot(PlotXs, np.log10(all_PlotY[5]) , color = 'blue'  , linewidth = 1, label = 'edgwo_gb')   
    line5, = plt.plot(PlotXs, np.log10(all_PlotY[0]) , color = 'orange'   , linewidth = 1, label = 'GWO'      )           
    line6, = plt.plot(PlotXs, np.log10(all_PlotY[2]) , color = 'green'   , linewidth = 1, label = 'PSO'      )      
    plt.legend(handles = [line1, line2, line3 , line4 , line5 , line6 ], loc = 'upper right')
    # plt.show()
    if Setting.Shift_Val_X == 0 :
        plt.savefig(os.path.join("photo" , f"{fitness_function.__name__}.png"))
    else:
        plt.savefig(os.path.join("photo_Shift" , f"{fitness_function.__name__}.png"))
    plt.clf(); plt.cla()  
    plt.close()
#-------------------------------------------Draw area------------------------------------------------


#-------------------------------------------Main area------------------------------------------------
if __name__ == '__main__':
    print(f"Now Setting: N->{Setting.num_particles}\tTmax->{Setting.max_iter}\tDim->{Setting.dim} \t Shift Value->{Setting.Shift_Val_X}")
    execute(CEC2022_f1  , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    execute(CEC2022_f2  , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    execute(CEC2022_f3  , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    execute(CEC2022_f4  , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    execute(CEC2022_f5  , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    execute(CEC2022_f6  , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    execute(CEC2022_f7  , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    execute(CEC2022_f8  , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    execute(CEC2022_f9  , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    execute(CEC2022_f10 , Setting.max_iter , Setting.num_particles , Setting.dim , Setting.LB , Setting.UB)
    print("Program End ...")
#-------------------------------------------Main area------------------------------------------------