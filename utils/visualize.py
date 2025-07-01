from matplotlib import pyplot as plt
from mealpy.optimizer import Optimizer

def plot_fitness(model: Optimizer, title: str = "Fitness") -> None:
    """
    Metaheuristik algoritmaların performansını görselleştirmek için kullanılan sınıf.
    """
        
    print(f"Best accuracy: {model.g_best.target.fitness}")
    print(f"Best parameters: \n{model.problem.decode_solution(model.g_best.solution)}")

    try:
        plt.plot(model.history.list_global_best_fit)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.grid(True)
        plt.show()
    except Exception as e:
        raise Exception(e)
