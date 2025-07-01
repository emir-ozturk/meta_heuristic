from matplotlib import pyplot as plt
from mealpy.optimizer import Optimizer

class MetaHeuristicVisualize:
    """
    Metaheuristik algoritmaların performansını görselleştirmek için kullanılan sınıf.
    """

    @staticmethod
    def plot(model: Optimizer, title: str = "Fitness"):
        print(f"Best accuracy: {model.g_best.target.fitness}")
        print(f"Best parameters: \n{model.problem.decode_solution(model.g_best.solution)}")

        plt.plot(model.history.list_global_best_fit)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.grid(True)
        plt.show()
