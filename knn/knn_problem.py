"""
KNN modeli için meta sezgisel algoritmalar için amaç fonksiyonu.
"""

from mealpy import Problem
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class KnnMetaHeuristicProblem(Problem):
    """
    KNN modeli için meta sezgisel algoritmalar için amaç fonksiyonu ve problem tanımları.
    """

    def __init__(self, attributes: pd.DataFrame, target: pd.Series, bounds: list[dict] = None, **kwargs):
        super().__init__(bounds, minmax="max", **kwargs)
        self.attributes = attributes
        self.target = target

    def obj_func(self, design_parameter: np.ndarray) -> float:
        """
        KNN modeli eğitilir ve test verisi üzerinde tahmin yapılır.
        Meta sezgisel algoritmalar için amaç fonksiyonu olarak kullanılır.
        """

        try:
            design_parameter = self.decode_solution(design_parameter)

            # Veri seti eğitim ve test olarak ayrılır
            x_train, x_test, y_train, y_test = train_test_split(
                self.attributes, self.target, 
                test_size=design_parameter["test_size"], 
                random_state=42,
            )
            
            # Veri seti normalize edilir
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # KNN modeli oluşturulur
            model = KNeighborsClassifier(
                n_neighbors=design_parameter["n_neighbors"], 
                weights=design_parameter["weights"], 
                algorithm=design_parameter["algorithm"], 
                leaf_size=design_parameter["leaf_size"], 
                p=design_parameter["p"], 
                metric=design_parameter["metric"]
            )

            # Model eğitilir ve tahmin yapılır
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            return accuracy_score(y_test, y_pred)
        except Exception as e:
            raise Exception(f"Error in knn_objective_function: {e}")