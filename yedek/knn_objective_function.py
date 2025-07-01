"""Meta sezgisel algoritmalar için amaç fonksiyonu olarak kullanılır."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def knn_objective_function(
        attributes: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2, 
        random_state: int = 42, 
        n_neighbors: int = 42, 
        weights: str = 'uniform', 
        algorithm: str = 'auto', 
        leaf_size: int = 30, 
        p: int = 2, 
        metric: str = 'euclidean') -> float:
    """
    KNN modeli eğitilir ve test verisi üzerinde tahmin yapılır.
    Meta sezgisel algoritmalar için amaç fonksiyonu olarak kullanılır.
    """

    try:
        # Veri seti eğitim ve test olarak ayrılır
        x_train, x_test, y_train, y_test = train_test_split(attributes, target, test_size=test_size, random_state=random_state)
        
        # Veri seti normalize edilir
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # KNN modeli oluşturulur
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights=weights, 
            algorithm=algorithm, 
            leaf_size=leaf_size, 
            p=p, 
            metric=metric
        )

        # Model eğitilir ve tahmin yapılır
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        return accuracy_score(y_test, y_pred)
    except Exception as e:
        raise Exception(f"Error in knn_objective_function: {e}")