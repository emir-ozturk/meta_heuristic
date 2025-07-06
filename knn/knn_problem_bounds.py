"""
Metaheuristik algoritmalarda kullanılan amaç fonksiyonu için sınır değerleri.
"""

from mealpy import StringVar, IntegerVar, FloatVar

problem_bounds: list[dict] = [
    # Hiperparametreler
    # Sürekli değerler için kullanılır
    FloatVar(lb=0.1, ub=0.3, name="test_size"),

    # KNN Hiperparametreleri
    # Ayrık değerler için kullanılır
    IntegerVar(lb=1, ub=100, name="n_neighbors"),
    IntegerVar(lb=1, ub=5, name="leaf_size"),
    IntegerVar(lb=1, ub=5, name="p"),

    # String Ayrık değerler için kullanılır
    StringVar(valid_sets=('uniform', 'distance'), name="weights"),
    StringVar(valid_sets=('auto', 'ball_tree', 'kd_tree', 'brute'), name="algorithm"),
    StringVar(valid_sets=('euclidean', 'manhattan', 'minkowski'), name="metric"),
]

termination: dict[str, int | float] = {
    "max_epoch": 100,        # Maksimum iterasyon sayısı
    "max_fe": 20000,         # Maksimum amaç fonksiyonu değerlendirmesi
    # "max_time": 60.0,        # Maksimum çalışma süresi (saniye)
    "max_early_stop": 10,    # İyileşme olmadan durma eşiği (epoch)
    "epsilon": 1e-8,         # Erken durdurmada minimum iyileşme eşiği
}
