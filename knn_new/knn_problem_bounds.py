"""
Geliştirilmiş metaheuristik algoritmalarda kullanılan amaç fonksiyonu için sınır değerleri.
Cross-validation ve multi-objective optimization için optimize edilmiş.
"""

from mealpy import StringVar, IntegerVar, FloatVar

# Ana problem bounds - Cross-validation kullanırken test_size gereksiz
problem_bounds_cv: list[dict] = [
    # KNN Hiperparametreleri - Cross-validation için
    IntegerVar(lb=3, ub=50, name="n_neighbors"),  # Daha makul aralık
    IntegerVar(lb=10, ub=50, name="leaf_size"),   # Daha geniş aralık
    IntegerVar(lb=1, ub=3, name="p"),             # 1=Manhattan, 2=Euclidean, 3=Minkowski
    
    # String Ayrık değerler
    StringVar(valid_sets=('uniform', 'distance'), name="weights"),
    StringVar(valid_sets=('auto', 'ball_tree', 'kd_tree', 'brute'), name="algorithm"),
    StringVar(valid_sets=('euclidean', 'manhattan', 'minkowski'), name="metric"),
]

# Geleneksel train/test split için bounds
problem_bounds_split: list[dict] = [
    # Test verisi boyutu
    FloatVar(lb=0.15, ub=0.25, name="test_size"),  # Daha makul aralık
    
    # KNN Hiperparametreleri
    IntegerVar(lb=3, ub=50, name="n_neighbors"),
    IntegerVar(lb=10, ub=50, name="leaf_size"),
    IntegerVar(lb=1, ub=3, name="p"),
    
    # String değerler
    StringVar(valid_sets=('uniform', 'distance'), name="weights"),
    StringVar(valid_sets=('auto', 'ball_tree', 'kd_tree', 'brute'), name="algorithm"),
    StringVar(valid_sets=('euclidean', 'manhattan', 'minkowski'), name="metric"),
]

# Hızlı test için küçük bounds
problem_bounds_quick: list[dict] = [
    # Hızlı test için daha dar aralık
    IntegerVar(lb=3, ub=15, name="n_neighbors"),
    IntegerVar(lb=10, ub=30, name="leaf_size"),
    IntegerVar(lb=1, ub=2, name="p"),  # Sadece Manhattan ve Euclidean
    
    StringVar(valid_sets=('uniform', 'distance'), name="weights"),
    StringVar(valid_sets=('auto', 'brute'), name="algorithm"),  # Hızlı algoritmalar
    StringVar(valid_sets=('euclidean', 'manhattan'), name="metric"),
]

# Multi-objective optimization için farklı termination criteria
termination_cv: dict[str, int | float] = {
    "max_epoch": 50,         # CV kullanırken daha az epoch
    "max_fe": 5000,          # Daha az function evaluation
    "max_early_stop": 15,    # Erken durdurmada daha az sabır
    "epsilon": 1e-6,         # Daha hassas convergence
}

termination_standard: dict[str, int | float] = {
    "max_epoch": 100,        # Orijinal değerler
    "max_fe": 20000,
    "max_early_stop": 20,
    "epsilon": 1e-8,
}

termination_quick: dict[str, int | float] = {
    "max_epoch": 25,         # Hızlı test için
    "max_fe": 2500,
    "max_early_stop": 10,
    "epsilon": 1e-5,
}

# Farklı complexity weight'ler için predefined configs
COMPLEXITY_CONFIGS = {
    'accuracy_focused': {
        'complexity_weight': 0.01,  # Çok az penalty
        'description': 'Sadece accuracy odaklı'
    },
    'balanced': {
        'complexity_weight': 0.1,   # Dengeli
        'description': 'Accuracy ve complexity dengeli'
    },
    'efficiency_focused': {
        'complexity_weight': 0.2,   # Yüksek penalty
        'description': 'Computational efficiency odaklı'
    }
}

# Algoritma önerileri farklı problem boyutları için
ALGORITHM_RECOMMENDATIONS = {
    'small_dataset': {  # < 1000 samples
        'bounds': problem_bounds_quick,
        'termination': termination_quick,
        'cv_folds': 3,
        'complexity_config': 'efficiency_focused'
    },
    'medium_dataset': {  # 1000-10000 samples
        'bounds': problem_bounds_cv,
        'termination': termination_cv,
        'cv_folds': 5,
        'complexity_config': 'balanced'
    },
    'large_dataset': {  # > 10000 samples
        'bounds': problem_bounds_split,  # CV çok yavaş olabilir
        'termination': termination_standard,
        'cv_folds': 3,
        'complexity_config': 'efficiency_focused'
    }
}

def get_recommended_config(dataset_size: int) -> dict:
    """
    Dataset boyutuna göre önerilen konfigürasyonu döndür.
    """
    if dataset_size < 1000:
        config_key = 'small_dataset'
    elif dataset_size < 10000:
        config_key = 'medium_dataset'
    else:
        config_key = 'large_dataset'
    
    config = ALGORITHM_RECOMMENDATIONS[config_key].copy()
    complexity_key = config['complexity_config']
    config['complexity_weight'] = COMPLEXITY_CONFIGS[complexity_key]['complexity_weight']
    
    return config

def get_bounds_for_strategy(strategy: str = 'cv') -> list[dict]:
    """
    Strateji türüne göre bounds döndür.
    
    Args:
        strategy: 'cv', 'split', veya 'quick'
    """
    if strategy == 'cv':
        return problem_bounds_cv
    elif strategy == 'split':
        return problem_bounds_split
    elif strategy == 'quick':
        return problem_bounds_quick
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def get_termination_for_strategy(strategy: str = 'cv') -> dict:
    """
    Strateji türüne göre termination criteria döndür.
    """
    if strategy == 'cv':
        return termination_cv
    elif strategy == 'split':
        return termination_standard
    elif strategy == 'quick':
        return termination_quick
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# Backward compatibility için
problem_bounds = problem_bounds_cv
termination = termination_cv 