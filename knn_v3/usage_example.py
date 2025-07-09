"""
Geliştirilmiş KNN meta sezgisel optimizasyonu kullanım örneği.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from mealpy import PSO, StringVar, IntegerVar, FloatVar
from knn_v3.knn_problem_improved import ImprovedKnnMetaHeuristicProblem
from knn_v3.knn_penalties import KnnPenaltyCalculator
from knn_v3.mealpy_compatibility import safe_solve


def create_sample_bounds():
    """
    KNN parametreleri için Mealpy değişken tipleri ile bounds tanımlar.
    
    Returns:
        list: Mealpy için parametre sınırları
    """
    bounds = [
        # n_neighbors: 1-50 arası
        IntegerVar(lb=1, ub=50, name="n_neighbors"),
        
        # test_size: 0.1-0.4 arası
        FloatVar(lb=0.1, ub=0.4, name="test_size"),
        
        # weights: 'uniform' veya 'distance'
        StringVar(valid_sets=('uniform', 'distance'), name="weights"),
        
        # algorithm: KNN algoritması seçenekleri
        StringVar(valid_sets=('auto', 'ball_tree', 'kd_tree', 'brute'), name="algorithm"),
        
        # leaf_size: 10-100 arası (ball_tree ve kd_tree için)
        IntegerVar(lb=10, ub=100, name="leaf_size"),
        
        # p: 1-2 arası (Minkowski distance için)
        FloatVar(lb=1.0, ub=2.0, name="p"),
        
        # metric: Distance metric seçenekleri
        StringVar(valid_sets=('euclidean', 'manhattan', 'minkowski'), name="metric")
    ]
    
    return bounds


def decode_solution_example(solution: np.ndarray) -> dict:
    """
    Çözüm vektörünü KNN parametrelerine dönüştürür.
    Mealpy StringVar bazen numeric değer döndürebilir, bu durumda mapping yapılır.
    
    Args:
        solution: Optimizasyon algoritması tarafından üretilen çözüm
        
    Returns:
        dict: KNN parametreleri
    """
    # Mapping dictionaries for backwards compatibility
    weights_map = {0: 'uniform', 1: 'distance'}
    algorithm_map = {0: 'auto', 1: 'ball_tree', 2: 'kd_tree', 3: 'brute'}
    metric_map = {0: 'euclidean', 1: 'manhattan', 2: 'minkowski'}
    
    # Safe conversion functions
    def safe_weights(val):
        if isinstance(val, str):
            return val
        return weights_map.get(int(val), 'uniform')
    
    def safe_algorithm(val):
        if isinstance(val, str):
            return val
        return algorithm_map.get(int(val), 'auto')
    
    def safe_metric(val):
        if isinstance(val, str):
            return val
        return metric_map.get(int(val), 'euclidean')
    
    return {
        "n_neighbors": int(solution[0]),        # IntegerVar
        "test_size": float(solution[1]),        # FloatVar
        "weights": safe_weights(solution[2]),   # StringVar with fallback
        "algorithm": safe_algorithm(solution[3]), # StringVar with fallback
        "leaf_size": int(solution[4]),          # IntegerVar
        "p": float(solution[5]),                # FloatVar
        "metric": safe_metric(solution[6])      # StringVar with fallback
    }


def run_optimization_example():
    """
    Optimizasyon örneği çalıştırır.
    """
    print("KNN Meta Sezgisel Optimizasyon Örneği")
    print("=" * 50)
    
    # Veri seti yükle (Iris dataset)
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    print(f"Veri seti: {X.shape[0]} örnek, {X.shape[1]} öznitelik")
    print(f"Sınıf sayısı: {len(y.unique())}")
    
    # Problem tanımla
    bounds = create_sample_bounds()
    
    # Geliştirilmiş problem sınıfı
    problem = ImprovedKnnMetaHeuristicProblem(
        attributes=X,
        target=y,
        bounds=bounds,
        complexity_weight=0.15,  # Biraz daha yüksek penalty
        use_cross_validation=True,  # Cross-validation kullan
        cv_folds=5,
        verbose=True
    )
    
    # decode_solution metodunu tanımla
    problem.decode_solution = decode_solution_example
    
    # PSO algoritması
    optimizer = PSO.OriginalPSO(epoch=50, pop_size=20)
    
    print("\nOptimizasyon başlıyor...")
    
    # Optimizasyonu çalıştır - Version safe
    best_position, best_fitness = safe_solve(optimizer, problem)
    
    print("\nOptimizasyon tamamlandı!")
    print("=" * 50)
    
    # Sonuçları göster
    best_params = decode_solution_example(best_position)
    print(f"En iyi fitness: {best_fitness:.4f}")
    print(f"En iyi parametreler: {best_params}")
    
    # Optimizasyon özeti
    summary = problem.get_optimization_summary()
    print(f"\nOptimizasyon Özeti:")
    print(f"Toplam değerlendirme: {summary['total_evaluations']}")
    print(f"Veri seti boyutu: {summary['dataset_shape']}")
    print(f"Cross-validation kullanıldı: {summary['cross_validation_used']}")
    
    # Penalty breakdown
    if 'penalty_breakdown' in summary:
        print(f"\nPenalty Analizi:")
        for param, penalty in summary['penalty_breakdown'].items():
            print(f"  {param}: {penalty:.4f}")


def compare_penalty_strategies():
    """
    Farklı penalty stratejilerini karşılaştırır.
    """
    print("\nPenalty Stratejileri Karşılaştırması")
    print("=" * 50)
    
    # Örnek parametre seti
    test_params = {
        "n_neighbors": 15,
        "test_size": 0.2,
        "weights": "distance",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2.0,
        "metric": "euclidean"
    }
    
    # Farklı complexity_weight değerleri
    weights = [0.05, 0.1, 0.2, 0.3]
    
    print("Test parametreleri:", test_params)
    print("\nFarklı complexity_weight değerleri için penalty'ler:")
    
    for weight in weights:
        calculator = KnnPenaltyCalculator(complexity_weight=weight)
        total_penalty = calculator.calculate_total_penalty(test_params)
        breakdown = calculator.get_penalty_breakdown(test_params)
        
        print(f"\nComplexity Weight: {weight}")
        print(f"Toplam Penalty: {total_penalty:.4f}")
        print("Detay:", {k: f"{v:.4f}" for k, v in breakdown.items()})


def test_different_datasets():
    """
    Farklı veri setlerinde optimizasyon testleri.
    """
    print("\nFarklı Veri Setlerinde Test")
    print("=" * 50)
    
    datasets = [
        ("Iris", load_iris()),
        ("Wine", load_wine()),
        ("Breast Cancer", load_breast_cancer())
    ]
    
    bounds = create_sample_bounds()
    
    for name, data in datasets:
        print(f"\n{name} veri seti:")
        
        X = pd.DataFrame(data.data)
        y = pd.Series(data.target)
        
        problem = ImprovedKnnMetaHeuristicProblem(
            attributes=X,
            target=y,
            bounds=bounds,
            complexity_weight=0.1,
            use_cross_validation=False,  # Hızlı test için
            verbose=False
        )
        
        problem.decode_solution = decode_solution_example
        
        # Kısa optimizasyon
        optimizer = PSO.OriginalPSO(epoch=10, pop_size=10)
        best_position, best_fitness = safe_solve(optimizer, problem)
        best_params = decode_solution_example(best_position)
        summary = problem.get_optimization_summary()
        
        print(f"  Veri boyutu: {X.shape}")
        print(f"  Sınıf sayısı: {len(y.unique())}")
        print(f"  En iyi fitness: {best_fitness:.4f}")
        print(f"  En iyi k: {best_params['n_neighbors']}")
        print(f"  Değerlendirme sayısı: {summary['total_evaluations']}")


if __name__ == "__main__":
    # Ana optimizasyon örneği
    run_optimization_example()
    
    # Penalty stratejileri karşılaştırması
    compare_penalty_strategies()
    
    # Farklı veri setlerinde test
    test_different_datasets() 