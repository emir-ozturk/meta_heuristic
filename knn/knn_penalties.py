"""
KNN meta sezgisel optimizasyonu için penalty (ceza) fonksiyonları.
Bu modül, farklı hiperparametreler için ceza fonksiyonlarını içerir.
"""

from typing import Dict, Any
from knn.knn_problem_bounds import problem_bounds

class KnnPenaltyCalculator:
    """
    KNN parametreleri için penalty hesaplama sınıfı.
    Her parametre için ayrı penalty fonksiyonu tanımlanmıştır.
    """

    def __init__(self, complexity_weight: float = 0.1):
        """
        Args:
            complexity_weight: Genel karmaşıklık ağırlığı (0-1 arası)
        """
        self.complexity_weight: float = complexity_weight
        self.max_k: int = 100
        self.max_leaf_size: int = 5
        self.max_p: int = 5

    def calculate_total_penalty(self, design_parameter: Dict[str, Any]) -> float:
        """
        Tüm parametreler için toplam penalty hesaplar.
        
        Args:
            design_parameter: KNN parametreleri dictionary'si
            
        Returns:
            float: Toplam penalty değeri
        """
        total_penalty = 0.0
        
        # K-neighbors penalty (en önemli)
        total_penalty += self._k_neighbors_penalty(design_parameter["n_neighbors"])
        
        # Leaf size penalty (algoritma ball_tree veya kd_tree ise)
        if design_parameter["algorithm"] in ["ball_tree", "kd_tree"]:
            total_penalty += self._leaf_size_penalty(design_parameter["leaf_size"])
        
        # P parameter penalty (Minkowski distance için)
        if design_parameter["metric"] == "minkowski":
            total_penalty += self._p_parameter_penalty(design_parameter["p"])
        
        # Algorithm complexity penalty
        total_penalty += self._algorithm_complexity_penalty(design_parameter["algorithm"])
        
        # Weights penalty
        total_penalty += self._weights_penalty(design_parameter["weights"])
        
        return total_penalty

    def _k_neighbors_penalty(self, k_value: int) -> float:
        """
        K değeri için penalty fonksiyonu.
        Düşük k değerleri daha az penalty alır (overfitting riski).
        Yüksek k değerleri de underfitting riski taşır.
        
        Args:
            k_value: K-neighbors değeri
            
        Returns:
            float: K penalty değeri
        """
        # Normalize k value (0-1 arası)
        normalized_k = k_value / self.max_k
        
        # U-shaped penalty: hem çok düşük hem çok yüksek k değerleri ceza alır
        # Optimal k genellikle sqrt(n_samples) civarındadır
        optimal_ratio = 0.1  # Yaklaşık %10 civarı optimal
        
        # Optimal noktadan uzaklığa göre penalty
        distance_from_optimal = abs(normalized_k - optimal_ratio)
        penalty = distance_from_optimal ** 2  # Quadratic penalty
        
        return penalty * self.complexity_weight * 2.0  # K en önemli parametre

    def _leaf_size_penalty(self, leaf_size: int) -> float:
        """
        Leaf size için penalty fonksiyonu.
        Çok düşük leaf size hesaplama maliyetini artırır.
        
        Args:
            leaf_size: Leaf size değeri
            
        Returns:
            float: Leaf size penalty değeri
        """
        # Normalize leaf size
        normalized_leaf = leaf_size / self.max_leaf_size
        
        # Düşük leaf size değerlerine penalty (hesaplama maliyeti)
        if normalized_leaf < 0.1:  # %10'dan düşükse
            penalty = (0.1 - normalized_leaf) ** 2
        else:
            penalty = 0.0
            
        return penalty * self.complexity_weight * 0.5

    def _p_parameter_penalty(self, p_value: float) -> float:
        """
        Minkowski distance p parametresi için penalty.
        p=1 (Manhattan) ve p=2 (Euclidean) en yaygın kullanılan değerlerdir.
        
        Args:
            p_value: Minkowski distance p parametresi
            
        Returns:
            float: P parameter penalty değeri
        """
        # p=1 veya p=2 için penalty yok
        if p_value in [1.0, 2.0]:
            return 0.0
        
        # Diğer değerler için minimal penalty
        penalty = 0.1
        return penalty * self.complexity_weight * 0.3

    def _algorithm_complexity_penalty(self, algorithm: str) -> float:
        """
        Algoritma karmaşıklığı için penalty.
        
        Args:
            algorithm: KNN algoritması ('auto', 'ball_tree', 'kd_tree', 'brute')
            
        Returns:
            float: Algorithm penalty değeri
        """
        # Algoritma karmaşıklık sıralaması
        complexity_map = {
            'auto': 0.0,        # Otomatik seçim, penalty yok
            'kd_tree': 0.1,     # Düşük boyutlarda etkili
            'ball_tree': 0.2,   # Yüksek boyutlarda etkili
            'brute': 0.5        # Her zaman çalışır ama yavaş
        }
        
        penalty = complexity_map.get(algorithm, 0.0)
        return penalty * self.complexity_weight * 0.4

    def _weights_penalty(self, weights: str) -> float:
        """
        Weights parametresi için penalty.
        
        Args:
            weights: Ağırlık türü ('uniform', 'distance')
            
        Returns:
            float: Weights penalty değeri
        """
        # Distance weights genellikle daha iyi sonuç verir
        if weights == 'uniform':
            penalty = 0.1  # Uniform weights için minimal penalty
        else:  # 'distance'
            penalty = 0.0
            
        return penalty * self.complexity_weight * 0.2

    def get_penalty_breakdown(self, design_parameter: Dict[str, Any]) -> Dict[str, float]:
        """
        Her parametre için penalty değerlerini ayrı ayrı döndürür.
        Debug ve analiz amaçlı kullanılır.
        
        Args:
            design_parameter: KNN parametreleri
            
        Returns:
            Dict[str, float]: Parametre bazında penalty değerleri
        """
        breakdown = {
            'k_neighbors': self._k_neighbors_penalty(design_parameter["n_neighbors"]),
            'algorithm': self._algorithm_complexity_penalty(design_parameter["algorithm"]),
            'weights': self._weights_penalty(design_parameter["weights"])
        }
        
        # Conditional penalties
        if design_parameter["algorithm"] in ["ball_tree", "kd_tree"]:
            breakdown['leaf_size'] = self._leaf_size_penalty(design_parameter["leaf_size"])
        
        if design_parameter["metric"] == "minkowski":
            breakdown['p_parameter'] = self._p_parameter_penalty(design_parameter["p"])
        
        breakdown['total'] = sum(breakdown.values())
        
        return breakdown 