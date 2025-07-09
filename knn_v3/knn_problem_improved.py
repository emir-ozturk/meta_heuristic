"""
KNN modeli için meta sezgisel algoritmalar için geliştirilmiş amaç fonksiyonu.
Bu versiyon daha modüler ve penalty fonksiyonları ayrı bir modülde tanımlanmıştır.
"""

from mealpy import Problem
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Optional
import logging

from .knn_penalties import KnnPenaltyCalculator


class ImprovedKnnMetaHeuristicProblem(Problem):
    """
    Geliştirilmiş KNN meta sezgisel optimizasyon problemi.
    
    Özellikler:
    - Modüler penalty sistemi
    - Detaylı logging
    - Cross-validation seçeneği
    - Daha iyi hata yönetimi
    """

    def __init__(
        self, 
        attributes: pd.DataFrame, 
        target: pd.Series, 
        bounds: list[dict] = None,
        complexity_weight: float = 0.1,
        use_cross_validation: bool = False,
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = False,
        **kwargs
    ):
        """
        Args:
            attributes: Öznitelik matrisi
            target: Hedef değişken
            bounds: Parametre sınırları
            complexity_weight: Karmaşıklık ağırlığı (0-1 arası)
            use_cross_validation: Cross-validation kullanılıp kullanılmayacağı
            cv_folds: Cross-validation fold sayısı
            random_state: Rastgelelik tohumu
            verbose: Detaylı çıktı kontrolü
        """
        super().__init__(bounds, minmax="max", **kwargs)
        
        # Temel veriler
        self.attributes = attributes
        self.target = target
        self.random_state = random_state
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.verbose = verbose
        
        # Penalty hesaplayıcı
        self.penalty_calculator = KnnPenaltyCalculator(complexity_weight)
        
        # Logging setup
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        
        # Performans istatistikleri
        self.evaluation_count = 0
        self.best_fitness = -np.inf
        self.best_parameters = None

    def obj_func(self, design_parameter: np.ndarray) -> float:
        """
        Geliştirilmiş amaç fonksiyonu.
        
        Args:
            design_parameter: Optimizasyon parametreleri
            
        Returns:
            float: Fitness değeri (accuracy - penalty)
        """
        self.evaluation_count += 1
        
        try:
            # Parametreleri decode et
            decoded_params = self.decode_solution(design_parameter)
            
            # Model performansını değerlendir
            accuracy = self._evaluate_model_performance(decoded_params)
            
            # Penalty hesapla
            total_penalty = self.penalty_calculator.calculate_total_penalty(decoded_params)
            
            # Final fitness
            fitness = accuracy - total_penalty
            
            # En iyi sonucu güncelle
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_parameters = decoded_params.copy()
                
                if self.verbose:
                    self.logger.info(f"New best fitness: {fitness:.4f} (accuracy: {accuracy:.4f}, penalty: {total_penalty:.4f})")
                    self.logger.info(f"Best parameters: {decoded_params}")
            
            return fitness
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error in evaluation {self.evaluation_count}: {e}")
            # Hata durumunda düşük fitness döndür
            return -1.0

    def _evaluate_model_performance(self, params: Dict[str, Any]) -> float:
        """
        Model performansını değerlendirir.
        
        Args:
            params: KNN parametreleri
            
        Returns:
            float: Accuracy değeri
        """
        if self.use_cross_validation:
            return self._cross_validation_accuracy(params)
        else:
            return self._single_split_accuracy(params)

    def _single_split_accuracy(self, params: Dict[str, Any]) -> float:
        """
        Tek split ile accuracy hesaplar.
        
        Args:
            params: KNN parametreleri
            
        Returns:
            float: Accuracy değeri
        """
        # Veri bölme
        x_train, x_test, y_train, y_test = train_test_split(
            self.attributes, 
            self.target, 
            test_size=params["test_size"], 
            random_state=self.random_state,
            stratify=self.target  # Sınıf dengesini koru
        )
        
        # Model eğitimi ve test
        accuracy = self._train_and_evaluate(x_train, x_test, y_train, y_test, params)
        
        return accuracy

    def _cross_validation_accuracy(self, params: Dict[str, Any]) -> float:
        """
        Cross-validation ile accuracy hesaplar.
        
        Args:
            params: KNN parametreleri
            
        Returns:
            float: Ortalama accuracy değeri
        """
        from sklearn.model_selection import cross_val_score
        
        # Model oluştur
        model = self._create_knn_model(params)
        
        # Veriyi normalize et
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(self.attributes)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, 
            x_scaled, 
            self.target, 
            cv=self.cv_folds, 
            scoring='accuracy',
            n_jobs=-1  # Paralel işlem
        )
        
        return cv_scores.mean()

    def _train_and_evaluate(
        self, 
        x_train: np.ndarray, 
        x_test: np.ndarray, 
        y_train: pd.Series, 
        y_test: pd.Series, 
        params: Dict[str, Any]
    ) -> float:
        """
        Model eğitimi ve değerlendirmesi.
        
        Args:
            x_train, x_test: Eğitim ve test öznitelikleri
            y_train, y_test: Eğitim ve test hedef değişkenleri
            params: KNN parametreleri
            
        Returns:
            float: Test accuracy'si
        """
        # Normalizasyon
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        # Model oluştur ve eğit
        model = self._create_knn_model(params)
        model.fit(x_train_scaled, y_train)
        
        # Tahmin ve değerlendirme
        y_pred = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy

    def _create_knn_model(self, params: Dict[str, Any]) -> KNeighborsClassifier:
        """
        KNN modeli oluşturur.
        
        Args:
            params: Model parametreleri
            
        Returns:
            KNeighborsClassifier: Yapılandırılmış KNN modeli
        """
        # Temel parametreler
        model_params = {
            'n_neighbors': params["n_neighbors"],
            'weights': params["weights"],
            'algorithm': params["algorithm"]
        }
        
        # Koşullu parametreler
        if params["algorithm"] in ["ball_tree", "kd_tree"]:
            model_params['leaf_size'] = params["leaf_size"]
        
        if params["metric"] == "minkowski":
            model_params['p'] = params["p"]
            
        model_params['metric'] = params["metric"]
        
        return KNeighborsClassifier(**model_params)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Optimizasyon sürecinin özetini döndürür.
        
        Returns:
            Dict: Optimizasyon özet bilgileri
        """
        summary = {
            'total_evaluations': self.evaluation_count,
            'best_fitness': self.best_fitness,
            'best_parameters': self.best_parameters,
            'dataset_shape': self.attributes.shape,
            'target_classes': len(self.target.unique()),
            'cross_validation_used': self.use_cross_validation
        }
        
        if self.best_parameters:
            # En iyi parametreler için penalty breakdown
            penalty_breakdown = self.penalty_calculator.get_penalty_breakdown(
                self.best_parameters
            )
            summary['penalty_breakdown'] = penalty_breakdown
        
        return summary

    def reset_statistics(self):
        """Optimizasyon istatistiklerini sıfırlar."""
        self.evaluation_count = 0
        self.best_fitness = -np.inf
        self.best_parameters = None 