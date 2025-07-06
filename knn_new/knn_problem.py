"""
KNN modeli için meta sezgisel algoritmalar için geliştirilmiş amaç fonksiyonu.
Cross-validation ve multi-objective optimization içerir.
"""

from mealpy import Problem
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class KnnMetaHeuristicProblem(Problem):
    """
    KNN modeli için meta sezgisel algoritmalar için geliştirilmiş amaç fonksiyonu.
    
    Yeni özellikler:
    - Cross-validation ile daha güvenilir performans değerlendirmesi
    - Multi-objective optimization (accuracy + model simplicity)
    - K değeri optimizasyonu (computational efficiency)
    - Stratified sampling ile class balance korunması
    """

    def __init__(self, attributes: pd.DataFrame, target: pd.Series, bounds: list[dict] = None, 
                 use_cv: bool = True, cv_folds: int = 5, complexity_weight: float = 0.1, **kwargs):
        super().__init__(bounds, minmax="max", **kwargs)
        self.attributes = attributes
        self.target = target
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.complexity_weight = complexity_weight  # Model karmaşıklığı ağırlığı
        
        # Veri setini önceden normalize et (CV için)
        self.scaler = StandardScaler()
        self.attributes_scaled = self.scaler.fit_transform(self.attributes)
        
        # Performans tracking için
        self.evaluation_history = []

    def obj_func(self, design_parameter: np.ndarray) -> float:
        """
        KNN modeli eğitilir ve cross-validation ile değerlendirilir.
        Multi-objective: accuracy + model simplicity (k değerinin düşük olması)
        """

        try:
            design_parameter = self.decode_solution(design_parameter)

            if self.use_cv:
                # Cross-validation ile değerlendirme
                accuracy = self._evaluate_with_cv(design_parameter)
            else:
                # Geleneksel train/test split ile değerlendirme
                accuracy = self._evaluate_with_split(design_parameter)
            
            # Multi-objective: Accuracy + Model Simplicity
            # k değeri küçükse bonus puan ver (computational efficiency için)
            k_penalty = self._calculate_complexity_penalty(design_parameter["n_neighbors"])
            
            # Final score = accuracy - complexity_penalty
            final_score = accuracy - (self.complexity_weight * k_penalty)
            
            # Performans geçmişini kaydet
            self.evaluation_history.append({
                'k_neighbors': design_parameter["n_neighbors"],
                'raw_accuracy': accuracy,
                'complexity_penalty': k_penalty,
                'final_score': final_score,
                'parameters': design_parameter.copy()
            })
            
            return final_score
            
        except Exception as e:
            # Hata durumunda çok düşük değer döndür
            print(f"Evaluation error: {e}")
            return 0.0

    def _evaluate_with_cv(self, design_parameter: dict) -> float:
        """Cross-validation ile model performansını değerlendir"""
        
        # KNN modeli oluştur
        model = KNeighborsClassifier(
            n_neighbors=design_parameter["n_neighbors"], 
            weights=design_parameter["weights"], 
            algorithm=design_parameter["algorithm"], 
            leaf_size=design_parameter["leaf_size"], 
            p=design_parameter["p"], 
            metric=design_parameter["metric"]
        )
        
        # Stratified K-Fold CV kullan (class distribution'ı koru)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation skorları hesapla
        cv_scores = cross_val_score(
            model, self.attributes_scaled, self.target, 
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        # Ortalama accuracy döndür
        return cv_scores.mean()

    def _evaluate_with_split(self, design_parameter: dict) -> float:
        """Geleneksel train/test split ile değerlendirme"""
        
        # Veri seti eğitim ve test olarak ayrılır
        x_train, x_test, y_train, y_test = train_test_split(
            self.attributes, self.target, 
            test_size=design_parameter["test_size"], 
            random_state=42,
            stratify=self.target  # Class balance'ı koru
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

    def _calculate_complexity_penalty(self, k_value: int) -> float:
        """
        K değeri için karmaşıklık penaltısı hesapla.
        Daha düşük k değerleri daha az penaltı alır.
        """
        # Normalize edilmiş penalty (0-1 arası)
        max_k = 100  # bounds'da tanımlanan maksimum k değeri
        normalized_k = k_value / max_k
        
        # Logaritmik penalty (küçük k değerlerine daha az ceza)
        penalty = np.log(1 + normalized_k)
        
        return penalty

    def get_model_info(self, design_parameter: np.ndarray) -> dict:
        """Model bilgilerini döndür (debugging için)"""
        params = self.decode_solution(design_parameter)
        accuracy = self.obj_func(design_parameter)
        complexity_penalty = self._calculate_complexity_penalty(params["n_neighbors"])
        
        return {
            'parameters': params,
            'raw_accuracy': accuracy + (self.complexity_weight * complexity_penalty),
            'complexity_penalty': complexity_penalty,
            'final_score': accuracy,
            'k_value': params["n_neighbors"]
        }

    def get_pareto_front_analysis(self) -> pd.DataFrame:
        """
        Pareto front analizi için evaluation history'yi döndür.
        Bu sayede accuracy vs complexity trade-off'unu görebiliriz.
        """
        if not self.evaluation_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.evaluation_history)
        
        # En iyi k değerlerini analiz et
        df_grouped = df.groupby('k_neighbors').agg({
            'raw_accuracy': 'max',
            'complexity_penalty': 'mean',
            'final_score': 'max'
        }).reset_index()
        
        return df_grouped.sort_values('final_score', ascending=False)

    def recommend_optimal_k(self) -> dict:
        """
        Optimal k değeri önerisi.
        Accuracy ve complexity trade-off'unu göz önünde bulundurur.
        """
        if not self.evaluation_history:
            return {"error": "No evaluation history available"}
        
        df = pd.DataFrame(self.evaluation_history)
        
        # En yüksek accuracy'yi bulan k değerleri
        max_accuracy = df['raw_accuracy'].max()
        high_accuracy_models = df[df['raw_accuracy'] >= max_accuracy - 0.01]  # %1 tolerance
        
        # Bunlar arasından en düşük k değerini seç
        optimal_k = high_accuracy_models.loc[high_accuracy_models['k_neighbors'].idxmin()]
        
        return {
            'recommended_k': int(optimal_k['k_neighbors']),
            'accuracy': optimal_k['raw_accuracy'],
            'reasoning': f"K={int(optimal_k['k_neighbors'])} gives {optimal_k['raw_accuracy']:.4f} accuracy with lower computational cost",
            'alternative_ks': high_accuracy_models['k_neighbors'].tolist()
        } 