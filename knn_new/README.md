# Geliştirilmiş KNN Meta-sezgisel Hiperparametre Optimizasyonu

Bu klasör, KNN hiperparametre optimizasyonu için geliştirilmiş meta-sezgisel algoritma implementasyonunu içerir.

## 🚨 Ana Problem ve Çözümü

### Problem:
Mevcut implementasyonda **"K=15 ve K=67 aynı accuracy veriyorsa neden K=67 öneriliyor?"** sorunu vardı. Bu problem şu nedenlerden kaynaklanıyordu:
- Sadece accuracy maksimize ediliyordu
- Computational cost göz önünde bulundurulmuyordu
- Cross-validation kullanılmıyordu (overfitting riski)
- Model simplicity optimize edilmiyordu

### Çözüm:
✅ **Multi-objective optimization**: Accuracy + Model Simplicity  
✅ **Cross-validation**: Daha güvenilir performans değerlendirmesi  
✅ **Complexity penalty**: Düşük K değerleri tercih ediliyor  
✅ **Smart K recommendation**: Aynı accuracy'deki en düşük K öneriliyor  

## 🎯 Yeni Özellikler

### 1. Cross-Validation Desteği
```python
problem = KnnMetaHeuristicProblem(
    attributes, target,
    use_cv=True,           # Cross-validation kullan
    cv_folds=5,           # 5-fold CV
    complexity_weight=0.1  # Complexity penalty ağırlığı
)
```

### 2. Multi-objective Optimization
- **Accuracy**: Ana performans metriği
- **Complexity Penalty**: K değeri büyüdükçe penalty artar
- **Final Score** = Accuracy - (complexity_weight × complexity_penalty)

### 3. Akıllı K Önerisi
```python
recommendation = problem.recommend_optimal_k()
print(f"Önerilen K: {recommendation['recommended_k']}")
print(f"Accuracy: {recommendation['accuracy']}")
print(f"Açıklama: {recommendation['reasoning']}")
```

### 4. Pareto Front Analizi
Accuracy vs Complexity trade-off'unu görselleştirin:
```python
from utils.visualize_new import plot_pareto_analysis
plot_pareto_analysis(problem)
```

## 📁 Dosya Yapısı

```
knn_new/
├── knn_problem.py           # Geliştirilmiş KNN problem sınıfı
├── knn_problem_bounds.py    # Esnek bound konfigürasyonları
└── README.md               # Bu dosya

utils/
└── visualize_new.py        # Gelişmiş görselleştirme araçları

knn_new_example.ipynb       # Detaylı örnek notebook
```

## 🚀 Hızlı Başlangıç

### Basit Kullanım
```python
from knn_new.knn_problem import KnnMetaHeuristicProblem
from knn_new.knn_problem_bounds import get_recommended_config
from mealpy import PSO

# Dataset yükle
df = pd.read_csv("datasets/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Otomatik konfigürasyon
config = get_recommended_config(len(df))

# Problem oluştur
problem = KnnMetaHeuristicProblem(
    X, y, 
    bounds=config['bounds'],
    complexity_weight=config['complexity_weight']
)

# Optimize et
model = PSO.OriginalPSO(epoch=50)
model.solve(problem)

# Optimal K önerisi al
recommendation = problem.recommend_optimal_k()
print(f"Önerilen K: {recommendation['recommended_k']}")
```

### Gelişmiş Kullanım
```python
# Farklı optimizasyon stratejileri
strategies = ['accuracy_focused', 'balanced', 'efficiency_focused']

for strategy in strategies:
    problem = KnnMetaHeuristicProblem(
        X, y,
        bounds=get_bounds_for_strategy('cv'),
        complexity_weight=COMPLEXITY_CONFIGS[strategy]['complexity_weight']
    )
    # ... optimize et
```

## 📊 Konfigürasyon Seçenekleri

### 1. Dataset Boyutuna Göre Otomatik Ayar
```python
config = get_recommended_config(dataset_size)
# < 1000 samples: Quick mode
# 1000-10000: Balanced mode  
# > 10000: Efficiency mode
```

### 2. Complexity Weight Stratejileri
- **accuracy_focused** (0.01): Sadece accuracy odaklı
- **balanced** (0.1): Dengeli optimizasyon
- **efficiency_focused** (0.2): Düşük K değerleri tercih

### 3. Bounds Stratejileri
- **cv**: Cross-validation için (test_size gereksiz)
- **split**: Train/test split için
- **quick**: Hızlı test için dar aralık

## 🔬 CV'ye Katkıları

### Teknik Yeterlikler
1. **Multi-objective Optimization**: Birden fazla hedefi aynı anda optimize etme
2. **Cross-validation Expertise**: Overfitting önleme ve güvenilir performans
3. **Meta-heuristic Algorithms**: PSO, GA, ABC, SOS algoritmalarında uzmanlik
4. **Hyperparameter Tuning**: Production-ready hiperparametre optimizasyonu

### Pratik Faydalar
1. **Daha az computational cost**: Düşük K değerleri tercih
2. **Daha güvenilir sonuçlar**: CV ile overfitting azalması
3. **Scalable solution**: Dataset boyutuna göre adaptive konfigürasyon
4. **Interpretable results**: Pareto analizi ile trade-off görünürlüğü

## 🆚 Eski vs Yeni Karşılaştırma

| Özellik | Eski Implementation | Yeni Implementation |
|---------|-------------------|-------------------|
| Evaluation | Single train/test split | Cross-validation |
| Objective | Sadece accuracy | Multi-objective (accuracy + simplicity) |
| K Selection | Rastgele yüksek K | Akıllı düşük K önerisi |
| Overfitting | Risk var | CV ile azaltılmış |
| Computational Cost | Göz ardı ediliyor | Optimize ediliyor |
| Analysis | Basit | Pareto front + efficiency analysis |

## 📈 Örnek Sonuçlar

**Eski Sistem:**
- K=67, Accuracy=0.804
- Neden yüksek K? → Belirsiz

**Yeni Sistem:**
- K=15, Accuracy=0.803, Final Score=0.791
- Reasoning: "K=15 gives 0.803 accuracy with lower computational cost"
- Alternative K values: [15, 17, 19, 23]

## 🔧 Troubleshooting

### Yaygın Problemler
1. **Yavaş çalışma**: `strategy='quick'` kullanın
2. **Memory issues**: `cv_folds` azaltın
3. **Poor convergence**: `complexity_weight` azaltın

### Performance Tuning
```python
# Hızlı test için
bounds = get_bounds_for_strategy('quick')
termination = get_termination_for_strategy('quick')

# Büyük dataset için
config = get_recommended_config(dataset_size)
# Otomatik olarak efficiency_focused olacak
```

## 📚 Daha Fazla Bilgi

- `knn_new_example.ipynb`: Detaylı örnek ve karşılaştırmalar
- `utils/visualize_new.py`: Görselleştirme fonksiyonları
- Mealpy Documentation: [https://mealpy.readthedocs.io/](https://mealpy.readthedocs.io/)

---

**🎯 Sonuç**: Bu geliştirilmiş implementasyon ile artık K=67 yerine K=15 gibi daha efficiency odaklı ve mantıklı öneriler alacaksınız! 