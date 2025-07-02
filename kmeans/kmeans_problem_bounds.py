"""
Metaheuristik algoritmalarda kullanılan amaç fonksiyonu için sınır değerleri.
"""

from mealpy import StringVar, IntegerVar, FloatVar, BoolVar

problem_bounds: list[dict] = [
    # Test verisi boyutu için sınır değerleri
    FloatVar(lb=0.1, ub=0.3, name="test_size"),

    # KMeans Hiperparametreleri
    IntegerVar(lb=2, ub=20, name="n_clusters"),  # Küme sayısı
    StringVar(valid_sets=('k-means++', 'random'), name="init"),  # Başlangıç merkez seçim metodu
    IntegerVar(lb=5, ub=20, name="n_init"),  # Farklı başlangıç noktalarıyla deneme sayısı
    IntegerVar(lb=100, ub=500, name="max_iter"),  # Maksimum iterasyon sayısı
    FloatVar(lb=1e-5, ub=1e-2, name="tol"),  # Yakınsama toleransı
    StringVar(valid_sets=('lloyd', 'elkan'), name="algorithm"),  # Kullanılacak algoritma
    BoolVar(name="copy_x"),  # Veri kopyalama
]

termination: dict[str, int | float] = {
    "max_epoch": 100,        # Maksimum iterasyon sayısı
    "max_fe": 20000,         # Maksimum amaç fonksiyonu değerlendirmesi
    "max_early_stop": 20,    # İyileşme olmadan durma eşiği (epoch)
    "epsilon": 1e-8,         # Erken durdurmada minimum iyileşme eşiği
}
