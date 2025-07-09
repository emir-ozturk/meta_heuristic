"""
Mealpy versiyonları arası uyumluluk için yardımcı fonksiyonlar.
"""

def extract_best_solution(result):
    """
    Mealpy solve() sonucundan en iyi çözümü çıkarır.
    Hem eski (tuple) hem yeni (Agent) formatları destekler.
    
    Args:
        result: optimizer.solve() dönüş değeri
        
    Returns:
        tuple: (best_position, best_fitness)
    """
    # Eski format kontrolü (tuple)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    
    # Yeni format kontrolü (Agent object)
    elif hasattr(result, 'solution') and hasattr(result, 'target'):
        return result.solution, result.target.fitness
    
    # Agent object alternatif erişim
    elif hasattr(result, 'position') and hasattr(result, 'fitness'):
        return result.position, result.fitness
    
    # Eğer hiçbiri değilse hata
    else:
        raise ValueError(f"Desteklenmeyen Mealpy sonuç formatı: {type(result)}")


def safe_solve(optimizer, problem):
    """
    Optimizer.solve() fonksiyonunu güvenli şekilde çalıştırır.
    
    Args:
        optimizer: Mealpy optimizer instance
        problem: Problem instance
        
    Returns:
        tuple: (best_position, best_fitness)
    """
    try:
        result = optimizer.solve(problem)
        return extract_best_solution(result)
    except Exception as e:
        print(f"Optimizasyon hatası: {e}")
        raise 