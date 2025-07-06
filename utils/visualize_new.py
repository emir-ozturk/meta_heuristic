import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mealpy.optimizer import Optimizer
import seaborn as sns
from typing import Optional

def plot_fitness(model: Optimizer, title: str = "Fitness Evolution", save_path: Optional[str] = None) -> None:
    """
    Meta-sezgisel algoritmaların performansını görselleştirmek için geliştirilmiş fonksiyon.
    """
    
    print(f"Best accuracy: {model.g_best.target.fitness}")
    print(f"Best parameters: \n{model.problem.decode_solution(model.g_best.solution)}")

    try:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Fitness evolution
        plt.subplot(2, 2, 1)
        plt.plot(model.history.list_global_best_fit, 'b-', linewidth=2, label='Global Best')
        plt.plot(model.history.list_current_best_fit, 'r--', alpha=0.7, label='Current Best')
        plt.xlabel("Epoch")
        plt.ylabel("Fitness Score")
        plt.title("Fitness Evolution Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Convergence analysis
        plt.subplot(2, 2, 2)
        improvements = np.diff(model.history.list_global_best_fit)
        plt.plot(improvements, 'g-', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Epoch")
        plt.ylabel("Fitness Improvement")
        plt.title("Convergence Rate")
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Performance statistics
        plt.subplot(2, 2, 3)
        final_fitness = model.history.list_global_best_fit[-1]
        convergence_epoch = find_convergence_epoch(model.history.list_global_best_fit)
        
        stats_text = f"""
Final Fitness: {final_fitness:.4f}
Convergence Epoch: {convergence_epoch}
Total Epochs: {len(model.history.list_global_best_fit)}
Best Improvement: {max(improvements) if len(improvements) > 0 else 0:.6f}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        plt.title("Performance Summary")
        
        # Subplot 4: Distribution of fitness values
        plt.subplot(2, 2, 4)
        plt.hist(model.history.list_global_best_fit, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(final_fitness, color='red', linestyle='--', linewidth=2, label=f'Final: {final_fitness:.4f}')
        plt.xlabel("Fitness Value")
        plt.ylabel("Frequency")
        plt.title("Fitness Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except Exception as e:
        raise Exception(f"Visualization error: {e}")

def plot_pareto_analysis(problem, title: str = "Pareto Front Analysis", save_path: Optional[str] = None) -> None:
    """
    Multi-objective optimization için Pareto front analizi.
    """
    
    if not hasattr(problem, 'get_pareto_front_analysis'):
        print("Problem does not support Pareto analysis")
        return
    
    df = problem.get_pareto_front_analysis()
    
    if df.empty:
        print("No evaluation history available for Pareto analysis")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Accuracy vs K neighbors
    plt.subplot(2, 3, 1)
    plt.scatter(df['k_neighbors'], df['raw_accuracy'], c=df['final_score'], 
                cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='Final Score')
    plt.xlabel("K Neighbors")
    plt.ylabel("Raw Accuracy")
    plt.title("Accuracy vs K Neighbors")
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Complexity penalty vs K neighbors
    plt.subplot(2, 3, 2)
    plt.scatter(df['k_neighbors'], df['complexity_penalty'], c='orange', s=100, alpha=0.7)
    plt.xlabel("K Neighbors")
    plt.ylabel("Complexity Penalty")
    plt.title("Complexity Penalty vs K Neighbors")
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Pareto front
    plt.subplot(2, 3, 3)
    plt.scatter(df['complexity_penalty'], df['raw_accuracy'], c=df['final_score'], 
                cmap='RdYlBu', s=100, alpha=0.7)
    plt.colorbar(label='Final Score')
    plt.xlabel("Complexity Penalty")
    plt.ylabel("Raw Accuracy")
    plt.title("Pareto Front: Accuracy vs Complexity")
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Top K values
    plt.subplot(2, 3, 4)
    top_10 = df.head(10)
    plt.bar(range(len(top_10)), top_10['final_score'], color='lightgreen', alpha=0.7)
    plt.xticks(range(len(top_10)), [f"K={k}" for k in top_10['k_neighbors']], rotation=45)
    plt.ylabel("Final Score")
    plt.title("Top 10 K Values by Final Score")
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: K distribution
    plt.subplot(2, 3, 5)
    plt.hist(df['k_neighbors'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel("K Neighbors")
    plt.ylabel("Frequency")
    plt.title("K Neighbors Distribution")
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Statistics
    plt.subplot(2, 3, 6)
    best_k = df.iloc[0]['k_neighbors']
    best_accuracy = df.iloc[0]['raw_accuracy']
    best_score = df.iloc[0]['final_score']
    
    stats_text = f"""
Best K: {best_k}
Best Accuracy: {best_accuracy:.4f}
Best Final Score: {best_score:.4f}
Total K values tested: {len(df)}
K range: {df['k_neighbors'].min()}-{df['k_neighbors'].max()}
Accuracy range: {df['raw_accuracy'].min():.4f}-{df['raw_accuracy'].max():.4f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    plt.axis('off')
    plt.title("Analysis Summary")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_optimization_comparison(results: dict, title: str = "Algorithm Comparison", save_path: Optional[str] = None) -> None:
    """
    Farklı meta-sezgisel algoritmaların karşılaştırılması.
    
    Args:
        results: {'algorithm_name': model} şeklinde dictionary
    """
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Fitness evolution comparison
    plt.subplot(2, 2, 1)
    for name, model in results.items():
        plt.plot(model.history.list_global_best_fit, label=name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Fitness Score")
    plt.title("Fitness Evolution Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Final performance comparison
    plt.subplot(2, 2, 2)
    names = list(results.keys())
    final_scores = [model.g_best.target.fitness for model in results.values()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    
    bars = plt.bar(names, final_scores, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel("Final Fitness Score")
    plt.title("Final Performance Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, final_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Convergence comparison
    plt.subplot(2, 2, 3)
    convergence_epochs = []
    for name, model in results.items():
        conv_epoch = find_convergence_epoch(model.history.list_global_best_fit)
        convergence_epochs.append(conv_epoch)
    
    plt.bar(names, convergence_epochs, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel("Convergence Epoch")
    plt.title("Convergence Speed Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Best parameters comparison
    plt.subplot(2, 2, 4)
    param_text = ""
    for name, model in results.items():
        params = model.problem.decode_solution(model.g_best.solution)
        k_value = params.get('n_neighbors', 'N/A')
        param_text += f"{name}: K={k_value}\n"
    
    plt.text(0.1, 0.5, param_text, fontsize=10, verticalalignment='center')
    plt.axis('off')
    plt.title("Best K Values")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def find_convergence_epoch(fitness_history: list, tolerance: float = 1e-6) -> int:
    """
    Algoritmanın convergence epoch'unu bulur.
    """
    if len(fitness_history) < 2:
        return 0
    
    for i in range(1, len(fitness_history)):
        if abs(fitness_history[i] - fitness_history[i-1]) < tolerance:
            # Son 5 epoch'ta değişim olmadığını kontrol et
            stable_count = 0
            for j in range(max(0, i-5), i):
                if abs(fitness_history[j] - fitness_history[i]) < tolerance:
                    stable_count += 1
            
            if stable_count >= 3:  # En az 3 epoch stable
                return i
    
    return len(fitness_history) - 1

def plot_k_efficiency_analysis(problem, title: str = "K Efficiency Analysis", save_path: Optional[str] = None) -> None:
    """
    K değerlerinin efficiency analizi.
    """
    
    if not hasattr(problem, 'recommend_optimal_k'):
        print("Problem does not support K efficiency analysis")
        return
    
    recommendation = problem.recommend_optimal_k()
    
    if 'error' in recommendation:
        print(recommendation['error'])
        return
    
    df = problem.get_pareto_front_analysis()
    
    plt.figure(figsize=(12, 8))
    
    # K değerleri vs accuracy
    plt.subplot(2, 2, 1)
    plt.scatter(df['k_neighbors'], df['raw_accuracy'], alpha=0.6, s=50)
    plt.axvline(recommendation['recommended_k'], color='red', linestyle='--', 
                label=f"Recommended K={recommendation['recommended_k']}")
    plt.xlabel("K Neighbors")
    plt.ylabel("Raw Accuracy")
    plt.title("K Neighbors vs Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Efficiency ratio
    plt.subplot(2, 2, 2)
    efficiency_ratio = df['raw_accuracy'] / df['k_neighbors']
    plt.scatter(df['k_neighbors'], efficiency_ratio, alpha=0.6, s=50)
    plt.axvline(recommendation['recommended_k'], color='red', linestyle='--')
    plt.xlabel("K Neighbors")
    plt.ylabel("Accuracy/K Ratio")
    plt.title("Efficiency Ratio (Accuracy/K)")
    plt.grid(True, alpha=0.3)
    
    # Alternative K values
    plt.subplot(2, 2, 3)
    alt_ks = recommendation.get('alternative_ks', [])
    if alt_ks:
        alt_data = df[df['k_neighbors'].isin(alt_ks)]
        plt.bar(range(len(alt_data)), alt_data['raw_accuracy'], 
                color='lightgreen', alpha=0.7)
        plt.xticks(range(len(alt_data)), [f"K={k}" for k in alt_data['k_neighbors']])
        plt.ylabel("Accuracy")
        plt.title("Alternative K Values Performance")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Recommendation summary
    plt.subplot(2, 2, 4)
    summary_text = f"""
Recommended K: {recommendation['recommended_k']}
Accuracy: {recommendation['accuracy']:.4f}

Reasoning:
{recommendation['reasoning']}

Alternative K values:
{', '.join(map(str, alt_ks[:5]))}
    """
    plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    plt.axis('off')
    plt.title("Recommendation Summary")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 