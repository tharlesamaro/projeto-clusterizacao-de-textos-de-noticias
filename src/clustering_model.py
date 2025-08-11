"""
Módulo para clusterização de embeddings de texto.

Este módulo implementa diferentes algoritmos de clusterização e métodos
para avaliar a qualidade dos clusters formados.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class ClusteringModel:
    """
    Classe para aplicar diferentes algoritmos de clusterização em embeddings.
    
    Algoritmos implementados:
    1. K-Means: Particiona dados em K clusters baseado em centróides
    2. DBSCAN: Agrupa pontos densos e identifica outliers
    3. Hierarchical: Cria hierarquia de clusters
    """
    
    def __init__(self, algorithm: str = 'kmeans'):
        """
        Inicializa o modelo de clusterização.
        
        Args:
            algorithm: 'kmeans', 'dbscan', ou 'hierarchical'
        """
        self.algorithm = algorithm
        self.model = None
        self.labels_ = None
        self.n_clusters_ = None
        self.scaler = StandardScaler()
        
        print(f"Modelo de clusterização inicializado: {algorithm.upper()}")
    
    def find_optimal_k(self, embeddings: np.ndarray, k_range: range = range(2, 11)) -> Dict:
        """
        Encontra o número ótimo de clusters usando método do cotovelo e silhouette.
        
        O método do cotovelo mostra onde a melhoria na qualidade do cluster
        diminui drasticamente (formando um "cotovelo" no gráfico).
        
        Args:
            embeddings: Array com embeddings
            k_range: Range de valores K para testar
            
        Returns:
            Dicionário com métricas para cada K
        """
        print("Buscando número ótimo de clusters...")
        
        # Normaliza os embeddings (importante para K-means)
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        metrics = {
            'k_values': [],
            'inertia': [],           # Soma das distâncias ao centróide (menor é melhor)
            'silhouette': [],        # Qualidade dos clusters (-1 a 1, maior é melhor)
            'calinski_harabasz': [], # Relação variância entre/dentro clusters (maior é melhor)
            'davies_bouldin': []     # Separação dos clusters (menor é melhor)
        }
        
        for k in k_range:
            # Treina K-means para este K
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_scaled)
            
            # Calcula métricas
            metrics['k_values'].append(k)
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(embeddings_scaled, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(embeddings_scaled, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(embeddings_scaled, labels))
        
        # Encontra K ótimo baseado em silhouette score
        best_k_idx = np.argmax(metrics['silhouette'])
        best_k = metrics['k_values'][best_k_idx]
        
        print(f"Análise concluída! K ótimo sugerido: {best_k} (silhouette: {metrics['silhouette'][best_k_idx]:.3f})")
        
        return metrics
    
    def plot_elbow_analysis(self, metrics: Dict, save_path: Optional[str] = None):
        """
        Plota gráficos para análise do cotovelo e outras métricas.
        
        Args:
            metrics: Métricas retornadas por find_optimal_k
            save_path: Caminho para salvar o gráfico
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        k_values = metrics['k_values']
        
        # 1. Método do cotovelo (Inertia)
        ax1.plot(k_values, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Clusters (K)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Método do Cotovelo')
        ax1.grid(True, alpha=0.3)
        
        # 2. Silhouette Score
        ax2.plot(k_values, metrics['silhouette'], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Número de Clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score por K')
        ax2.grid(True, alpha=0.3)
        
        # Marca o melhor K
        best_k_idx = np.argmax(metrics['silhouette'])
        ax2.axvline(x=k_values[best_k_idx], color='red', linestyle='--', alpha=0.7)
        
        # 3. Calinski-Harabasz Score
        ax3.plot(k_values, metrics['calinski_harabasz'], 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Número de Clusters (K)')
        ax3.set_ylabel('Calinski-Harabasz Score')
        ax3.set_title('Calinski-Harabasz Score por K')
        ax3.grid(True, alpha=0.3)
        
        # 4. Davies-Bouldin Score
        ax4.plot(k_values, metrics['davies_bouldin'], 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Número de Clusters (K)')
        ax4.set_ylabel('Davies-Bouldin Score')
        ax4.set_title('Davies-Bouldin Score por K')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em {save_path}")
        
        plt.show()
    
    def fit_kmeans(self, embeddings: np.ndarray, n_clusters: int = None, **kwargs) -> np.ndarray:
        """
        Aplica algoritmo K-Means.
        
        K-Means particiona os dados em K clusters, onde cada ponto
        pertence ao cluster cujo centróide está mais próximo.
        
        Args:
            embeddings: Array com embeddings
            n_clusters: Número de clusters (se None, tenta encontrar automaticamente)
            **kwargs: Parâmetros adicionais para KMeans
            
        Returns:
            Array com labels dos clusters
        """
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        if n_clusters is None:
            # Busca K ótimo automaticamente
            metrics = self.find_optimal_k(embeddings)
            n_clusters = metrics['k_values'][np.argmax(metrics['silhouette'])]
        
        print(f"Aplicando K-Means com {n_clusters} clusters...")
        
        # Parâmetros padrão otimizados
        default_params = {
            'n_clusters': n_clusters,
            'random_state': 42,
            'n_init': 10,
            'max_iter': 300
        }
        default_params.update(kwargs)
        
        self.model = KMeans(**default_params)
        self.labels_ = self.model.fit_predict(embeddings_scaled)
        self.n_clusters_ = n_clusters
        
        print(f"K-Means concluído! Clusters formados: {len(np.unique(self.labels_))}")
        return self.labels_
    
    def fit_dbscan(self, embeddings: np.ndarray, eps: float = None, min_samples: int = None, **kwargs) -> np.ndarray:
        """
        Aplica algoritmo DBSCAN.
        
        DBSCAN agrupa pontos que estão densamente agrupados e marca
        como outliers pontos que estão em regiões de baixa densidade.
        
        Args:
            embeddings: Array com embeddings
            eps: Distância máxima entre pontos do mesmo cluster
            min_samples: Número mínimo de pontos para formar um cluster
            **kwargs: Parâmetros adicionais para DBSCAN
            
        Returns:
            Array com labels dos clusters (-1 indica outlier)
        """
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Estima parâmetros se não fornecidos
        if eps is None:
            # Usa regra empírica baseada no dataset
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=4)
            neighbors_fit = neighbors.fit(embeddings_scaled)
            distances, indices = neighbors_fit.kneighbors(embeddings_scaled)
            distances = np.sort(distances[:, 3], axis=0)
            eps = np.percentile(distances, 90)  # Usa 90º percentil
        
        if min_samples is None:
            # Regra empírica: 2 * dimensões ou pelo menos 3
            min_samples = max(3, min(10, 2 * embeddings_scaled.shape[1]))
        
        print(f"Aplicando DBSCAN com eps={eps:.3f}, min_samples={min_samples}...")
        
        # Parâmetros padrão
        default_params = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': 'euclidean'
        }
        default_params.update(kwargs)
        
        self.model = DBSCAN(**default_params)
        self.labels_ = self.model.fit_predict(embeddings_scaled)
        
        # Conta clusters (excluindo outliers -1)
        unique_labels = np.unique(self.labels_)
        self.n_clusters_ = len(unique_labels[unique_labels != -1])
        n_outliers = np.sum(self.labels_ == -1)
        
        print(f"DBSCAN concluído! Clusters: {self.n_clusters_}, Outliers: {n_outliers}")
        return self.labels_
    
    def fit_hierarchical(self, embeddings: np.ndarray, n_clusters: int = None, linkage: str = 'ward', **kwargs) -> np.ndarray:
        """
        Aplica clusterização hierárquica.
        
        Clusterização hierárquica cria uma árvore de clusters,
        permitindo diferentes níveis de granularidade.
        
        Args:
            embeddings: Array com embeddings
            n_clusters: Número de clusters finais
            linkage: Critério de ligação ('ward', 'complete', 'average', 'single')
            **kwargs: Parâmetros adicionais
            
        Returns:
            Array com labels dos clusters
        """
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        if n_clusters is None:
            # Usa análise do cotovelo para estimar
            metrics = self.find_optimal_k(embeddings)
            n_clusters = metrics['k_values'][np.argmax(metrics['silhouette'])]
        
        print(f"🔄 Aplicando Clusterização Hierárquica com {n_clusters} clusters...")
        
        # Parâmetros padrão
        default_params = {
            'n_clusters': n_clusters,
            'linkage': linkage
        }
        default_params.update(kwargs)
        
        self.model = AgglomerativeClustering(**default_params)
        self.labels_ = self.model.fit_predict(embeddings_scaled)
        self.n_clusters_ = n_clusters
        
        print(f"Clusterização Hierárquica concluída! Clusters: {self.n_clusters_}")
        return self.labels_
    
    def fit(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """
        Aplica o algoritmo de clusterização configurado.
        
        Args:
            embeddings: Array com embeddings
            **kwargs: Parâmetros específicos do algoritmo
            
        Returns:
            Array com labels dos clusters
        """
        if self.algorithm == 'kmeans':
            return self.fit_kmeans(embeddings, **kwargs)
        elif self.algorithm == 'dbscan':
            return self.fit_dbscan(embeddings, **kwargs)
        elif self.algorithm == 'hierarchical':
            return self.fit_hierarchical(embeddings, **kwargs)
        else:
            raise ValueError(f"Algoritmo não suportado: {self.algorithm}")
    
    def evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray = None) -> Dict:
        """
        Avalia a qualidade da clusterização usando múltiplas métricas.
        
        Args:
            embeddings: Array com embeddings originais
            labels: Labels dos clusters (usa self.labels_ se None)
            
        Returns:
            Dicionário com métricas de avaliação
        """
        if labels is None:
            labels = self.labels_
        
        if labels is None:
            raise ValueError("Nenhum label de cluster disponível")
        
        embeddings_scaled = self.scaler.transform(embeddings)
        
        # Remove outliers para cálculo das métricas (apenas para DBSCAN)
        if -1 in labels:
            mask = labels != -1
            embeddings_clean = embeddings_scaled[mask]
            labels_clean = labels[mask]
        else:
            embeddings_clean = embeddings_scaled
            labels_clean = labels
        
        metrics = {}
        
        # Métricas básicas
        metrics['n_clusters'] = len(np.unique(labels_clean))
        metrics['n_outliers'] = np.sum(labels == -1) if -1 in labels else 0
        metrics['cluster_sizes'] = Counter(labels_clean)
        
        # Métricas de qualidade (apenas se tivermos mais de 1 cluster)
        if len(np.unique(labels_clean)) > 1:
            metrics['silhouette_score'] = silhouette_score(embeddings_clean, labels_clean)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings_clean, labels_clean)
            metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings_clean, labels_clean)
        else:
            metrics['silhouette_score'] = -1
            metrics['calinski_harabasz_score'] = 0
            metrics['davies_bouldin_score'] = float('inf')
        
        # Inertia (apenas para K-means)
        if hasattr(self.model, 'inertia_'):
            metrics['inertia'] = self.model.inertia_
        
        return metrics
    
    def get_cluster_summaries(self, df: pd.DataFrame, text_column: str = 'processed_text', top_words: int = 10) -> Dict:
        """
        Gera resumos dos clusters baseados nos textos mais representativos.
        
        Args:
            df: DataFrame com os dados originais
            text_column: Coluna com os textos
            top_words: Número de palavras mais frequentes por cluster
            
        Returns:
            Dicionário com resumos dos clusters
        """
        if self.labels_ is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        # Adiciona labels ao DataFrame
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = self.labels_
        
        summaries = {}
        
        for cluster_id in np.unique(self.labels_):
            if cluster_id == -1:  # Outliers
                cluster_name = 'Outliers'
            else:
                cluster_name = f'Cluster {cluster_id}'
            
            # Filtra textos do cluster
            cluster_texts = df_with_clusters[df_with_clusters['cluster'] == cluster_id][text_column]
            
            # Conta palavras mais frequentes
            all_words = ' '.join(cluster_texts).split()
            word_freq = Counter(all_words)
            
            summaries[cluster_name] = {
                'size': len(cluster_texts),
                'top_words': word_freq.most_common(top_words),
                'sample_texts': cluster_texts.head(3).tolist()
            }
        
        return summaries
    
    def print_cluster_summaries(self, summaries: Dict):
        """
        Imprime resumos dos clusters de forma organizada.
        
        Args:
            summaries: Resumos retornados por get_cluster_summaries
        """
        print("\n" + "="*60)
        print("RESUMO DOS CLUSTERS")
        print("="*60)
        
        for cluster_name, info in summaries.items():
            print(f"\n {cluster_name} ({info['size']} textos)")
            print("-" * 40)
            
            print("Palavras mais frequentes:")
            for word, freq in info['top_words'][:5]:
                print(f"   • {word}: {freq}x")
            
            print("\n Exemplos de textos:")
            for i, text in enumerate(info['sample_texts'][:2], 1):
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"   {i}. {preview}")


def main():
    """Função principal para testar o modelo de clusterização."""
    # Gera dados de exemplo
    from embedding_generator import EmbeddingGenerator
    
    sample_texts = [
        "machine learning artificial intelligence neural networks",
        "climate change global warming carbon emissions",
        "economic growth financial markets stock prices",
        "space exploration mars mission astronauts",
        "healthcare medical treatment disease prevention",
        "technology innovation software development",
        "environmental protection renewable energy sustainability",
        "investment banking financial services trading"
    ]
    
    print("=== Testando Clusterização ===")
    
    # Gera embeddings
    generator = EmbeddingGenerator(method='sentence_transformers')
    embeddings = generator.generate_embeddings(sample_texts)
    
    # Testa K-means
    clustering = ClusteringModel(algorithm='kmeans')
    
    # Encontra K ótimo
    metrics = clustering.find_optimal_k(embeddings, k_range=range(2, 6))
    
    # Aplica clusterização
    labels = clustering.fit(embeddings, n_clusters=3)
    
    # Avalia resultados
    evaluation = clustering.evaluate_clustering(embeddings)
    print(f"\nAvaliação da clusterização:")
    for metric, value in evaluation.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.3f}")
        else:
            print(f"   {metric}: {value}")
    
    # Mostra clusters
    df = pd.DataFrame({'processed_text': sample_texts})
    summaries = clustering.get_cluster_summaries(df)
    clustering.print_cluster_summaries(summaries)


if __name__ == "__main__":
    main()