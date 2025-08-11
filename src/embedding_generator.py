"""
Módulo para geração de embeddings de texto.

Embeddings são representações numéricas de texto que capturam o significado semântico.
Eles são essenciais para clusterização de textos, pois permitem que algoritmos
trabalhem com a "distância" entre diferentes textos.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


class EmbeddingGenerator:
    """
    Classe para gerar embeddings de texto usando diferentes métodos.
    
    Oferecemos duas abordagens:
    1. TF-IDF: Método tradicional baseado em frequência de palavras
    2. Sentence Transformers: Método moderno usando modelos pré-treinados
    """
    
    def __init__(self, method: str = 'sentence_transformers', model_name: str = 'all-MiniLM-L6-v2'):
        """
        Inicializa o gerador de embeddings.
        
        Args:
            method: 'tfidf' ou 'sentence_transformers'
            model_name: Nome do modelo para sentence transformers
        """
        self.method = method
        self.model_name = model_name
        
        if method == 'sentence_transformers':
            self._init_sentence_transformer()
        elif method == 'tfidf':
            self._init_tfidf()
        else:
            raise ValueError("Método deve ser 'tfidf' ou 'sentence_transformers'")
    
    def _init_sentence_transformer(self):
        """
        Inicializa o modelo Sentence Transformer.
        
        Sentence Transformers são modelos baseados em BERT que foram
        treinados especificamente para gerar embeddings de qualidade
        para sentenças e parágrafos.
        """
        print(f"Carregando modelo Sentence Transformer: {self.model_name}")
        
        try:
            # Modelos populares:
            # - 'all-MiniLM-L6-v2': Pequeno e rápido (384 dimensões)
            # - 'all-mpnet-base-v2': Maior e mais preciso (768 dimensões)
            # - 'paraphrase-multilingual-MiniLM-L12-v2': Multilíngue
            
            self.model = SentenceTransformer(self.model_name)
            print(f"Modelo carregado com sucesso!")
            print(f"   - Dimensões do embedding: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            print("Tentando modelo fallback...")
            
            # Fallback para um modelo mais básico
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Modelo fallback carregado!")
            except:
                raise Exception("Não foi possível carregar nenhum modelo Sentence Transformer")
    
    def _init_tfidf(self):
        """
        Inicializa o vetorizador TF-IDF.
        
        TF-IDF (Term Frequency-Inverse Document Frequency) é um método
        tradicional que mede a importância de uma palavra em um documento
        dentro de uma coleção de documentos.
        """
        print("Inicializando TF-IDF Vectorizer...")
        
        self.model = TfidfVectorizer(
            max_features=5000,      # Limita o vocabulário às 5000 palavras mais importantes
            ngram_range=(1, 2),     # Usa unigramas e bigramas
            min_df=2,               # Palavra deve aparecer em pelo menos 2 documentos
            max_df=0.8,             # Palavra não pode aparecer em mais de 80% dos documentos
            stop_words='english'    # Remove stopwords automaticamente
        )
        
        print("TF-IDF Vectorizer inicializado!")
    
    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.
        
        Args:
            texts: Lista de textos para converter em embeddings
            
        Returns:
            Array numpy com os embeddings (cada linha é um texto)
        """
        if not texts:
            raise ValueError("Lista de textos não pode estar vazia")
        
        print(f"Gerando embeddings para {len(texts)} textos usando {self.method}...")
        
        if self.method == 'sentence_transformers':
            # Usa o modelo pré-treinado para gerar embeddings
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,  # Processa em lotes para ter mais eficiência
                convert_to_numpy=True
            )
            
        elif self.method == 'tfidf':
            # Treina o TF-IDF nos textos e gera embeddings
            embeddings = self.model.fit_transform(texts).toarray()
        
        print(f"Embeddings gerados! Shape: {embeddings.shape}")
        return embeddings
    
    def generate_embeddings_from_dataframe(self, df: pd.DataFrame, text_column: str = 'processed_text') -> np.ndarray:
        """
        Gera embeddings diretamente de um DataFrame.
        
        Args:
            df: DataFrame com os textos
            text_column: Nome da coluna com os textos
            
        Returns:
            Array numpy com os embeddings
        """
        texts = df[text_column].fillna('').tolist()
        return self.generate_embeddings(texts)
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str = 'embeddings.pkl'):
        """
        Salva os embeddings em arquivo para reutilização.
        
        Args:
            embeddings: Array com os embeddings
            filename: Nome do arquivo
        """
        os.makedirs('data/processed', exist_ok=True)
        filepath = f'data/processed/{filename}'
        
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"Embeddings salvos em {filepath}")
    
    def load_embeddings(self, filename: str = 'embeddings.pkl') -> np.ndarray:
        """
        Carrega embeddings salvos anteriormente.
        
        Args:
            filename: Nome do arquivo
            
        Returns:
            Array numpy com os embeddings
        """
        filepath = f'data/processed/{filename}'
        
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        
        print(f"Embeddings carregados de {filepath}")
        return embeddings
    
    def reduce_dimensions(self, embeddings: np.ndarray, method: str = 'pca', n_components: int = 50) -> np.ndarray:
        """
        Reduz a dimensionalidade dos embeddings.
        
        Isso é útil para:
        1. Acelerar algoritmos de clusterização
        2. Reduzir ruído nos dados
        3. Permitir visualização em 2D/3D
        
        Args:
            embeddings: Embeddings originais
            method: 'pca' ou 'tsne'
            n_components: Número de dimensões finais
            
        Returns:
            Embeddings com dimensionalidade reduzida
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        print(f"Reduzindo dimensionalidade de {embeddings.shape[1]} para {n_components} usando {method.upper()}...")
        
        if method == 'pca':
            # PCA: Linear, rápido, preserva variância global
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            
            # Mostra quanta variância foi preservada
            explained_var = reducer.explained_variance_ratio_.sum()
            print(f"PCA preservou {explained_var:.2%} da variância original")
            
        elif method == 'tsne':
            # t-SNE: Não-linear, mais lento, preserva estrutura local
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(30, len(embeddings) - 1),  # Ajusta perplexity se dataset for pequeno
                n_iter=1000
            )
            reduced_embeddings = reducer.fit_transform(embeddings)
            print("t-SNE aplicado com sucesso")
            
        else:
            raise ValueError("Método deve ser 'pca' ou 'tsne'")
        
        print(f"Dimensionalidade reduzida! Nova shape: {reduced_embeddings.shape}")
        return reduced_embeddings
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de similaridade entre todos os embeddings.
        
        Isso mostra quão similares são os textos entre si.
        Útil para análise exploratória.
        
        Args:
            embeddings: Array com embeddings
            
        Returns:
            Matriz de similaridade
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("Calculando matriz de similaridade...")
        
        # Usa similaridade do cosseno (padrão para embeddings)
        similarity_matrix = cosine_similarity(embeddings)
        
        print(f"Matriz de similaridade calculada! Shape: {similarity_matrix.shape}")
        return similarity_matrix
    
    def find_most_similar_texts(self, embeddings: np.ndarray, texts: list[str], query_idx: int, top_k: int = 5) -> list[tuple]:
        """
        Encontra os textos mais similares a um texto de referência.
        
        Args:
            embeddings: Array com embeddings
            texts: Lista original de textos
            query_idx: Índice do texto de referência
            top_k: Quantos textos similares retornar
            
        Returns:
            Lista de tuplas (índice, similaridade, texto)
        """
        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        
        # Pega similaridades do texto de referência
        similarities = similarity_matrix[query_idx]
        
        # Ordena por similaridade (excluindo o próprio texto)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in similar_indices:
            results.append((idx, similarities[idx], texts[idx]))
        
        return results
    
    def get_embedding_statistics(self, embeddings: np.ndarray) -> dict:
        """
        Calcula estatísticas dos embeddings gerados.
        
        Args:
            embeddings: Array com embeddings
            
        Returns:
            Dicionário com estatísticas
        """
        stats = {
            'shape': embeddings.shape,
            'mean': np.mean(embeddings),
            'std': np.std(embeddings),
            'min': np.min(embeddings),
            'max': np.max(embeddings),
            'sparsity': np.mean(embeddings == 0)  # Porcentagem de zeros
        }
        
        return stats


def main():
    """Função principal para testar o gerador de embeddings."""

    sample_texts = [
        "artificial intelligence machine learning technology",
        "climate change global warming environment",
        "economic growth financial markets investment",
        "space exploration mars mission discovery",
        "healthcare medical innovation treatment"
    ]
    
    print("=== Testando Sentence Transformers ===")
    
    # Testa Sentence Transformers
    generator_st = EmbeddingGenerator(method='sentence_transformers')
    embeddings_st = generator_st.generate_embeddings(sample_texts)
    
    print(f"Embeddings ST shape: {embeddings_st.shape}")
    
    similar_texts = generator_st.find_most_similar_texts(
        embeddings_st, sample_texts, query_idx=0, top_k=2
    )
    
    print(f"\nTextos mais similares ao primeiro:")
    for idx, sim, text in similar_texts:
        print(f"  Similaridade {sim:.3f}: {text}")
    
    print("\n=== Testando TF-IDF ===")
    
    # Testa TF-IDF
    generator_tfidf = EmbeddingGenerator(method='tfidf')
    embeddings_tfidf = generator_tfidf.generate_embeddings(sample_texts)
    
    print(f"Embeddings TF-IDF shape: {embeddings_tfidf.shape}")
    
    # Redução de dimensionalidade
    if embeddings_st.shape[1] > 10:
        reduced = generator_st.reduce_dimensions(embeddings_st, method='pca', n_components=10)
        print(f"Embeddings reduzidos shape: {reduced.shape}")
    
    # Estatísticas
    stats = generator_st.get_embedding_statistics(embeddings_st)
    print(f"\nEstatísticas dos embeddings: {stats}")


if __name__ == "__main__":
    main()
