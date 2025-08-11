"""
Módulo para pré-processamento de textos.

O pré-processamento é crucial em ML de texto. Aqui limpamos e preparamos
os textos para que os algoritmos possam trabalhar melhor com eles.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import spacy


class TextPreprocessor:
    """
    Classe para pré-processamento de textos para clusterização.
    
    O pré-processamento inclui:
    1. Limpeza básica (remoção de caracteres especiais, etc.)
    2. Normalização (minúsculas, remoção de stopwords)
    3. Tokenização (divisão em palavras)
    4. Stemming/Lemmatização (redução à raiz das palavras)
    """

    def __init__(self, language: str = 'english'):
        self.language = language
        self.stemmer = PorterStemmer()
        
        # Download dos recursos necessários do NLTK
        self._download_nltk_resources()
        
        # Carrega spaCy se disponível (mais avançado que NLTK)
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.use_spacy = True
            print("spaCy carregado com sucesso")
        except OSError:
            print("spaCy não encontrado, usando NLTK")
            self.use_spacy = False
            self.nlp = None

    
    def _download_nltk_resources(self):
        """Download dos recursos necessários do NLTK."""
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Baixando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)


    def clean_text(self, text: str) -> str:
        """
        Limpa o texto removendo elementos desnecessários.
        
        Args:
            text: Texto para limpar
            
        Returns:
            Texto limpo
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove números (opcional, depende do caso de uso)
        text = re.sub(r'\d+', '', text)
        
        # Remove pontuação excessiva (mas mantém pontos e vírgulas)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove espaços extras
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    

    def normalize_text(self, text: str) -> str:
        """
        Normaliza o texto (minúsculas, etc.).
        
        Args:
            text: Texto para normalizar
            
        Returns:
            Texto normalizado
        """
        # Converte para minúsculas
        text = text.lower()
        
        # Remove acentos (opcional)
        # text = unidecode(text)  # Descomente se quiser remover acentos
        
        return text
    

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """
        Remove stopwords (palavras muito comuns como 'a', 'the', 'is').
        
        Stopwords são palavras que aparecem muito frequentemente
        mas não carregam muito significado para análise.
        
        Args:
            tokens: Lista de palavras
            
        Returns:
            Lista de palavras sem stopwords
        """
        try:
            stop_words = set(stopwords.words(self.language))
        except:
            # Fallback caso não tenha o idioma
            stop_words = set(stopwords.words('english'))
        
        # Adiciona algumas stopwords customizadas
        custom_stopwords = {'said', 'would', 'could', 'one', 'two', 'also', 'new'}
        stop_words.update(custom_stopwords)
        
        return [token for token in tokens if token.lower() not in stop_words and len(token) > 2]
    

    def tokenize_and_lemmatize(self, text: str) -> list[str]:
        """
        Tokeniza o texto e aplica lemmatização.
        
        Tokenização: divide o texto em palavras individuais
        Lemmatização: reduz palavras à sua forma base (ex: 'running' -> 'run')
        
        Args:
            text: Texto para processar
            
        Returns:
            Lista de tokens lemmatizados
        """
        if self.use_spacy and self.nlp:
            # Usa spaCy (mais avançado)
            doc = self.nlp(text)
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if not token.is_space 
                and not token.is_punct 
                and token.is_alpha
                and len(token.text) > 2
            ]
        else:
            # Usa NLTK (mais básico)
            tokens = word_tokenize(text)
            # Aplica stemming (menos preciso que lemmatização)
            tokens = [
                self.stemmer.stem(token.lower())
                for token in tokens
                if token.isalpha() and len(token) > 2
            ]
        
        return tokens
    

    def preprocess_text(self, text: str) -> str:
        """
        Pipeline completo de pré-processamento.
        
        Este é o método principal que será usado na maioria dos casos.
        Ele aplica todas as etapas de limpeza em sequência.
        
        Args:
            text: Texto original
            
        Returns:
            Texto pré-processado
        """
        text = self.clean_text(text)
        
        text = self.normalize_text(text)
        
        tokens = self.tokenize_and_lemmatize(text)
        
        tokens = self.remove_stopwords(tokens)
        
        # Reconstroi o texto
        return ' '.join(tokens)
    

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Aplica pré-processamento a uma coluna de texto em um DataFrame.
        
        Args:
            df: DataFrame com os dados
            text_column: Nome da coluna com texto
            
        Returns:
            DataFrame com coluna adicional de texto processado
        """
        print("Iniciando pré-processamento dos textos...")
        
        # Combina título e descrição se disponível
        if 'title' in df.columns and 'description' in df.columns:
            df['combined_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
            text_col = 'combined_text'
        else:
            text_col = text_column
        
        # Aplica pré-processamento
        from tqdm import tqdm
        tqdm.pandas(desc="Processando textos")
        
        df['processed_text'] = df[text_col].progress_apply(self.preprocess_text)
        
        # Remove textos muito curtos (menos úteis para clusterização)
        initial_size = len(df)
        df = df[df['processed_text'].str.len() > 10].copy()
        removed = initial_size - len(df)
        
        if removed > 0:
            print(f"Removidos {removed} textos muito curtos")
        
        print(f"Pré-processamento concluído! {len(df)} textos processados")
        
        return df
    

    def get_text_statistics(self, df: pd.DataFrame, text_column: str = 'processed_text') -> dict:
        """
        Calcula estatísticas básicas dos textos processados.
        
        Args:
            df: DataFrame com textos
            text_column: Nome da coluna com texto
            
        Returns:
            Dicionário com estatísticas
        """
        texts = df[text_column].dropna()
        
        word_counts = texts.str.split().str.len()
        
        stats = {
            'total_texts': len(texts),
            'avg_words_per_text': word_counts.mean(),
            'min_words': word_counts.min(),
            'max_words': word_counts.max(),
            'total_unique_words': len(set(' '.join(texts).split()))
        }
        
        return stats
    

def main():
    """Função principal para testar o pré-processador."""
    # Exemplo de uso
    preprocessor = TextPreprocessor()
    
    # Texto de exemplo
    sample_text = """
    This is a sample news article about artificial intelligence and machine learning!
    AI technology is revolutionizing the world, and companies are investing heavily in ML research.
    Visit https://example.com for more information. Contact us at info@example.com.
    """
    
    print("Texto original:")
    print(sample_text)
    
    print("\nTexto processado:")
    processed = preprocessor.preprocess_text(sample_text)
    print(processed)
    
    # Exemplo com DataFrame
    sample_df = pd.DataFrame({
        'title': ['AI Revolution', 'Climate Change News', 'Economic Update'],
        'description': [
            'New AI systems are changing how we work',
            'Global warming affects weather patterns worldwide',
            'Stock markets show positive trends this quarter'
        ]
    })
    
    processed_df = preprocessor.preprocess_dataframe(sample_df)
    print("\nDataFrame processado:")
    print(processed_df[['title', 'processed_text']])
    
    # Estatísticas
    stats = preprocessor.get_text_statistics(processed_df)
    print(f"\nEstatísticas: {stats}")


if __name__ == "__main__":
    main()
