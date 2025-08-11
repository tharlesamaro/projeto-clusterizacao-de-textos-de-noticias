"""
Módulo para coleta de dados de notícias usando RSS feeds.

Este módulo demonstra commo coletar dados para clusterização de textos.
Coleta notícias de diferentes fontes para ter uma variedade de tópicos.
"""

import feedparser
import pandas as pd
import requests
import time
from datetime import datetime
import os


class NewsCollector:
    """
    Classe para coletar notícias de várias fontes RSS.

    RSS (Really Simple Syndication) é um formato que sites usam
    para compartilhar suas últimas notícias automaticamente.
    """

    def __init__(self):
        self.rss_feeds = [
            {
                "name": "BBC News",
                "url": "http://feeds.bbci.co.uk/news/rss.xml",
                "category": "general"
            },
            {
                "name": "CNN",
                "url": "http://rss.cnn.com/rss/edition.rss",
                "category": "general"
            },
            {
                'name': 'TechCrunch',
                'url': 'https://techcrunch.com/feed/',
                'category': 'technology'
            },
            {
                'name': 'Reuters',
                'url': 'https://www.reutersagency.com/feed/?best-regions=north-america&post_type=best',
                'category': 'general'
            }
        ]

    def collect_news(self, max_articles: int = 100) -> pd.DataFrame:
        """
        Coleta notícias de todos os feeds RSS configurados.
        
        Args:
            max_articles: Número máximo de artigos por fonte
            
        Returns:
            DataFrame com as notícias coletadas
        """
        all_articles = []

        print("Iniciando coleta de notícias...")

        for feed_info in self.rss_feeds:
            print(f"Coletando de {feed_info['name']}...")

            try:
                # Faz o parse do feed RSS
                # feedparse é uma biblioteca que entende o formato XML dos feeds RSS
                feed = feedparser.parse(feed_info['url'])

                # Extrai informações de cada artigo
                for i, entry in enumerate(feed.entries[:max_articles]):
                    article = {
                        'title': entry.get('title', ''),
                        'description': entry.get('description', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': feed_info['name'],
                        'category': feed_info['category'],
                        'collected_at': datetime.now().isoformat()
                    }

                    # Remove tags HTML da descrição se houver
                    article['description'] = self._clean_html(article['description'])
                    
                    all_articles.append(article)
                
                print(f"Coletados {len(feed.entries[:max_articles])} artigos de {feed_info['name']}")
                
                # Pausa para evitar sobrecarga no servidor
                time.sleep(1)
                
            except Exception as e:
                print(f"Erro ao coletar de {feed_info['name']}: {str(e)}")
                continue
        
        print(f"Coleta finalizada! Total: {len(all_articles)} artigos")
        
        return pd.DataFrame(all_articles)


    def _clean_html(self, text: str) ->str:
        """
        Remove tags HTML básicas do texto.
        Precisamos remover para ter apenas o texto limpo para análise.
        """
        import re
        # Remove tags HTML simples
        clean = re.sub('<.*?>', '', text)
        # Remove caracteres especiais HTML
        clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&')
        return clean.strip()

    def save_data(self, df: pd.DataFrame, filename: str = "news_data.csv") -> None:
        """
        Salva os dados coletados em arquivo CSV.
        
        Args:
            df: DataFrame com os dados
            filename: Nome do arquivo para salvar
        """
        # Cria o diretório se não existir
        os.makedirs('data/raw', exist_ok=True)
        
        filepath = f'data/raw/{filename}'
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Dados salvos em {filepath}")

    def load_sample_data(self) -> pd.DataFrame:
        """
        Carrega dados de exemplo caso não consiga coletar dos RSS feeds.
        
        Isso é útil para garantir que o projeto funcione mesmo se
        houver problemas com os feeds externos.
        """
        sample_articles = [
            {
                'title': 'New AI Technology Breakthrough',
                'description': 'Scientists develop new artificial intelligence system that can understand natural language better than ever before.',
                'source': 'Tech News',
                'category': 'technology'
            },
            {
                'title': 'Climate Change Impact on Oceans',
                'description': 'Research shows significant changes in ocean temperatures affecting marine life worldwide.',
                'source': 'Science Daily',
                'category': 'science'
            },
            {
                'title': 'Economic Growth Forecast',
                'description': 'Economists predict steady growth for the next quarter despite global uncertainties.',
                'source': 'Financial Times',
                'category': 'economics'
            },
            {
                'title': 'Space Exploration Mission Success',
                'description': 'Latest space mission successfully lands on Mars, sending back unprecedented images.',
                'source': 'Space News',
                'category': 'science'
            },
            {
                'title': 'Healthcare Innovation',
                'description': 'New medical device promises to revolutionize treatment for chronic diseases.',
                'source': 'Medical Journal',
                'category': 'health'
            }
        ]
        
        return pd.DataFrame(sample_articles)


def main():
    """Função principal para testar o coletor de dados."""
    collector = NewsCollector()
    
    # Tenta coletar dados reais
    try:
        news_df = collector.collect_news(max_articles=50)
        if len(news_df) == 0:
            print("Nenhum dado coletado, usando dados de exemplo...")
            news_df = collector.load_sample_data()
    except:
        print("Erro na coleta, usando dados de exemplo...")
        news_df = collector.load_sample_data()
    
    collector.save_data(news_df)
    
    print(f"\nResumo dos dados coletados:")
    print(f"Total de artigos: {len(news_df)}")
    print(f"Fontes: {news_df['source'].unique()}")
    print(f"\nPrimeiros artigos:")
    print(news_df[['title', 'source']].head())


if __name__ == "__main__":
    main()
