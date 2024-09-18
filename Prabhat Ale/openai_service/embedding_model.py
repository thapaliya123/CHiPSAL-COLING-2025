import openai

from config import (
    api_key
)

class EmbeddingModel:
    
    openai.api_key = api_key

    def __init__(self, embedding_model_name):
        """
        Initialize the ProductSentimentAnalyzer with model settings and API key.
        """

        self.embedding_model_name = embedding_model_name

    def extract_embeddings(self, text):
        self.text = text
        response = openai.Embedding.create(
            input=f"{self.text}",
            model= self.embedding_model_name  # Ada model for embeddings
        )
        return response['data'][0]['embedding']

