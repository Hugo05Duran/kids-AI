import re
import spacy
from better_profanity import profanity
from textblob import TextBlob

class ContentFilter:
    def __init__(self, custom_badwords=None):
        self.nlp = spacy.load("es_core_news_sm")
        if custom_badwords is None:
            custom_badwords = ['palabrota1', 'palabrota2', 'contenido_inapropiado']
        self.custom_badwords = set(custom_badwords)
        profanity.add_censor_words(custom_badwords)

    def contains_inappropriate_content(self, text):
        """
        Verifies if the text contains inappropriate content.
        """
        doc = self.nlp(text)

        # Check for inappropriate words
        if profanity.contains_profanity(text):
            return True

        # Check inappropriate context 
        if self.detect_inappropriate_context(doc):
            return True
        
        # Verificar patrones de información personal
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN (Número de Seguridad Social)
            r'\b\d{9}\b',  # Número de teléfono en España (puede necesitar ajuste según formato)
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Correo electrónico
            r'\b\d{4} \d{4} \d{4} \d{4}\b',  # Número de tarjeta de crédito (regex simple)
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

    def detect_inappropriate_context(self, doc):
        """
        Verifica si el texto contiene contexto inapropiado utilizando análisis de sentimientos y palabras clave.
        """
        blob = TextBlob(doc.text)
        sentiment = blob.sentiment.polarity
        if sentiment < -0.5:  # Umbral para detectar contenido negativo
            return True
        
        # Analizar entidades y detectar contexto inapropiado
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:  # Etiquetas que podrían indicar contenido inapropiado
                if ent.text.lower() in self.custom_badwords:
                    return True
        
        return False

    def filter_text(self, text):
        """
        Censura palabras inapropiadas en el texto.
        """
        return profanity.censor(text)

    def is_appropriate(self, text):
        """
        Verifica si el texto es apropiado asegurando que no contiene contenido inapropiado.
        """
        return not self.contains_inappropriate_content(text)


content_filter = ContentFilter()
