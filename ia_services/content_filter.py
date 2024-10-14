import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

class ContentFilter:
    def __init__(self):
        # Tokenizadores específicos para cada modelo
        self.tokenizer_general = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.tokenizer_specific = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        
        # Modelos de clasificación
        self.model_general = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model_specific = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        
        # Obtener las etiquetas del modelo específico
        self.specific_labels = self.model_specific.config.id2label
        
        # Mapeo de emociones a categorías generales
        self.emotion_categories = {
            'admiration': 'positive',
            'amusement': 'positive',
            'approval': 'positive',
            'caring': 'positive',
            'desire': 'positive',
            'excitement': 'positive',
            'gratitude': 'positive',
            'joy': 'positive',
            'love': 'positive',
            'optimism': 'positive',
            'pride': 'positive',
            'relief': 'positive',
            'curiosity': 'neutral',
            'realization': 'neutral',
            'surprise': 'neutral',
            'confusion': 'neutral',
            'neutral': 'neutral',
            'anger': 'negative',
            'annoyance': 'negative',
            'disapproval': 'negative',
            'disappointment': 'negative',
            'disgust': 'negative',
            'embarrassment': 'negative',
            'fear': 'negative',
            'grief': 'negative',
            'nervousness': 'negative',
            'remorse': 'negative',
            'sadness': 'negative',
            'shame': 'negative'
        }

    def classify_general(self, text):
        encoded_input = self.tokenizer_general(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.model_general(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        classification = {
            'negative': scores[0],
            'neutral': scores[1],
            'positive': scores[2]
        }
        return classification

    def classify_specific(self, text):
        encoded_input = self.tokenizer_specific(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.model_specific(**encoded_input)
        logits = output.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].detach().numpy()

        # Crear un diccionario de etiqueta: probabilidad
        specific_classification = {self.specific_labels[i]: probabilities[i] for i in range(len(probabilities))}

        # Agrupar las emociones bajo categorías generales
        grouped_emotions = {'positive': {}, 'neutral': {}, 'negative': {}}
        for emotion, probability in specific_classification.items():
            category = self.emotion_categories.get(emotion, 'neutral')
            grouped_emotions[category][emotion] = probability

        return grouped_emotions

    def classify_text(self, text):
        general_classification = self.classify_general(text)
        specific_classification = self.classify_specific(text)
        return {
            'general': general_classification,
            'specific': specific_classification
        }

content_filter = ContentFilter()

