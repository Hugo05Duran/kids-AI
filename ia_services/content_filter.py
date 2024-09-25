import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

class ContentFilter:
    def __init__(self):
        # Usamos DistilRoBERTa para ambas clasificaciones, más ligero
        self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        self.model_general = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-distilroberta-sentiment")
        self.model_specific = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")  # Puedes cambiar esto a un modelo más ligero si es necesario

        self.dangerous_phrases = [
            "cross the railway tracks", "play with fire", "run into the street",
            "talk to strangers", "take unknown pills", "jump from high places"
        ]

    def classify_general(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.model_general(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        classification = {
            'negative': scores[0],
            'neutral': scores[1],
            'positive': scores[2]
        }

        if any(phrase in text.lower() for phrase in self.dangerous_phrases):
            classification['negative'] = max(classification['negative'], 0.7)
            classification['neutral'] = min(classification['neutral'], 0.3)
        
        return classification

    def classify_specific(self, text, general_classification):
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.model_specific(**encoded_input)
        logits = output.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].detach().numpy()
        
        if general_classification == 'positive':
            specific_classification = {
                'admiration': probabilities[0],
                'amusement': probabilities[1],
                'approval': probabilities[2],
                'caring': probabilities[3],
                'curiosity': probabilities[4],
                'desire': probabilities[5],
                'excitement': probabilities[6],
                'gratitude': probabilities[7],
                'joy': probabilities[8],
                'love': probabilities[9],
                'optimism': probabilities[10],
                'pride': probabilities[11],
                'realization': probabilities[12],
                'relief': probabilities[13],
                'surprise': probabilities[14]
            }
        elif general_classification == 'neutral':
            specific_classification = {
                'neutral': probabilities[15],
                'nervousness': probabilities[16],
                'confusion': probabilities[17]
            }
        elif general_classification == 'negative':
            specific_classification = {
                'anger': probabilities[18],
                'annoyance': probabilities[19],
                'disappointment': probabilities[20],
                'disapproval': probabilities[21],
                'disgust': probabilities[22],
                'fear': probabilities[23],
                'grief': probabilities[24],
                'sadness': probabilities[25],
                'remorse': probabilities[26]
            }
        
        return specific_classification

    def classify_text(self, text):
        general_classification = self.classify_general(text)
        general_label = max(general_classification, key=general_classification.get)
        
        specific_classification = self.classify_specific(text, general_label)
        
        return {
            'general': general_classification,
            'specific': specific_classification
        }

content_filter = ContentFilter()
