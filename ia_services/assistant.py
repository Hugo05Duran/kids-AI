import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()


import openai
from django.conf import settings
from transformers import pipeline
from ia_services.content_filter import ContentFilter  


openai.api_key = settings.OPENAI_API_KEY
if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY no está configurado en settings.")


class AIAssistant:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        openai.api_key = self.api_key
        self.content_filter = ContentFilter()
        self.user_history = {}  # Diccionario para almacenar historial de sentimientos
        self.topic_detector = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')  # Detector de temas

    def get_age_appropriate_prompt(self, age):
        if 5 <= age <= 6:
            return "Responde como si hablaras con un niño de 5 a 6 años. Usa oraciones cortas y palabras simples."
        elif 7 <= age <= 8:
            return "Responde como si hablaras con un niño de 7 a 8 años. Usa explicaciones claras y ejemplos sencillos."
        elif 9 <= age <= 10:
            return "Responde como si hablaras con un niño de 9 a 10 años. Usa explicaciones claras y ejemplos sencillos."
        else:
            return "Responde como si hablaras con un preadolescente de 11 a 12 años. Puedes usar conceptos un poco más avanzados, pero mantén un tono amigable y educativo."

    def classify_and_track_sentiment(self, user_id, text):
        classification = self.content_filter.classify_text(text)
        
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        self.user_history[user_id].append(classification)

        return classification

    def detect_topic(self, text):
        topics = ["school", "friends", "family", "hobbies", "health", "emotions"]
        result = self.topic_detector(text, candidate_labels=topics)
        return result['labels'][0]  # Devolvemos el tema con mayor puntuación

    def get_response(self, user_id, message, age):
        classification = self.classify_and_track_sentiment(user_id, message)

        negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'fear', 'grief', 'sadness', 'remorse']
        if any(classification['specific_classification'][emotion] > 0.5 for emotion in negative_emotions):
            return "Parece que estás pasando por algo difícil. ¿Quieres hablar más sobre ello? Estoy aquí para ayudarte."
        else:
            age_prompt = self.get_age_appropriate_prompt(age)
            topic = self.detect_topic(message)  # Detectamos el tema de la conversación
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": f"{age_prompt} Asegúrate de que tu respuesta sea educativa, apropiada para niños y fácil de entender. Tema: {topic}."},
                        {"role": "user", "content": message}
                    ]
                )
                ai_response = response.choices[0].message['content']

                return ai_response
            except Exception as e:
                print(f"Error al obtener respuesta de AI: {e}")
                return "Lo siento, no pude entender eso. ¿Podrías intentar preguntar de otra manera?"

    def get_sentiment_trend(self, user_id):
        if user_id not in self.user_history:
            return "No hay datos suficientes para mostrar una tendencia."
        
        history = self.user_history[user_id]
        trend = {
            'negative': sum(entry['general']['negative'] for entry in history) / len(history),
            'neutral': sum(entry['general']['neutral'] for entry in history) / len(history),
            'positive': sum(entry['general']['positive'] for entry in history) / len(history)
        }
        return trend

ai_assistant = AIAssistant()

