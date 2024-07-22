from .fine_tuning.pipeline import ContentFilter  
import openai
from django.conf import settings

openai.api_key = settings.OPENAI_API_KEY

class AIAssistant:
    def __init__(self):
        self.content_filter = ContentFilter('ia_services/fine_tuning/model')

    def get_age_appropriate_prompt(self, age): 
        if 5 <= age <= 6:
            return "Responde como si hablaras con un niño de 5 a 7 años. Usa oraciones cortas y palabras simples."
        elif 7 <= age <= 8:
            return "Responde como si hablaras con un niño de 7 a 8 años. Usa explicaciones claras y ejemplos sencillos."
        elif 9 <= age <= 10:
            return "Responde como si hablaras con un niño de 9 a 10 años. Usa explicaciones claras y ejemplos sencillos."
        else:
            return "Responde como si hablaras con un preadolescente de 11 a 12 años. Puedes usar conceptos un poco más avanzados, pero mantén un tono amigable y educativo."

    def get_additional_features(self, text):
        # Aquí podrías agregar lógica para derivar las características adicionales basadas en el texto.
        # Por simplicidad, se utilizan valores ficticios.
        return {
            'age_group': 3,  # Ejemplo: 3 corresponde a "11-12 años"
            'phrase_type': 2,  # Ejemplo: 2 corresponde a "negación"
            'intent': 4,  # Ejemplo: 4 corresponde a "negarse"
            'polarity': -0.6,
            'subjectivity': 0.6,
            'complexity': 0.6,
            'context': 1,  # Ejemplo: 1 corresponde a "Protesta contra asistir a la escuela"
            'contains_sarcasm': 0  # 0 = No, 1 = Sí
        }

    def get_response(self, message, age):
        features = self.get_additional_features(message)
        if not self.content_filter.is_appropriate(message, features):
            return "Lo siento, no puedo responder a eso. ¿Podrías hacer una pregunta diferente?"

        age_prompt = self.get_age_appropriate_prompt(age)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"{age_prompt} Asegúrate de que tu respuesta sea educativa, apropiada para niños y fácil de entender."},
                    {"role": "user", "content": message}
                ]
            )
            ai_response = response.choices[0].message['content']
            
            # Aplicar filtro a la respuesta del AI también
            features = self.get_additional_features(ai_response)
            if not self.content_filter.is_appropriate(ai_response, features):
                return "Lo siento, la respuesta generada no es apropiada. ¿Podrías hacer otra pregunta?"
            
            return ai_response
        except Exception as e:
            print(f"Error al obtener respuesta de AI: {e}")
            return "Lo siento, no pude entender eso. ¿Podrías intentar preguntar de otra manera?"

ai_assistant = AIAssistant()
