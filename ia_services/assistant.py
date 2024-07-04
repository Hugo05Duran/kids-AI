import openai
from django.conf import settings
from .content_filter import content_filter

openai.api_key = settings.OPENAI_API_KEY

class AIAssistant:
    def __init__(self):
        self.content_filter = content_filter

    def get_age_appropriate_prompt(self, age):
        if 5 <= age <= 7:
            return "Responde como si hablaras con un niño de 5 a 7 años. Usa oraciones cortas y palabras simples."
        elif 8 <= age <= 10:
            return "Responde como si hablaras con un niño de 8 a 10 años. Usa explicaciones claras y ejemplos sencillos."
        else:
            return "Responde como si hablaras con un preadolescente de 11 a 12 años. Puedes usar conceptos un poco más avanzados, pero mantén un tono amigable y educativo."

    def get_response(self, message, age):
        # Verificar si el mensaje contiene contenido inapropiado
        if not self.content_filter.is_appropriate(message):
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
            
            # Verificar si la respuesta generada por la IA contiene contenido inapropiado
            if not self.content_filter.is_appropriate(ai_response):
                return "Lo siento, la respuesta generada no es apropiada. ¿Podrías intentar preguntar de otra manera?"

            # Censurar cualquier palabra inapropiada en la respuesta de la IA
            filtered_response = self.content_filter.filter_text(ai_response)
            return filtered_response
        except Exception as e:
            print(f"Error al obtener respuesta de AI: {e}")
            return "Lo siento, no pude entender eso. ¿Podrías intentar preguntar de otra manera?"

ai_assistant = AIAssistant()
