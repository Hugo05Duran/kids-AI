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

    def get_response(self, message, age):
        classification = self.content_filter.classify_text(message)

        if classification['is_obscene']:
            return "Por favor, usa un lenguaje apropiado."

        elif classification['is_emotionally_negative']:
            return "Parece que estás pasando por algo difícil. ¿Quieres hablar más sobre ello o pedir ayuda a un adulto?"
        
        elif classification['is_appropriate']:

            age_prompt = self.get_age_appropriate_prompt(age)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": f"{age_prompt} Asegúrate de que tu respuesta sea educativa, apropiada para niños y fácil de entender."},
                        {"role": "user", "content": message}
                    ]
                )
                ai_response = response.choices[0].message['content']

                # Aplicar filtro a la respuesta del AI también
                response_classification = self.content_filter.classify_text(ai_response)
                if not response_classification['is_appropriate']:
                    return "Lo siento, la respuesta generada no es apropiada. ¿Podrías hacer otra pregunta?"

                return ai_response
            except Exception as e:
                print(f"Error al obtener respuesta de AI: {e}")
                return "Lo siento, no pude entender eso. ¿Podrías intentar preguntar de otra manera?"

ai_assistant = AIAssistant()
