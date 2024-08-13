import os
import django
import openai
import boto3
import time
import requests
from django.conf import settings
from transformers import pipeline
from ia_services.content_filter import ContentFilter  
from botocore.exceptions import BotoCoreError, ClientError

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()


openai.api_key = settings.OPENAI_API_KEY
if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY no está configurado en settings.")


class AIAssistant:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        openai.api_key = self.api_key
        self.content_filter = ContentFilter()
        self.user_history = {}  
        self.transcribe_client = boto3.client('transcribe', region_name='eu-west-1')  # Especifica la región
        self.polly_client = boto3.client('polly', region_name='eu-west-1')


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
        try:
            response = openai.Completion.create(
                engine="davinci",
                prompt=f"Identifica el tema principal de este texto: {text}",
                max_tokens=10
            )
            topic = response.choices[0].text.strip()
            return topic
        except Exception as e:
            print(f"Error al detectar tema: {e}")
            return "No se pudo identificar el tema"

    
    def get_response(self, user_id, message, age):
        classification = self.classify_and_track_sentiment(user_id, message)
        negative_emotions = {
            'anger': [
                "Reconoce el enfado del niño de manera empática.",
                "Valida sus sentimientos, haciéndole saber que es normal sentirse así.",
                "Ofrece estrategias para manejar el enfado, como respirar profundamente o contar hasta diez.",
                "Guía al niño a reflexionar sobre lo que lo hizo enojar y cómo podría manejarlo mejor en el futuro."
            ],
            'annoyance': [
                "Reconoce la molestia del niño.",
                "Explora la causa de la molestia con él.",
                "Proporciona maneras de calmarse, como cambiar de actividad o hacer algo que le guste.",
                "Anima al niño a pensar en cómo podría evitar o manejar mejor estas situaciones en el futuro."
            ],
            'disappointment': [
                "Reconoce la decepción del niño.",
                "Ayúdalo a entender que es normal sentirse así cuando las cosas no salen como uno quiere.",
                "Proporciona ejemplos de cómo superar la decepción y seguir adelante.",
                "Anímalo a pensar en cómo podría manejar sus expectativas y sentimientos la próxima vez."
            ],
            'disapproval': [
                "Reconoce que el niño está expresando desaprobación.",
                "Valida sus sentimientos, recordándole que es válido tener opiniones fuertes.",
                "Explora con él maneras constructivas de expresar su desaprobación.",
                "Guía al niño a pensar en cómo puede comunicar sus opiniones de manera respetuosa y efectiva."
            ],
            'disgust': [
                "Reconoce que el niño siente disgusto.",
                "Explora la causa del disgusto con él.",
                "Ayúdalo a encontrar maneras de lidiar con este sentimiento, como hablar sobre ello o escribirlo.",
                "Anímalo a pensar en cómo manejar estos sentimientos en situaciones futuras."
            ],
            'fear': [
                "Reconoce que el niño siente miedo.",
                "Valida sus sentimientos, haciéndole saber que es normal sentir miedo a veces.",
                "Proporciona maneras de enfrentar el miedo, como hablar sobre ello o pedir ayuda a un adulto.",
                "Guía al niño a pensar en cómo puede sentirse seguro y protegido."
            ],
            'grief': [
                "Reconoce el dolor que el niño está sintiendo.",
                "Valida sus sentimientos, asegurándole que está bien sentir tristeza.",
                "Ofrece apoyo emocional, invitándolo a hablar sobre su pérdida o lo que lo entristece.",
                "Anímalo a recordar momentos felices o cosas que lo hagan sentir mejor."
            ],
            'sadness': [
                "Reconoce la tristeza del niño.",
                "Valida sus sentimientos, haciéndole saber que es normal sentirse triste a veces.",
                "Proporciona maneras de mejorar su estado de ánimo, como hacer algo que le guste o hablar con alguien de confianza.",
                "Guía al niño a pensar en cosas positivas que lo hagan sentir mejor."
            ],
            'remorse': [
                "Reconoce que el niño se siente arrepentido.",
                "Valida sus sentimientos, recordándole que es normal sentirse mal por los errores.",
                "Explora con él maneras de enmendar el error o aprender de la situación.",
                "Guía al niño a pensar en cómo podría evitar cometer el mismo error en el futuro."
            ]
        }

        # Determinar la emoción predominante
        predominant_emotion = max(classification['specific_classification'], key=classification['specific_classification'].get)
        
        if classification['specific_classification'][predominant_emotion] > 0.15:
            instructions = negative_emotions.get(predominant_emotion, [])
            # Construir el prompt para ChatGPT
            prompt = f"El niño está experimentando {predominant_emotion}. " + " ".join(instructions) + f" Responde como si hablaras con un niño de {age} años, asegurándote de ser educativo y apropiado."
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": message}
                    ]
                )
                ai_response = response.choices[0].message['content']
                return ai_response
            except Exception as e:
                print(f"Error al obtener respuesta de AI: {e}")
                return "Lo siento, no pude entender eso. ¿Podrías intentar preguntar de otra manera?"
        else:
            # Sin emociones negativas detectadas, proceder normalmente
            age_prompt = self.get_age_appropriate_prompt(age)
            topic = self.detect_topic(message)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
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
        total_entries = len(history)
        trend = {
            'negative': sum(entry['general']['negative'] for entry in history) / total_entries,
            'neutral': sum(entry['general']['neutral'] for entry in history) / total_entries,
            'positive': sum(entry['general']['positive'] for entry in history) / total_entries
        }
        return trend

    # Amazon
    def transcribe_audio(self, audio_content):
        job_name = "transcription_job"
        self.transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': audio_content},
            MediaFormat='wav',
            LanguageCode='en-US'
        )

        while True:
            status = self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcript_file_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcript = requests.get(transcript_file_uri).json()
            return transcript['results']['transcripts'][0]['transcript']
        else:
            raise ValueError("Transcription job failed")

    def synthesize_speech(self, text):
        try:
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId='Joanna'
            )
            return response['AudioStream'].read()
        except (BotoCoreError, ClientError) as error:
            print(f"Error al sintetizar discurso: {error}")
            return None

    def process_audio_input(self, user_id, audio_content, age):
        transcript = self.transcribe_audio(audio_content)
        classification = self.classify_and_track_sentiment(user_id, transcript)
        
        negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'fear', 'grief', 'sadness', 'remorse']
        
        # Inicializamos las instrucciones como None
        emotion_instructions = None
        
        # Verificamos si alguna emoción negativa es superior a 0.5
        for emotion in negative_emotions:
            if classification['specific_classification'][emotion] > 0.15:
                emotion_instructions = self.get_instructions_for_emotion(emotion)
                break
        
        # Generamos la respuesta basada en las instrucciones si es que existen
        if emotion_instructions:
            response_text = self.get_response(user_id, transcript, age, emotion_instructions)
        else:
            response_text = self.get_response(user_id, transcript, age)
        
        audio_response = self.synthesize_speech(response_text)
        return audio_response


ai_assistant = AIAssistant()

