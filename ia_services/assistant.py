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
        self.transcribe_client = boto3.client('transcribe', region_name='us-east-1')  # Especifica la región
        self.polly_client = boto3.client('polly', region_name='us-east-1')


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

        negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'fear', 'grief', 'sadness', 'remorse']
        if any(classification['specific_classification'][emotion] > 0.15 for emotion in negative_emotions):
            return "Parece que estás pasando por algo difícil. ¿Quieres hablar más sobre ello? Estoy aquí para ayudarte."
        else:
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
        if any(classification['specific_classification'][emotion] > 0.5 for emotion in negative_emotions):
            response_text = "Parece que estás pasando por algo difícil. ¿Quieres hablar más sobre ello? Estoy aquí para ayudarte."
        else:
            response_text = self.get_response(user_id, transcript, age)
        audio_response = self.synthesize_speech(response_text)
        return audio_response
    

ai_assistant = AIAssistant()

''' Google

def transcribe_audio(self, audio_content):
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        response = self.speech_client.recognize(config=config, audio=audio)
        return response.results[0].alternatives[0].transcript

    def synthesize_speech(self, text):
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        )
        response = self.tts_client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        return response.audio_content

    def process_audio_input(self, user_id, audio_content, age):
        transcript = self.transcribe_audio(audio_content)
        classification = self.classify_and_track_sentiment(user_id, transcript)
        negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'fear', 'grief', 'sadness', 'remorse']
        if any(classification['specific_classification'][emotion] > 0.5 for emotion in negative_emotions):
            response_text = "Parece que estás pasando por algo difícil. ¿Quieres hablar más sobre ello? Estoy aquí para ayudarte."
        else:
            response_text = self.get_response(user_id, transcript, age)
        audio_response = self.synthesize_speech(response_text)
        return audio_response

'''