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
from model_selection import ComplexityClassifier



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
        self.model_selection = ComplexityClassifier()
        self.model_map = {
            "simple": "gpt-3.5-turbo",
            "moderate": "gpt-4-mini",
            "complex": "gpt-4o"
        }


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
        complexity, _ = self.model_selection.calculate_complexity(message)
        selected_model = self.model_map[complexity]

        negative_emotions = {
            'anger': [
                "Reconocer la emoción: Ayuda al niño a identificar y nombrar su enfado, enseñándole que es normal sentir enojo",
                "Validar los sentimientos: Hazle saber que está bien sentirse enojado, pero que las reacciones negativas pueden controlarse.",
                "Respiración y calma: Enséñales técnicas de respiración profunda o a contar hasta 10 para calmarse.",
                "Canalizar la energía: Proporciona alternativas como dibujar, jugar o salir a caminar para liberar el enojo.",
                "Enseñar a expresar emociones: Promueve el uso de palabras para expresar cómo se siente, en lugar de recurrir a la agresión.",
                "Resolución de problemas: Ayuda al niño a identificar qué lo hace enojar y cómo solucionar la situación.",
                "Reflexión posterior: Una vez calmado, habla sobre lo ocurrido para mejorar la comprensión emocional y cómo gestionar mejor situaciones similares en el futuro."
            ],
            'annoyance': [
                "Identificar la molestia: Ayuda al niño a identificar qué le molesta.",
                "Validar sus sentimientos: Asegúrate de que el niño se sienta escuchado.",
                "Promover la empatía: Explora cómo los demás pueden sentirse en la misma situación.",
                "Buscar soluciones prácticas: Anima a pensar en maneras de resolver la situación.",
                "Técnicas de relajación: Practicar respiración profunda para calmarse.",
                "Proporcionar alternativas: Ofrece actividades que distraigan o calmen.",
                "Fomentar la comunicación asertiva: Enseña a expresar la molestia sin agresión."
            ],
            'disappointment': [
                "Reconocer la emoción: Valida la decepción que siente el niño.",
                "Normalizar las expectativas: Enséñale que no siempre obtenemos lo que deseamos.",
                "Buscar lo positivo: Ayuda al niño a encontrar algo positivo en la situación.",
                "Practicar la resiliencia: Enseña cómo manejar la frustración y seguir adelante.",
                "Modelar reacciones sanas: Muestra cómo manejar la decepción con calma.",
                "Ofrecer apoyo emocional: Escuchar sin minimizar sus sentimientos.",
                "Establecer nuevas metas: Motívale a intentar de nuevo o explorar otras alternativas."            
                ],
            'disapproval': [
                "Identificar la causa: Ayuda al niño a entender por qué desaprueba algo.",
                "Validar la opinión: Reconoce su derecho a tener una opinión.",
                "Promover el respeto: Enseña cómo expresar la desaprobación respetuosamente.",
                "Explorar perspectivas: Fomenta la empatía y el entendimiento de otras opiniones.",
                "Fomentar la auto-reflexión: Invítalo a analizar si su desaprobación es justa.",
                "Canalizar la emoción: Ofrece actividades que le ayuden a pensar de forma positiva.",
                "Enseñar la negociación: Motívalo a buscar soluciones en lugar de solo criticar."
            ],
            'disgust': [
                "Explorar la reacción: Pregunta qué es lo que provoca repugnancia.",
                "Validar sus sentimientos: Asegúrate de que el niño se sienta comprendido.",
                "Promover la curiosidad: Fomenta el análisis de la situación para entenderla mejor.",
                "Fomentar el autocontrol: Enseña a manejar la repulsión sin exageraciones.",
                "Cambiar el foco de atención: Ayuda al niño a distraerse o pensar en cosas agradables.",
                "Modelar reacciones adecuadas: Demuestra cómo manejar la repugnancia sin reacciones extremas",
                "Fomentar la apertura: Explora cómo algunas cosas pueden parecer menos desagradables con el tiempo."
            ],
            'fear': [
                "Nombrar el miedo: Ayuda al niño a identificar lo que le da miedo.",
                "Crear un espacio seguro: Proporciónale un entorno donde se sienta protegido.",
                "Validar el miedo: Reconoce que su miedo es real para él.",
                "Proporcionar información: Explica la situación para reducir el miedo irracional.",
                "Enseñar técnicas de relajación: Practica la respiración profunda o visualizaciones.",
                "Enfrentar gradualmente el miedo: Introduce la situación temida en pequeños pasos.",
                "Fomentar la valentía: Elogia los esfuerzos por enfrentar el miedo."
            ],
            'grief': [
                "Validar la tristeza: Deja que el niño exprese su dolor sin presionarlo a sentirse mejor.",
                "Explicar el duelo: Ayuda a entender que el duelo es una respuesta natural.",
                "Crear un espacio de memoria: Proporciónale formas de recordar lo perdido.",
                "Fomentar la expresión emocional: Motívalo a hablar o dibujar sobre cómo se siente.",
                "Proporcionar consuelo: Sé un apoyo constante y atento.",
                "Promover la paciencia: Explica que el duelo lleva tiempo y es un proceso personal.",
                "Fomentar la esperanza: Ayuda a encontrar momentos de alegría en medio de la tristeza."
            ],
            'sadness': [
                "Identificar la causa: Pregunta qué lo está poniendo triste.",
                "Validar los sentimientos: Hazle saber que es normal sentirse triste a veces.",
                "Proporcionar apoyo emocional: Ofrece un abrazo o compañía.",
                "Realizar actividades gratificantes: Invítalo a hacer algo que disfrute.",
                "Enseñar a expresar la tristeza: Anímalo a hablar o a escribir sobre lo que siente.",
                "Promover el autocuidado: Motívalo a descansar y cuidar de sí mismo.",
                "Fomentar la esperanza: Recuérdale que la tristeza es temporal y que siempre hay cosas positivas en el futuro."
            ],
            'remorse': [
                "Entender la causa: Pregunta qué lo hace sentir remordimiento.",
                "Validar el sentimiento: Reafirma que está bien sentir remordimiento después de un error.",
                "Fomentar la auto-reflexión: Ayúdalo a pensar en lo que podría haber hecho diferente.",
                "Enseñar a pedir disculpas: Guíalo en cómo disculparse sinceramente si ha hecho daño.",
                "Reparar el daño: Motívalo a hacer algo positivo para corregir su error.",
                "Fomentar el perdón: Ayúdalo a perdonarse a sí mismo y aprender de la experiencia.",
                "Establecer nuevos compromisos: Promueve el aprendizaje para evitar repetir el error."
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
                    model="gpt-4o",
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
                    model=selected_model,
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

