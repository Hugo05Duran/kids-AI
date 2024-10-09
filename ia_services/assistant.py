import os
import django
import openai
import time
from django.conf import settings
from transformers import pipeline
from ia_services.content_filter import ContentFilter  
from model_selection import ComplexityClassifier
import soundfile as sf
import vosk
import coqui_tts
from coqui_tts.tts import TTS
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.model_selection = ComplexityClassifier()

       
        self.vosk_model = vosk.Model("model")  
        
       
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

        self.model_map = {
            "simple": "DistilGPT2",  
            "moderate": "T5-base",   
            "complex": "gpt-4o"      
        }
  
        self.emotion_model_map = {
            "simple": "DistilRoBERTa",  
            "complex": "RoBERTa"        
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

    def detect_topic(self, text, complexity):
        if complexity == "simple":
           
            return "simple_topic_detected"
        elif complexity == "moderate":
            
            response = openai.Completion.create(
                engine="T5-base",
                prompt=f"Identifica el tema principal de este texto: {text}",
                max_tokens=10
            )
            topic = response.choices[0].text.strip()
            return topic
        else:
          
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

    def get_emotion_model(self, complexity):
        
        return self.emotion_model_map.get(complexity, "RoBERTa")           

    def get_model_for_complexity(self, complexity):
        
        return self.model_map.get(complexity, "gpt-4o")  

     def get_response(self, user_id, message, age):
        classification = self.classify_and_track_sentiment(user_id, message)
        complexity, _ = self.model_selection.calculate_complexity(message)
        selected_model = self.get_model_for_complexity(complexity)

        negative_emotions = {
            'anger': [
                "Evitar reprimir las emociones",
                "Reconocer la emoción: Ayuda al niño a identificar y nombrar su enfado, enseñándole que es normal sentir ira",
                "Validar los sentimientos: Hazle saber que está bien sentirse enojado, pero que las reacciones negativas pueden controlarse.",
                "Respiración y calma: Enséñales técnicas de respiración profunda o a contar hasta 10 para calmarse.",
                "Canalizar la energía: Proporciona alternativas como dibujar, jugar o salir a caminar para liberar el enojo.",
                "Enseñar a expresar emociones: Promueve el uso de palabras para expresar cómo se siente, en lugar de recurrir a la agresión.",
                "Resolución de problemas: Ayuda al niño a identificar qué lo hace enojar y cómo solucionar la situación.",
                "Reflexión posterior: Una vez calmado, habla sobre lo ocurrido para mejorar la comprensión emocional y cómo gestionar mejor situaciones similares en el futuro.",
                "Las emociones no son intrínsecamente buenas o malas, lo importante es cómo las gestionamos"
            ],
            'annoyance': [
                "Evitar reprimir las emociones",
                "Identificar la molestia: Ayuda al niño a identificar qué le molesta.",
                "Validar sus sentimientos: Asegúrate de que el niño se sienta escuchado.",
                "Promover la empatía: Explora cómo los demás pueden sentirse en la misma situación.",
                "Buscar soluciones prácticas: Anima a pensar en maneras de resolver la situación.",
                "Técnicas de relajación: Practicar respiración profunda para calmarse.",
                "Proporcionar alternativas: Ofrece actividades que distraigan o calmen.",
                "Fomentar la comunicación asertiva: Enseña a expresar la molestia sin agresión.",
                "Las emociones no son intrínsecamente buenas o malas, lo importante es cómo las gestionamos"
            ],
            'disappointment': [
                "Evitar reprimir las emociones",
                "Reconocer la emoción: Valida la decepción que siente el niño.",
                "Normalizar las expectativas: Enséñale que no siempre obtenemos lo que deseamos.",
                "Buscar lo positivo: Ayuda al niño a encontrar algo positivo en la situación.",
                "Practicar la resiliencia: Enseña cómo manejar la frustración y seguir adelante.",
                "Modelar reacciones sanas: Muestra cómo manejar la decepción con calma.",
                "Ofrecer apoyo emocional: Escuchar sin minimizar sus sentimientos.",
                "Establecer nuevas metas: Motívale a intentar de nuevo o explorar otras alternativas.",
                "Las emociones no son intrínsecamente buenas o malas, lo importante es cómo las gestionamos"            
                ],
            'disapproval': [
                "Evitar reprimir las emociones",
                "Identificar la causa: Ayuda al niño a entender por qué desaprueba algo.",
                "Validar la opinión: Reconoce su derecho a tener una opinión.",
                "Promover el respeto: Enseña cómo expresar la desaprobación respetuosamente.",
                "Explorar perspectivas: Fomenta la empatía y el entendimiento de otras opiniones.",
                "Fomentar la auto-reflexión: Invítalo a analizar si su desaprobación es justa.",
                "Canalizar la emoción: Ofrece actividades que le ayuden a pensar de forma positiva.",
                "Enseñar la negociación: Motívalo a buscar soluciones en lugar de solo criticar.",
                "Las emociones no son intrínsecamente buenas o malas, lo importante es cómo las gestionamos"
            ],
            'disgust': [
                "Evitar reprimir las emociones",
                "Explorar la reacción: Pregunta qué es lo que provoca repugnancia.",
                "Validar sus sentimientos: Asegúrate de que el niño se sienta comprendido.",
                "Promover la curiosidad: Fomenta el análisis de la situación para entenderla mejor.",
                "Fomentar el autocontrol: Enseña a manejar la repulsión sin exageraciones.",
                "Cambiar el foco de atención: Ayuda al niño a distraerse o pensar en cosas agradables.",
                "Modelar reacciones adecuadas: Demuestra cómo manejar la repugnancia sin reacciones extremas",
                "Fomentar la apertura: Explora cómo algunas cosas pueden parecer menos desagradables con el tiempo.",
                "Las emociones no son intrínsecamente buenas o malas, lo importante es cómo las gestionamos"
            ],
            'fear': [
                "Evitar reprimir las emociones",
                "Nombrar el miedo: Ayuda al niño a identificar lo que le da miedo.",
                "Crear un espacio seguro: Proporciónale un entorno donde se sienta protegido.",
                "Validar el miedo: Reconoce que su miedo es real para él.",
                "Proporcionar información: Explica la situación para reducir el miedo irracional.",
                "Enseñar técnicas de relajación: Practica la respiración profunda o visualizaciones.",
                "Enfrentar gradualmente el miedo: Introduce la situación temida en pequeños pasos.",
                "Fomentar la valentía: Elogia los esfuerzos por enfrentar el miedo.",
                "Las emociones no son intrínsecamente buenas o malas, lo importante es cómo las gestionamos"
            ],
            'grief': [
                "Evitar reprimir las emociones",
                "Validar la tristeza: Deja que el niño exprese su dolor sin presionarlo a sentirse mejor.",
                "Explicar el duelo: Ayuda a entender que el duelo es una respuesta natural.",
                "Crear un espacio de memoria: Proporciónale formas de recordar lo perdido.",
                "Fomentar la expresión emocional: Motívalo a hablar o dibujar sobre cómo se siente.",
                "Proporcionar consuelo: Sé un apoyo constante y atento.",
                "Promover la paciencia: Explica que el duelo lleva tiempo y es un proceso personal.",
                "Fomentar la esperanza: Ayuda a encontrar momentos de alegría en medio de la tristeza.",
                "Las emociones no son intrínsecamente buenas o malas, lo importante es cómo las gestionamos"
            ],
            'sadness': [
                "Evitar reprimir las emociones",
                "Identificar la causa: Pregunta qué lo está poniendo triste.",
                "Validar los sentimientos: Hazle saber que es normal sentirse triste a veces.",
                "Proporcionar apoyo emocional: Ofrece un abrazo o compañía.",
                "Realizar actividades gratificantes: Invítalo a hacer algo que disfrute.",
                "Enseñar a expresar la tristeza: Anímalo a hablar o a escribir sobre lo que siente.",
                "Promover el autocuidado: Motívalo a descansar y cuidar de sí mismo.",
                "Fomentar la esperanza: Recuérdale que la tristeza es temporal y que siempre hay cosas positivas en el futuro.",
                "Las emociones no son intrínsecamente buenas o malas, lo importante es cómo las gestionamos"
            ],
            'remorse': [
                "Evitar reprimir las emociones",
                "Entender la causa: Pregunta qué lo hace sentir remordimiento.",
                "Validar el sentimiento: Reafirma que está bien sentir remordimiento después de un error.",
                "Fomentar la auto-reflexión: Ayúdalo a pensar en lo que podría haber hecho diferente.",
                "Enseñar a pedir disculpas: Guíalo en cómo disculparse sinceramente si ha hecho daño.",
                "Reparar el daño: Motívalo a hacer algo positivo para corregir su error.",
                "Fomentar el perdón: Ayúdalo a perdonarse a sí mismo y aprender de la experiencia.",
                "Establecer nuevos compromisos: Promueve el aprendizaje para evitar repetir el error.",
                "Las emociones no son intrínsecamente buenas o malas, lo importante es cómo las gestionamos"
            ]
        }

        emotion_model = self.get_emotion_model(complexity)

        predominant_emotion = max(classification['specific_classification'], key=classification['specific_classification'].get)

        if classification['specific_classification'][predominant_emotion] > 0.15:
            instructions = negative_emotions.get(predominant_emotion, [])
            
            prompt = f"El niño está experimentando {predominant_emotion}. " + " ".join(instructions) + f" Responde como si hablaras con un niño de {age} años, asegurándote de ser educativo y apropiado."
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",  
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": message}
                    ]
                )
                return response.choices[0].message['content']
            except Exception as e:
                print(f"Error al obtener respuesta de AI: {e}")
                return "Lo siento, no pude entender eso. ¿Podrías intentar preguntar de otra manera?"

        else:
           
            age_prompt = self.get_age_appropriate_prompt(age)
            topic = self.detect_topic(message, complexity)
            try:
                response = openai.ChatCompletion.create(
                    model=selected_model, 
                    messages=[
                        {"role": "system", "content": f"{age_prompt} Asegúrate de que tu respuesta sea educativa, apropiada para niños y fácil de entender. Tema: {topic}."},
                        {"role": "user", "content": message}
                    ]
                )
                return response.choices[0].message['content']
            except Exception as e:
                print(f"Error al obtener respuesta de AI: {e}")
                return "Lo siento, no pude entender eso. ¿Podrías intentar preguntar de otra manera?"

    
    def create_story_with_child(self, user_id, age):
    story_elements = {}
    prompts = [
        "¿Cómo se llama tu personaje principal?",
        "¿Es tu personaje un niño, una niña, un animal u otra criatura?",
        "¿Dónde ocurre la historia? Por ejemplo: en un bosque, en una ciudad, en el espacio.",
        "¿Qué le gusta hacer a tu personaje?",
        "¿Qué problema tiene que resolver en la historia?",
        "¿Cómo termina la historia? ¿Qué aprende tu personaje?"
    ]

    for prompt in prompts:
        # Sintetizar y reproducir el prompt
        self.synthesize_speech(prompt, output_file="prompt.wav")
        # Reproducir el audio al niño (implementa según tu plataforma)
        # Capturar la respuesta del niño
        audio_response = self.capture_audio()
        child_response = self.transcribe_audio(audio_response)
        # Validar la respuesta
        if not child_response:
            child_response = "Respuesta no proporcionada"
        # Guardar la respuesta
        key = prompt[:prompt.find('?')].strip().lower().replace('¿', '').replace(',', '').replace(' ', '_')
        story_elements[key] = child_response

    # Generar y presentar el cuento
    story = self.generate_story(story_elements, age)
    self.present_story(story)
    return story

    def initialize_llama_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")

    def generate_story(self, story_elements, age):
        age_prompt = self.get_age_appropriate_prompt(age)
        story_prompt = f"""{age_prompt}
    Utiliza los siguientes elementos para crear un cuento corto y divertido para un niño de {age} años:
    - Personaje principal: {story_elements.get('cómo_se_llama_tu_personaje_principal', 'Un personaje')}
    - Tipo de personaje: {story_elements.get('es_tu_personaje_un_niño_una_niña_un_animal_u_otra_criatura', 'una criatura mágica')}
    - Lugar: {story_elements.get('dónde_ocurre_la_historia', 'un lugar mágico')}
    - Actividades favoritas: {story_elements.get('qué_le_gusta_hacer_a_tu_personaje', 'explorar y jugar')}
    - Problema a resolver: {story_elements.get('qué_problema_tiene_que_resolver_en_la_historia', 'un desafío emocionante')}
    - Final y aprendizaje: {story_elements.get('cómo_termina_la_historia_qué_aprende_tu_personaje', 'una valiosa lección')}
    Asegúrate de que el cuento sea **apropiado para niños**, **positivo** y **educativo**. Evita cualquier contenido que incluya violencia, lenguaje inapropiado o temas sensibles.
    """

   try:
        inputs = self.tokenizer.encode(story_prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=500, temperature=0.5)
        story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Aplicar la moderación usando la API de moderación de OpenAI
        moderation_response = openai.Moderation.create(
            input=story
        )
        moderation_result = moderation_response["results"][0]

        if moderation_result["flagged"]:
            # Si el contenido es inapropiado, informar al usuario y posiblemente regenerar el cuento
            return "Lo siento, el cuento generado no es apropiado. Vamos a intentar crear otro cuento juntos."
        else:
            return story

    except Exception as e:
        print(f"Error al generar o moderar el cuento: {e}")
        return "Lo siento, no pude generar el cuento en este momento."

    def present_story(self, story, output_file="story_output.wav"):
        # Sintetizar el cuento en audio
        self.synthesize_speech(story, output_file=output_file)
        # Reproducir el audio al niño (implementa según tu plataforma)
        # Aquí puedes añadir el código para reproducir el archivo de audio

    
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

  
    def transcribe_audio(self, audio_file):
        with sf.SoundFile(audio_file) as audio:
            recognizer = vosk.KaldiRecognizer(self.vosk_model, audio.samplerate)
            while True:
                data = audio.read(4000, dtype="int16")
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    return result['text']
        return None

   
    def synthesize_speech(self, text, output_file="output.wav"):
        self.tts.tts_to_file(text=text, file_path=output_file)
        return output_file

    
    def process_audio_input(self, user_id, audio_file, age):
        transcript = self.transcribe_audio(audio_file)
        classification = self.classify_and_track_sentiment(user_id, transcript)
        
        
        negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'fear', 'grief', 'sadness', 'remorse']
        emotion_instructions = None
        for emotion in negative_emotions:
            if classification['specific_classification'][emotion] > 0.15:
                emotion_instructions = self.get_instructions_for_emotion(emotion)
                break
        
        response_text = self.get_response(user_id, transcript, age, emotion_instructions)
        audio_response = self.synthesize_speech(response_text)
        return audio_response


ai_assistant = AIAssistant()
