import random
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from models import Text, ActivityLog  # Asumiendo que tienes modelos Django definidos

class ComprehensionActivity:
    def __init__(self):
        # Modelo y tokenizador para generación de preguntas y respuestas
        self.model_name = 'bigscience/bloom-3b'  # Reemplaza con el modelo que estás utilizando
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Modelo para cálculo de similitud semántica
        self.similarity_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def select_text(self, age, interests):
        try:
            # Intentar obtener un texto de la base de datos
            suitable_texts = Text.objects.filter(age_range=age, topic__in=interests)
            if suitable_texts.exists():
                selected_text = random.choice(suitable_texts).content
            else:
                # Generar texto personalizado
                story_elements = self.get_story_elements_from_interests(interests)
                selected_text = self.generate_story(story_elements, age)
            return selected_text
        except Exception as e:
            print(f"Error al acceder a la base de datos: {e}")
            # Como último recurso, generar un texto genérico
            selected_text = self.generate_generic_story(age)
            return selected_text

    def get_story_elements_from_interests(self, interests):
        # Implementar lógica para crear elementos del cuento basados en los intereses
        story_elements = {
            'personaje_principal': 'Alex',
            'tipo_de_personaje': 'niño',
            'lugar': 'un bosque mágico',
            'actividades_favoritas': 'explorar y aprender',
            'problema_a_resolver': 'encontrar un tesoro perdido',
            'final_y_aprendizaje': 'aprende el valor de la amistad'
        }
        # Personalizar según intereses
        if 'animales' in interests:
            story_elements['tipo_de_personaje'] = 'un animal'
        if 'aventuras' in interests:
            story_elements['problema_a_resolver'] = 'una emocionante aventura'
        # Añadir más personalizaciones según sea necesario
        return story_elements

    def generate_story(self, story_elements, age):
        prompt = f"""
Eres un escritor de cuentos para niños de {age} años. Crea un cuento corto y divertido utilizando los siguientes elementos:
- Personaje principal: {story_elements.get('personaje_principal', 'Un personaje')}
- Tipo de personaje: {story_elements.get('tipo_de_personaje', 'una criatura mágica')}
- Lugar: {story_elements.get('lugar', 'un lugar mágico')}
- Actividades favoritas: {story_elements.get('actividades_favoritas', 'explorar y jugar')}
- Problema a resolver: {story_elements.get('problema_a_resolver', 'un desafío emocionante')}
- Final y aprendizaje: {story_elements.get('final_y_aprendizaje', 'una valiosa lección')}
Asegúrate de que el cuento sea apropiado para niños, positivo y educativo.
"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=500, temperature=0.7)
        story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return story

    def generate_generic_story(self, age):
        # Generar un cuento genérico si no se puede obtener de otra manera
        story_elements = {
            'personaje_principal': 'Sam',
            'tipo_de_personaje': 'niño',
            'lugar': 'una ciudad amigable',
            'actividades_favoritas': 'ayudar a los demás',
            'problema_a_resolver': 'un misterio en el vecindario',
            'final_y_aprendizaje': 'la importancia de la cooperación'
        }
        return self.generate_story(story_elements, age)

    def generate_comprehension_questions(self, text, age):
        prompt = f"""Eres un profesor que crea preguntas de comprensión lectora para niños de {age} años. Lee el siguiente texto y genera cinco preguntas claras y sencillas para evaluar la comprensión del niño:

Texto:
{text}

Por favor, proporciona solo las preguntas enumeradas del 1 al 5.
"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=500, temperature=0.7)
        questions_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        questions = self.extract_questions(questions_text)
        return questions

    def extract_questions(self, questions_text):
        # Dividir el texto en líneas y filtrar las preguntas
        lines = questions_text.strip().split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                question = line[line.find('.')+1:].strip() if '.' in line else line[1:].strip()
                questions.append(question)
        return questions

    def evaluate_response(self, question, child_response, text):
        prompt = f"""A continuación se presenta un texto y una pregunta. Proporciona la respuesta correcta basada en el texto.

Texto:
{text}

Pregunta:
{question}

Respuesta:
"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=100, temperature=0.5)
        expected_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Calcular la similitud entre la respuesta del niño y la respuesta esperada
        similarity = self.calculate_similarity(child_response, expected_answer)
        is_correct = similarity > 0.7  # Umbral de similitud
        return is_correct, expected_answer

    def calculate_similarity(self, response1, response2):
        embeddings = self.similarity_model.encode([response1, response2], convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cosine_score.item()

    def record_activity(self, user_id, text, question, child_response, is_correct):
        # Registrar la actividad en la base de datos
        ActivityLog.objects.create(
            user_id=user_id,
            text=text,
            question=question,
            child_response=child_response,
            is_correct=is_correct
        )

    # Métodos para síntesis y reconocimiento de voz (placeholders)
    def synthesize_speech(self, text):
        # Implementar la síntesis de voz
        pass

    def capture_audio(self):
        # Implementar la captura de audio
        pass

    def transcribe_audio(self, audio):
        # Implementar la transcripción de audio a texto
        return "Respuesta del niño"

    def present_text(self, text):
        # Presentar el texto al niño, ya sea en pantalla o mediante síntesis de voz
        self.synthesize_speech("Ahora vamos a leer un texto. Escucha con atención.")
        self.synthesize_speech(text)

    def comprehension_activity(self, user_id, age, interests):
        # Seleccionar o generar el texto adecuado
        text = self.select_text(age, interests)

        # Presentar el texto al niño
        self.present_text(text)

        # Generar preguntas de comprensión
        questions = self.generate_comprehension_questions(text, age)

        for question in questions:
            # Presentar la pregunta
            self.synthesize_speech(question)
            # Capturar la respuesta del niño
            audio_response = self.capture_audio()
            child_response = self.transcribe_audio(audio_response)
            # Evaluar la respuesta
            is_correct, expected_answer = self.evaluate_response(question, child_response, text)
            # Proporcionar feedback
            self.provide_feedback(is_correct, expected_answer, age)
            # Registrar la actividad
            self.record_activity(user_id, text, question, child_response, is_correct)

    def provide_feedback(self, is_correct, expected_answer, age):
        if is_correct:
            feedback = "¡Muy bien! Has respondido correctamente."
        else:
            feedback = f"No es correcto. La respuesta correcta es: {expected_answer}."
        # Adaptar el lenguaje al nivel de edad si es necesario
        self.synthesize_speech(feedback)
