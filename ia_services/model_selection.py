import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class ComplexityClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Modo evaluación para desactivar dropout, etc.

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Tomamos la representación del [CLS] token
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding

    def calculate_complexity(self, text):
        embedding = self.get_embedding(text)

        # Definimos embeddings de referencia para cada nivel de complejidad
        simple_reference = self.get_embedding("This is a very basic sentence.")
        moderate_reference = self.get_embedding("The implementation of basic algorithms is essential.")
        complex_reference = self.get_embedding("Advanced statistical models involve complex mathematical constructs.")

        # Calculamos la distancia de coseno para evaluar similitudes
        simple_score = np.dot(embedding, simple_reference.T) / (np.linalg.norm(embedding) * np.linalg.norm(simple_reference))
        moderate_score = np.dot(embedding, moderate_reference.T) / (np.linalg.norm(embedding) * np.linalg.norm(moderate_reference))
        complex_score = np.dot(embedding, complex_reference.T) / (np.linalg.norm(embedding) * np.linalg.norm(complex_reference))

        # Clasificación basada en la mayor similitud
        scores = {"simple": simple_score, "moderate": moderate_score, "complex": complex_score}
        complexity = max(scores, key=scores.get)
        return complexity, scores



# Ejemplo de uso
classifier = ComplexityClassifier()
input_text = "Implementing a convolutional neural network requires a deep understanding of optimization techniques."
complexity, scores = classifier.calculate_complexity(input_text)
print(f"Complexity: {complexity}, Scores: {scores}")
