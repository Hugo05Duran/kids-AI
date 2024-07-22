import torch
from transformers import BertTokenizer, BertForSequenceClassification

class ContentFilterPipeline:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
    
    def classify_text(self, text, features):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Añadir las características adicionales al diccionario de inputs
        inputs['age_group'] = torch.tensor([features['age_group']])
        inputs['phrase_type'] = torch.tensor([features['phrase_type']])
        inputs['intent'] = torch.tensor([features['intent']])
        inputs['polarity'] = torch.tensor([features['polarity']])
        inputs['subjectivity'] = torch.tensor([features['subjectivity']])
        inputs['complexity'] = torch.tensor([features['complexity']])
        inputs['context'] = torch.tensor([features['context']])
        inputs['contains_sarcasm'] = torch.tensor([features['contains_sarcasm']])
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.item()

# Uso del pipeline para verificar el contenido
pipeline = ContentFilterPipeline('ia_services/fine_tuning/model')

text = "Ejemplo de texto para clasificar"
features = {
    'age_group': 2,  # ejemplo de valor, convertir a tensor numérico adecuado
    'phrase_type': 1,  # ejemplo de valor, convertir a tensor numérico adecuado
    'intent': 3,  # ejemplo de valor, convertir a tensor numérico adecuado
    'polarity': -0.2,  # ejemplo de
    'subjectivity': 0.5,
    'complexity': 0.5,
    'context': 1,  # Ejemplo: 1 corresponde a "Charla en clase"
    'contains_sarcasm': 0  # 0 = No, 1 = Sí
}

result = pipeline.classify_text(text, features)
print("Apropiado" if result == 1 else "Inapropiado")
