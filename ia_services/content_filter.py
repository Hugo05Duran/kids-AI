import torch
from transformers import BertTokenizer, BertForSequenceClassification

class ContentFilter:
    def __init__(self, model_path='ia_services/fine_tuning/model'):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
    
    def classify_text(self, text, features):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Añadir características adicionales a las entradas
        # Suponiendo que ya se han convertido a índices/valores adecuados
        inputs['age_group'] = torch.tensor([features['age_group']], dtype=torch.int64)
        inputs['phrase_type'] = torch.tensor([features['phrase_type']], dtype=torch.int64)
        inputs['intent'] = torch.tensor([features['intent']], dtype=torch.int64)
        inputs['polarity'] = torch.tensor([features['polarity']], dtype=torch.float32)
        inputs['subjectivity'] = torch.tensor([features['subjectivity']], dtype=torch.float32)
        inputs['complexity'] = torch.tensor([features['complexity']], dtype=torch.float32)
        inputs['context'] = torch.tensor([features['context']], dtype=torch.int64)
        inputs['contains_sarcasm'] = torch.tensor([features['contains_sarcasm']], dtype=torch.int64)
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.item()
    
    def is_appropriate(self, text, features):
        return self.classify_text(text, features) == 1

# Ejemplo de uso del filtro de contenido
content_filter = ContentFilter()

text = "No quiero ir a la escuela hoy."
features = {
    'age_group': 3,  # Ejemplo: 3 corresponde a "11-12 años"
    'phrase_type': 2,  # Ejemplo: 2 corresponde a "negación"
    'intent': 4,  # Ejemplo: 4 corresponde a "negarse"
    'polarity': -0.6,
    'subjectivity': 0.6,
    'complexity': 0.6,
    'context': 1,  # Ejemplo: 1 corresponde a "Protesta contra asistir a la escuela"
    'contains_sarcasm': 0  # 0 = No, 1 = Sí
}

result = content_filter.is_appropriate(text, features)
print("Apropiado" if result else "Inapropiado")

