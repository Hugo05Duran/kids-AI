import torch
from transformers import BertTokenizer, BertForSequenceClassification

class ContentFilter:
    def __init__(self, model_path='ia_services/fine_tuning/model'):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
    
    def classify_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.item()
    
    def categorize_text(self, text):
        category = self.classify_text(text)
        if category == 0:
            return "is_obscene"
        elif category == 1:
            return "is_emotionally_negative"
        else:
            return "is_appropriate"
    
    def is_obscene(self, text):
        return self.categorize_text(text) == "is_obscene"
    
    def is_emotionally_negative(self, text):
        return self.categorize_text(text) == "is_emotionally_negative"
    
    def is_appropriate(self, text):
        return self.categorize_text(text) == "is_appropriate"

# Uso del pipeline para verificar el contenido
content_filter = ContentFilter()

text = "Ejemplo de texto para clasificar"
result = content_filter.categorize_text(text)
print(result)

