import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric, Dataset
import json

# Cargar y categorizar el dataset
def load_and_categorize_data(path):
    obscene_words = ["malo", "feo", "tonto"]  # Puedes agregar más palabras a esta lista
    negative_context_keywords = ["triste", "ansiedad", "acoso", "maltrato"]
    
    def categorize_entry(entry):
        text = entry['text']
        is_obscene = any(word in text for word in obscene_words)
        
        sentiment = entry.get('sentiment', {})
        polarity = sentiment.get('polarity', 0)
        is_emotionally_negative = entry['is_negative'] or polarity < 0 or entry['is_subtle_negative']
        
        context = entry.get('context', "")
        if any(word in context for word in negative_context_keywords):
            is_emotionally_negative = True
        
        is_appropriate = not is_obscene and not is_emotionally_negative
        
        entry['is_obscene'] = is_obscene
        entry['is_emotionally_negative'] = is_emotionally_negative
        entry['is_appropriate'] = is_appropriate
        
        return entry
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for entry in data:
        categorize_entry(entry)
    
    return data

# Cargar y categorizar el dataset
dataset_path = 'ia_services/fine_tuning/dataset/text_dataset.json'
data = load_and_categorize_data(dataset_path)

# Convertir datos a un dataset de Hugging Face
dataset = Dataset.from_list(data)

# Dividir el dataset en entrenamiento y evaluación
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Tokenizar los datos
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Preparar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Métrica de evaluación
metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# Crear el modelo y el entrenador
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Asumiendo que hay tres etiquetas

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Entrenamiento del modelo
trainer.train()

# Guardar el modelo fine-tuneado
model.save_pretrained('ia_services/fine_tuning/model')
tokenizer.save_pretrained('ia_services/fine_tuning/model')
