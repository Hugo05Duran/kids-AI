import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import json

# Cargar el dataset
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Preparar el dataset para el entrenamiento
def prepare_dataset(data):
    texts = [item['text'] for item in data]
    is_negatives = [item['is_negative'] for item in data]
    age_groups = [item['age_group'] for item in data]
    phrase_types = [item['phrase_type'] for item in data]
    intents = [item['intent'] for item in data]
    polarities = [item['sentiment']['polarity'] for item in data]
    subjectivities = [item['sentiment']['subjectivity'] for item in data]
    complexities = [item['complexity'] for item in data]
    contexts = [item['context'] for item in data]
    contains_sarcasm = [item['contains_sarcasm'] for item in data]

    return Dataset.from_dict({
        'text': texts,
        'is_negative': is_negatives,
        'age_group': age_groups,
        'phrase_type': phrase_types,
        'intent': intents,
        'polarity': polarities,
        'subjectivity': subjectivities,
        'complexity': complexities,
        'context': contexts,
        'contains_sarcasm': contains_sarcasm
    })

# Cargar y preparar el dataset
dataset_path = 'ia_services/fine_tuning/dataset/dataset.json'
data = load_data(dataset_path)
dataset = prepare_dataset(data)

# Tokenizer y modelo preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenización de los textos y procesamiento de características adicionales
def tokenize_and_process(examples):
    tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True)
    tokenized_inputs['age_group'] = examples['age_group']
    tokenized_inputs['phrase_type'] = examples['phrase_type']
    tokenized_inputs['intent'] = examples['intent']
    tokenized_inputs['polarity'] = examples['polarity']
    tokenized_inputs['subjectivity'] = examples['subjectivity']
    tokenized_inputs['complexity'] = examples['complexity']
    tokenized_inputs['context'] = examples['context']
    tokenized_inputs['contains_sarcasm'] = examples['contains_sarcasm']
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_process, batched=True)

# Preparar datos para el Trainer
train_dataset = tokenized_datasets
# No se puede dividir en train y test si no tenemos un dataset suficientemente grande.

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

# Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # Usar el mismo dataset para evaluación por falta de datos
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Entrenamiento del modelo
trainer.train()

# Guardar el modelo fine-tuneado
model.save_pretrained('ia_services/fine_tuning/model')
tokenizer.save_pretrained('ia_services/fine_tuning/model')
