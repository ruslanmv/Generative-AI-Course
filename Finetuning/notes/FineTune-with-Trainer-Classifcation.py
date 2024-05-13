from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
# Model and tokenizer selection
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 for positive/negative sentiment
# Load dataset (smaller subsets) - adjust these ranges as needed
train_data = load_dataset("imdb", split="train").shuffle(seed=42).select(range(1000))
test_data = load_dataset("imdb", split="test").shuffle(seed=42).select(range(100))
# Define a function to preprocess data

def preprocess_function(examples):
  return tokenizer(examples["text"], truncation=True, padding="max_length")

# Preprocess data
train_data = train_data.map(preprocess_function, batched=True)
test_data = test_data.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    save_steps=100,
    save_total_limit=2,
    num_train_epochs=2,
    evaluation_strategy="epoch"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Train the model
trainer.train()
# Evaluate the model
predictions = trainer.predict(test_data)
print(f"Test Loss: {predictions.metrics['test_loss']}")
# Save the trained model
trainer.save_model("bert-classification-trained")
tokenizer.save_pretrained("bert-classification-trained")


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Load the fine-tuned model and tokenizer (replace with your model name)
model_name = "bert-classification-trained"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# Function to preprocess text for inference
def preprocess_text(text):
    """Preprocesses a single text sample for inference."""
    encoding = tokenizer(text, truncation=True, padding="max_length", return_tensors="pt")
    return encoding
def predict(text):
    # Preprocess the sample text
    input_ids = preprocess_text(text)
    # Perform inference using the model
    with torch.no_grad():
        outputs = model(**input_ids)
        logits = outputs.logits.squeeze(0)  # Remove batch dimension if present
    # Get the predicted class (positive or negative sentiment) based on the highest logit
    predicted_class = torch.argmax(logits).item()
    predicted_label = "positive" if predicted_class == 1 else "negative"

    print(f"Predicted sentiment for '{text}': {predicted_label}")
    
    
# Sample text for inference (replace with your text)
text = "This movie was absolutely fantastic! I highly recommend it."  # Replace with the text you want to classify
predict(text)

text = "This movie was not good I dislike."  # Replace with the text you want to classify
predict(text)    