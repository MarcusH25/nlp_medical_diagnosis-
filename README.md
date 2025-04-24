# Fine-Tuning OPT-1.3B with LoRA – Project Workflow

This README documents the workflow for fine-tuning a pre-trained OPT-1.3B language model on a medical diagnosis dataset using LoRA (Low-Rank Adaptation). The process includes preparing the data, formulating prompts, setting up the model, applying a PEFT (parameter-efficient fine-tuning) configuration with LoRA, training the model, and evaluating its performance.

## Data Preparation

We first load the labeled dataset (medical symptoms and corresponding diseases) and preprocess it for training. The raw data contains multiple symptom columns (Symptom_1, Symptom_2, ...) which we combine into a single textual description. We also remove duplicate entries and split the data into training and validation sets (e.g., 80/20 split with stratification to preserve class balance).

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset and inspect
df = pd.read_csv('dataset.csv')
# Combine symptom columns into one text field
symptom_cols = [col for col in df.columns if 'Symptom_' in col]
processed_df = pd.DataFrame()
processed_df['disease'] = df['Disease']
processed_df['symptoms'] = df[symptom_cols].apply(
    lambda x: f"Patient presents with: {'; '.join([str(s).replace('_', ' ').strip() for s in x.dropna().tolist()])}",
    axis=1
)
processed_df = processed_df.drop_duplicates(subset=['disease', 'symptoms'])

# Split into training and validation sets (e.g., 80/20 stratified)
train_df, val_df = train_test_split(
    processed_df,
    test_size=0.2,
    random_state=42,
    stratify=processed_df['disease']
)
```

After this step, we have a cleaned processed_df with two columns: symptoms (a consolidated symptom description) and disease (the diagnosis). We use train_df for fine-tuning and val_df for evaluating the model.

## Prompt Creation

Next, we define how to format each input for the language model. Given the symptoms text, we create a prompt that asks the model for the disease. This typically involves removing unnecessary phrasing and structuring the prompt clearly. For example, we might use a format like: "Given these symptoms: ... The disease is:".

```python
def build_improved_prompt(symptoms):
    """Build a direct prompt for the OPT model given a symptoms string."""
    symptoms_text = symptoms.replace("Patient presents with: ", "")
    prompt = f"""Given these symptoms: {symptoms_text}

The disease is:"""
    return prompt

# Example:
example_prompt = build_improved_prompt(train_df.iloc[0]['symptoms'])
print(example_prompt)
```

Each training sample will be converted into this prompt format. The model will be tasked with completing the prompt by generating the correct disease name.

## Model Setup (OPT-1.3B)

We use the Hugging Face Transformers library to load the pre-trained OPT-1.3B model and its tokenizer. We enable half-precision (float16) and utilize the GPU (device_map="cuda") to handle the model's size. The tokenizer is adjusted for left padding (since OPT is a causal language model expecting right-aligned context).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if model_name == "facebook/opt-1.3b":
    tokenizer.padding_side = 'left'  # OPT expects left-padded sequences
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",       # load model across available GPU(s)
    torch_dtype=torch.float16  # use FP16 precision for efficiency
)
```

At this stage, we optionally evaluate the base model on the task (baseline measurement) to see how it performs without fine-tuning. This involves feeding the validation prompts into the model and checking how often it correctly predicts the disease.

## PEFT Configuration (LoRA)

We apply Parameter-Efficient Fine-Tuning (PEFT) with LoRA to the loaded model. LoRA allows us to train low-rank adaptation matrices for a subset of the model's weights instead of full model fine-tuning. We configure LoRA by specifying the rank (r), alpha scaling factor, target modules (the parts of the model to attach LoRA weights to, such as projection layers), dropout for LoRA, and the task type (causal language modeling).

```python
from peft import LoraConfig, get_peft_model

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# Wrap the base model with LoRA adapters
print("Applying LoRA adapters to model...")
model_peft = get_peft_model(model, lora_config)
model_peft.print_trainable_parameters()
```

The print_trainable_parameters() call verifies that only a small percentage of parameters are now trainable (around 0.36% in this case, e.g. ~4.7M out of 1.3B) – illustrating the efficiency of LoRA. The rest of the model's weights remain frozen, which significantly reduces the memory and compute requirements for fine-tuning.

## Training

With the LoRA-enhanced model, we fine-tune on our domain-specific dataset. We prepare the training data for a causal language modeling objective by concatenating each prompt with its expected completion (disease name). During tokenization, we ensure that the label tokens corresponding to padding are masked out (set to -100) so they don't contribute to the loss. We use Hugging Face's Trainer API to handle the training loop.

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Prepare Dataset for training
train_inputs = [build_improved_prompt(s) for s in train_df['symptoms']]
train_outputs = list(train_df['disease'])
train_dataset = Dataset.from_dict({"input_text": train_inputs, "output_text": train_outputs})

# Tokenization function: concatenate input and output, then tokenize
def tokenize_function(examples):
    texts = [inp + out for inp, out in zip(examples["input_text"], examples["output_text"])]
    tokenized = tokenizer(texts, max_length=384, padding="max_length", truncation=True)
    # Set padding tokens to -100 in labels to ignore them in loss
    tokenized["labels"] = [
        [token if token != tokenizer.pad_token_id else -100 for token in ids]
        for ids in tokenized["input_ids"]
    ]
    return tokenized

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./medical-diagnosis-lora-adapter",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=30,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    save_strategy="epoch",
    save_total_limit=3,
    fp16=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    remove_unused_columns=False,
    logging_steps=10,
    report_to="tensorboard"
)

# Initialize Trainer with our LoRA model
trainer = Trainer(
    model=model_peft,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
)
# Start fine-tuning
trainer.train()
```

During training, only the LoRA adapter weights are updated. We train for the specified number of epochs (30 in this case), which is feasible given the low number of trainable parameters. After training, we save the fine-tuned adapter and tokenizer for later use:

```python
model_peft.save_pretrained("./medical-diagnosis-lora-adapter")
tokenizer.save_pretrained("./medical-diagnosis-lora-adapter")
```

This saves the LoRA adapter weights (not the full model weights) to the medical-diagnosis-lora-adapter directory, allowing us to reload and use the fine-tuned model in the future.

## Evaluation

Finally, we evaluate the fine-tuned model on the held-out validation set to measure improvement over the baseline. We generate predictions for each validation prompt using the fine-tuned model (with the LoRA adapter applied) and then compare these predictions to the true labels.

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Create prompts from validation symptoms
improved_prompts = [build_improved_prompt(sym) for sym in val_df['symptoms']]
# Generate predictions using the fine-tuned model
predictions = []  
for prompt in improved_prompts:
    # Tokenize and generate with the fine-tuned model (using a short generation length for disease name)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model_peft.generate(**inputs, max_new_tokens=20)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract the disease name from the model's output
    pred_disease = extract_disease_prediction(output_text, prompt)
    predictions.append(pred_disease)

# Compute evaluation metrics
acc = accuracy_score(val_df['disease'], predictions)
f1 = f1_score(val_df['disease'], predictions, average="macro")
print(f"Fine-tuned Model Accuracy: {acc:.4f}")
print(f"Fine-tuned Model Macro F1 Score: {f1:.4f}")
print(classification_report(val_df['disease'], predictions))
```

In the code above, we reuse the prompt-building function and a prediction extraction helper (extract_disease_prediction) to get the model's predicted disease from the generated text. We then calculate accuracy and the macro F1-score, and display a full classification report.
