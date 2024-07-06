from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer
import json
app = Flask(__name__)

# Define the file paths and model loading
model_path = "contrastive_entailment_model_best.pth"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Load the value descriptions from a JSON file
with open('value-categories.json', 'r') as file:
    value_descriptions = json.load(file)

# Load RoBERTa tokenizer
model_name = 'pepa/roberta-base-snli'
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_values', methods=['POST'])
def detect_values():
    premise = request.form['premise']
    stance = request.form['stance']
    conclusion = request.form['conclusion']

    detected_values = []

    # Add premise, stance, and conclusion to detected values
    detected_values.append(f"Premise: {premise}")
    detected_values.append(f"Stance: {stance}")
    detected_values.append(f"Conclusion: {conclusion}")

    # Combine premise, conclusion, and stance into a single argument
    argument = f"{stance} {conclusion} by saying {premise}"
    all_predictions = []

    # Iterate through each value description
    for value_category, value_info in value_descriptions.items():
        # Get the value description
        value_description = value_info['personal-motivation']
        
        # Combine premise with the current value description
        premise_value_combined = f"{stance} {conclusion} by saying {premise}"
        
        # Tokenize the combined premise and value description
        inputs = tokenizer(
            premise_value_combined,
            value_description,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Prepare input tensors
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Model inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Calculate probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        # Append the predictions for the current value
        all_predictions.append((value_category, predictions.item()))

    # Add detected values to the list
    for value_category, prediction in all_predictions:
        if prediction == 0:
            print(value_category)
            detected_values.append(f"Value: {value_category} is present")
        else:
            detected_values.append(f"Value: {value_category} is not present")
   
    return jsonify(values=detected_values)

if __name__ == '__main__':
    app.run(debug=True)
