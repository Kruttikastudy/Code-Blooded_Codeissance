import sys
import pickle
import torch
from transformers import BertTokenizer, BertModel
import json

# Load pickle model
with open(r"C:\Users\ketak\OneDrive\Desktop\final_codeissance\Code-Blooded_Codeissance\backend\autonomous_depression_model.pkl", "rb") as f:
    model = pickle.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:,0,:].numpy()[0]

# Input text from Node
text = sys.argv[1]
features = preprocess_text(text)
prediction = model.predict([features])[0]
risk_score = model.predict_proba([features])[0][1]

print(json.dumps({"prediction": int(prediction), "risk_score": float(risk_score)}))
