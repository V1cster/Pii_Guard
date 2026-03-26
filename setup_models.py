import os
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.makedirs("local_models", exist_ok=True)

print("Downloading Text AI (BERT)...")
text_model_name = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
model_text = AutoModelForSequenceClassification.from_pretrained(text_model_name)


tokenizer.save_pretrained("./local_models/bert_pii")
model_text.save_pretrained("./local_models/bert_pii")

print("Downloading Image AI (YOLO)...")
model_img = YOLO("yolo11n.pt") 

print("Done! Check your folder for 'local_models' and 'yolo11n.pt'.")