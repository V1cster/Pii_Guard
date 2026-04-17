#Open cmd and locate the folder once your in the folder use the following commands 

python -m venv venv

venv\Scripts\activate 

pip install streamlit ultralytics transformers pillow-heif pillow piexif (you only need to download the tools once so this is only for first time runs)

#finally run streamlit

streamlit run app.py 

#if you want to host the prototype from your pc onto your phone locally instead run this code, you'll be able to join by typing your http://"your ip address":8501

streamlit run app.py --server.address 0.0.0.0

