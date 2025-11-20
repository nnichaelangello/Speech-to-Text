# text/cleaners.py
import re

def indonesian_cleaners(text):
    text = text.lower()
    text = re.sub(r'[^ a-z\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()