# text/tokenizer.py
from text.symbols import symbol_to_id

def text_to_labels(text):
    text = indonesian_cleaners(text)
    return [symbol_to_id.get(c, symbol_to_id[' ']) for c in text]