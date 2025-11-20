# text/symbols.py
symbols = ['<blank>','<pad>',' ','a','i','u','e','ə','o','ɛ','ɔ','b','c','d','f','g','h','j','k','l','m','n','ŋ','ɲ','p','q','r','s','t','v','w','x','y','z','\'']

symbol_to_id = {s:i for i,s in enumerate(symbols)}
id_to_symbol = {i:s for i,s in enumerate(symbols)}