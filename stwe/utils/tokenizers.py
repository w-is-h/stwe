import re

def basic_tokenizer(text):
    return re.findall(r"[\w']+", text)
