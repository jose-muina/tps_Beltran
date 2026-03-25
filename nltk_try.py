import nltk
import tiktoken
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#nltk.download('wordnet') #por unica vez
encoder = tiktoken.get_encoding("gpt2") 
oracion = "The striped bats are hanging on their feet for best"
lemmatizer = WordNetLemmatizer()
oracion = "The striped bats are hanging on their feet for best"
tokens = encoder.encode(oracion)
word_list = [encoder.decode([token]) for token in tokens]
print(word_list)
#> ['The', 'striped', 'bats', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'best']

#Lemmatización de las palabras
for w in word_list:
    lemmatized_word = lemmatizer.lemmatize(w)
    print(lemmatized_word)