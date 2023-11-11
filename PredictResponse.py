import random
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

from data_preprocessing import preprocess_train_data
from data_preprocessing import get_stem_words
model = tensorflow.keras.load_model('C:/Users/VihanPC/Downloads/PRO-C120-Student-Boilerplate-Code-main/chatbot_model.h5')

ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('C:/Users/VihanPC/Downloads/PRO-C120-Student-Boilerplate-Code-main/PRO-C120-Student-Boilerplate-Code-main/intents.json').read()
intents = json.loads(train_data_file)

words = pickle.load(open('C:/Users/VihanPC/Downloads/PRO-C120-Student-Boilerplate-Code-main/words.pkl','rb'))
classes = pickle.load(open('C:/Users/VihanPC/Downloads/PRO-C120-Student-Boilerplate-Code-main/classes.pkl','rb'))

def preprocess_user_input(user_input):
    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1,ignore_words)
    input_word_token_2 = sorted(list(set(input_word_token_2)))
    bag = []
    bag_of_words = []
    for word in words:
        if word in input_word_token_2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label

def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    predicted_class = classes[predicted_class_label]
    for intent in intents["intents"]:
        if intent['tag'] == predicted_class:
             bot_response = random.choice(intent['responses'])
             return bot_response
while True:
    user_input = input("Type Your Message Here")
    print(user_input)
    response = bot_response(user_input)
    print(response)