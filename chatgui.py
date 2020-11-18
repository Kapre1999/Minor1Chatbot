import requests
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


senForTheMatching = ['covid cases in Maharashtra', 'covid cases in Andhra Pradesh', 'covid cases in Karnataka', 'covid cases in Tamil Nadu', 'covid cases in Uttar Pradesh', 'covid cases in Kerala', 'covid cases in Delhi', 'covid cases in West Bengal', 'covid cases in Odisha', 'covid cases in Telangana', 'covid cases in Bihar', 'covid cases in Assam', 'covid cases in Rajasthan', 'covid cases in Chhattisgarh', 'covid cases in Gujarat', 'covid cases in Madhya Pradesh', 'covid cases in Haryana', 'covid cases in Punjab', 'covid cases in Jharkhand', 'covid cases in Jammu and Kashmir', 'covid cases in Uttarakhand', 'covid cases in Goa', 'covid cases in Puducherry', 'covid cases in Tripura', 'covid cases in Himachal Pradesh', 'covid cases in Manipur', 'covid cases in Arunachal Pradesh', 'covid cases in Chandigarh', 'covid cases in Meghalaya', 'covid cases in Nagaland', 'covid cases in Ladakh', 'covid cases in Andaman and Nicobar Islands', 'covid cases in Sikkim', 'covid cases in Mizoram', 'covid cases in Daman and Diu', 'covid cases in Dadra and Nagar Haveli', 'covid cases in Lakshadweep']

# def getCovidCases(state):
#     response = requests.get("https://api.covidindiatracker.com/state_data.json").json()
#     data = response

#     cases = {}

#     for res in data:
#         state_code = res['id']
#         state_name = res['state']
#         active = res['active']
#         cases[state_code] = state_name , active


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def getCovidDataFromState(state):
    response = requests.get("https://api.covidindiatracker.com/state_data.json").json()
    data = response

    states = {}

    cases = {}

    for res in data:
        state_code = res['id']
        state_name = res['state']
        active = res['active']
        cases[state_code] = state_name , active
        states[state_name] = state_code

    return cases[state]

def getState(msg):
    msg = msg.split(" ")
    if len(msg) == 5:
        msg = msg[3]+" "+msg[4]
    elif len(msg) == 7:
        msg = msg[3]+" "+msg[4]+" "+msg[5]+" "+msg[6]
    else:
        msg = msg[-1]
    with open('/home/bhavesh/MY/b/python-project-chatbot/stateList.json') as jsonfile:
        data = json.load(jsonfile)
        if msg in data.keys():
            return(data[msg])

#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)


    if msg in senForTheMatching:
        stateR = getState(msg)
        casesOf = getCovidDataFromState(stateR)
        print(casesOf)
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END,"YOU: " +msg + '\n\n')
        ChatLog.insert(END,"Bot: The Active corona cases in "+str(casesOf[0])+" are "+str(casesOf[1])+'\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
    else:
        if msg != '':
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + '\n\n')
            ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        
            res = chatbot_response(msg)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
                
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
    

base = Tk()
base.title("Q/A System")
base.geometry("600x600")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=577,y=6, height=500)
ChatLog.place(x=6,y=6, height=500, width=575)
EntryBox.place(x=128, y=505, height=90, width=575)
SendButton.place(x=6, y=505, height=90)

base.mainloop()
