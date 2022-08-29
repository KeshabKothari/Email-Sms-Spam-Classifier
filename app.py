import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer_11.pkl', 'rb'))
model = pickle.load(open('spam_11.pkl', 'rb'))


st.markdown(""" <div style='background-color:#205454;padding:20px'>
 <h2 style='color:black;text-align:center;'>Email/SMS Spam Classifier </h2>
 </div>
 """, unsafe_allow_html=True)

input_sms = st.text_area("Enter the message", placeholder="Type Here")

if st.button('Predict'):


    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.error("This is A Spam Email")
    else:
        st.success("This is Not A Spam Email")
