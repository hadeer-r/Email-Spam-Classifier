import streamlit as st
import requests
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title='Email Classification', page_icon=':star:')

def load_lottie(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

vectorizer = joblib.load("vectorization_text")

def prepare_text(text):
    encoded_email = vectorizer.transform([text])
    return encoded_email


load_model_svm=joblib.load(open("email_classification_model_svm", 'rb'))
load_model_log=joblib.load(open("email_classification_model_logistic", 'rb'))
load_model_dt=joblib.load(open("email_classification_model_dt", 'rb'))

st.write('# Test Your Email..')

lottie_link = "https://lottie.host/9db3b8ea-e934-4609-a152-d0d3bd0c6957/gRFSY2LV1N.json"
animation = load_lottie(lottie_link)

st.write('---')
st.subheader('Enter Your Text Email')


email_text = st.text_area("Your email here...", height=100)
models=st.selectbox('Choose Model', ['Logistic Regression Model', 'SVC Model', 'Decision Tree Model'])

sample = prepare_text(email_text)

if st.button('Predict') and models=='Decision Tree Model':
    pred_Y=load_model_dt.predict(sample)
    if pred_Y==0:
        st.balloons()
        st.write("the email is ham")
    else:
        st.write("!!!sorry D: , this email is spam")

elif models=='Logistic Regression Model':
    pred_Y=load_model_log.predict(sample)
    if pred_Y==0:
        st.balloons()
        st.write("the email is ham")
    else:
        st.write("!!!sorry D: , this email is spam")

else :
    pred_Y=load_model_svm.predict(sample)
    if pred_Y==0:
        st.balloons()
        st.write("the email is ham")
    else:
        st.write("!!!sorry D: , this email is spam")


        


