import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image


pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_note_authentication(spl,spw,pel,pew):
     
    prediction=classifier.predict([[spl,spw,pel,pew]])
    print(prediction)
    return prediction



def main():
    st.title("Steamlit ML App")
    html_temp = """
    <div style="background-color:Red;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Flower Classification</h2>
    </div>
    <br></br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    spl = st.number_input("Sepal Length ")
    spw = st.number_input("Sepal Width")
    pel = st.number_input("Petal Length")
    pew = st.number_input("Petal Width")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(spl,spw,pel,pew)
        st.success('The output is {}'.format(result))
        if result == "Iris-setosa":
          image = Image.open('Iris setosa.jpg')
          st.image(image, caption='Iris setosa')
        elif result == "Iris-versicolor":
          image = Image.open('Iris versicolor.jpg')
          st.image(image, caption='Iris versicolor')
        elif result == "Iris-virginica":
          image = Image.open('Iris virginica.jpg')
          st.image(image, caption='Iris virginica')
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    