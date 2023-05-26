import streamlit as st
import pandas as pd
import numpy as np
import pickle

# This below code is only to add background image
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('Insurance.jpg')   

# From here code for our project starts
model = pickle.load(open('model_final.pkl','rb'))
encoder = pickle.load(open('target_encoder.pkl','rb'))
transformer = pickle.load(open('transformer.pkl','rb'))

st.title("Insurance Premium Prediction")

age = st.text_input("Enter your age", 23)

sex = st.selectbox("Please select your gender", 
                      ('male','female'))

bmi = st.text_input("Enter your BMI", 20)
bmi = float(bmi)

children = st.selectbox("No. of childrens",(0,1,2,3,4,5,6,7))
children = int(children)

smoker = st.selectbox("Please Select smoker category", ('yes','no'))

region = st.selectbox("Select the region",
                      ("southwest","northwest","southeast","northeast"))

# we have to store all in dictionary 
l = {}
l['age'] = age
l['sex'] = sex
l['bmi'] = bmi
l['children'] = children
l['smoker'] = smoker
l['region'] = region

# store dictionary in dataframe
df = pd.DataFrame(l, index=[0])

# Now we do encoding of all features
df['region'] = encoder.transform(df['region'])
df['sex'] = df['sex'].map({'male' : 1, 'female' : 0})
df['smoker'] = df['smoker'].map({'yes' : 1, 'no' : 0})

df = transformer.transform(df)
y_pred = model.predict(df)

# Add submit button
if st.button("Show Results"):
    st.header(f"{round(y_pred[0], 2)} INR")

