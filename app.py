import streamlit as st
import pandas as pd
import numpy as np
import pickle


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

