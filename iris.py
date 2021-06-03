#from itertools import Predicate
#from sklearn.base import is_regressor
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# SIMPLE IRIS FLOWER PREDICTION APP

This app predicts the **Iris flower** type!

***
""")

st.header('User Input Parameters')

def user_input_features():
    sepal_length = st.slider('Sepal length', 4.3,7.9,5.4)
    sepal_width = st.slider('Sepal width', 2.0,4.4,3.4)
    petal_length = st.slider('Petal length', 1.0,6.9,1.3)
    petal_width = st.slider('Petal width', 0.1,2.5,0.2)
    data={
        'sepal_length':sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width':petal_width
    }
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

iris = datasets.load_iris()
x = iris.data #4 feautres, sepal and petal length and width
y = iris.target #class number to which it belongs to

clf = RandomForestClassifier()
clf.fit(x,y)

prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)

st.subheader('Class labels and thier corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_prob)