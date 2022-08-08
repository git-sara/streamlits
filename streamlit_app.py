import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Streamlit Machine Learning App")

# Input bar 1
LengthRange = st.number_input("Enter LengthRange")

# Input bar 2
BodyMassRange = st.number_input("Enter BodyMassRange")

# Input bar 2
WingSpanRange = st.number_input("Enter WingSpanRange")


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    model = joblib.load("modelstreamlit.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[LengthRange, BodyMassRange, WingSpanRange]], 
                     columns = ["LengthRange", "BodyMassRange", "WingSpanRange"])
   
    
    # Get prediction
    prediction = model.predict(X)[0]
    
    # Output prediction
    st.text(f"This instance is a {prediction}")