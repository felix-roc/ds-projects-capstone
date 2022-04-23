import requests

import pandas as pd
import streamlit as st
import json

import warnings
warnings.filterwarnings('ignore')

# Define containers
container_intro = st.container()
container_test_data = st.container()
container_upload_data = st.container()

# URL for request to backend
url = 'https://hdd-predicitve-maintenance-api.herokuapp.com/'


def predict_rating(df, url=url):
    """Query the API to obtain predictions

    Args:
        df (_type_): Dataframe to send to the API
        url (_type_, optional): URL of the API.

    Returns:
        _type_: Predictions for each row in the dataframe
    """
    data = json.loads(df.to_json(orient='columns'))
    headers = {'Content-Type': 'application/json'}
    r = requests.post(url, json=data, headers=headers)
    y_pred = r.text
    return y_pred


# Sidebar
st.sidebar.title('Predictive maintenance of HDDs in data centers')
st.sidebar.image('jpg/Guardians_memory.jpg')
st.sidebar.subheader("Guardians of the Memory", )
st.sidebar.text('Felix, Chang Ming, Andreas & Daniela')

st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.text('Ⓒ 2022. All rights Reserved.')


# Intro Container
with container_intro:
    st.title('How long will your hard drive last?')
    st.write(
        "This is a web app to predict if a HDD drive will fail or not fail in"
        " the next 30 days.")
    st.write(
        "Please click on the Predict button to see the results of the "
        "classification.")


# Test Data Container
with container_test_data:
    st.markdown('**This is how a random sample of our raw data looks like:**')
    # Read the provided test data
    data = pd.read_csv('file/test_data_final.csv')
    # Sample a random drive from the list
    serial = data.sample(1)['serial_number'].to_list()[0]
    data = data[data['serial_number'] == serial]
    # Sort values chronologically
    data.sort_values('date', inplace=True, ascending=False)
    # Display the first 5 entries
    st.write(data.head(5))

    # predicting on the prepared test data
    if st.button('Predict on our provided test data'):
        y_pred = predict_rating(url, data)
        y_pred = y_pred.split(':')[1]
        y_pred = y_pred.split(',')[0]
        if y_pred == "false":
            st.write(
                '__**The HDD will NOT fail in the next 30 days**__')
            st.balloons()
        else:
            st.write('__**The HDD will fail in the next 30 days!**__')


# Upload Container
with container_upload_data:
    st.image('jpg/HDD.jpg')
    st.subheader('Want to predict for your own hard drive?')
    # Upload a file
    dataframe_upload = None
    uploaded_file = st.file_uploader(
        "Choose your file for your own hard drive to upload. Make sure it's "
        "only data for the model 'ST4000DM000' from Seagate.",
        help='Drag your files here')
    if uploaded_file is not None:
        # Read to uploaded file
        dataframe_upload = pd.read_csv(uploaded_file)
        # Sort rows chronologically
        dataframe_upload.sort_values('date', inplace=True, ascending=False)
        # Display header
        st.write(dataframe_upload.head(5))
        st.success('Your file was uploaded successfully!')

        if st.button('Predict'):
            # Predict on own data using API
            y_pred = predict_rating(url, dataframe_upload)
            # Process the returned predictions
            y_pred = y_pred.split(':')[1]
            y_pred = y_pred.split(',')[0]
            if y_pred == "false":
                st.write(
                    '__**Your hHDD will NOT fail in the next 30 days!**__')
                st.balloons()
            else:
                st.write(
                    '__**Your HDD will fail in the next 30 days!**__')
