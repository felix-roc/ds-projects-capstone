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
expander_upload_data = st.expander(label='Click to expand')

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


# # function for loading some data from the test data
# @st.cache
# def load_data(file):
#     """Load the stored test data and sample something.
#     Not sure what this function does though and not using it?

#     Args:
#         file (_type_): filename

#     Returns:
#         _type_: _description_
#     """
#     data = pd.read_csv(file)
#     model_pred = data.sample(30)
#     return model_pred


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
        "Please click on the Predict button to see the results of the"
        "classification.")
    # st.header('')


# Test Data Container
with container_test_data:
    st.markdown('**This is how a random sample of our raw data looks like:**')
    data = pd.read_csv('file/test_data_final.csv')
    serial = data.sample(1)['serial_number'].to_list()[0]
    data = data[data['serial_number'] == serial]
    data.sort_values('date', inplace=True, ascending=False)
    st.write(data.head(5))

    # predicting on the prepared test data df
    if st.button('Predict on our provided test data'):
        y_pred = predict_rating(url, data)
        y_pred = y_pred.split(':')[1]
        y_pred = y_pred.split(',')[0]
        if y_pred == "false":
            st.write('__**Your hard drive will not fail in the next 30 days!**__')
            st.balloons()
        else:
            st.write('__**Your hard drive might fail in the next 30 days!**__')

    st.header('')
    st.header('')


# UPLOAD CONTAINER
# subheader for uploading own data
with container_upload_data:
    st.image('jpg/HDD.jpg')
    st.subheader('Want to predict for your own hard drive?')
    # with expander_upload_data:
    #upload a file
    dataframe_upload = None
    uploaded_file = st.file_uploader("Choose your file for your own hard drive to upload. Make sure it's only data for the model 'ST4000DM000' from Seagate.", help= 'Drag your files here')
    if uploaded_file is not None:
        dataframe_upload = pd.read_csv(uploaded_file)
        dataframe_upload.sort_values('date', inplace=True, ascending=False)
        st.write(dataframe_upload.head(5))
        st.success('Your file was successfully uploaded!')

        # Predicting HDD failure on your own data
        # executing preprocessing via pipeline
        if st.button('Predict'):
            y_pred = predict_rating(url, dataframe_upload)
            y_pred = y_pred.split(':')[1]
            y_pred = y_pred.split(',')[0]
            if y_pred == "false":
                st.write('__**Your hard drive will not fail in the next 30 days!**__')
                st.balloons()
            else:
                st.write('__**Your hard drive might fail in the next 30 days!**__')
