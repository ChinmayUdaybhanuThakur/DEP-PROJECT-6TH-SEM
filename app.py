import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import time

from streamlit_gsheets import GSheetsConnection  # google sheet

import pandas as pd # for medical record 

#read/write data : google sheet
import gspread
from google.oauth2.service_account import Credentials

# states -> refresh username and password inputs
import SessionState

def main():

    st.title("Health-Companion: All HealthCare Detection Tools at one place")
    st.write("------------------------------------------")
    st.sidebar.title("Command Bar")
    choices = ["Home","Eyes", "COVID", "Skin", "Medical Records"]
    menu = st.sidebar.selectbox("Menu: ", choices)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if menu =="Home":
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text("Setting up the magic...")
        time.sleep(1)
        status_text.success("All Set!")
        st.write("---------------------------------")
        st.write("DetAll Contains 3 main sections: Explore the sections to your left sidebar. Once you select a section, you'll be asked to upload an image. Once uploaded, buttons will pop-up with function calls to the models. The results will be shown on the same page.")
    elif menu == "Eyes":
        st.sidebar.write("It analyzes cataract, diabetic retinopathy and redness levels. Upload an image to get started.")
        st.write("---------------------------")
        image_input = st.sidebar.file_uploader("Choose an eye image: ", type="jpg")
        if image_input:
            img = image_input.getvalue()
            st.sidebar.image(img, width=300)#, height=300)
            detect = st.sidebar.button("Detect Cataract")
            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('eye_models/cataract/model.h5')
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = Image.open(image_input)
            size = (224, 224)
            image = ImageOps.fit(image, size,Image.LANCZOS)  # Image.ANTIALIAS) #PIL.Image.LANCZOS
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            size = st.slider("Adjust Image Size: ", 300, 1000)
            st.image(img, width=size)#, height=size)
            st.write("------------------------------------------------------")
            dr = st.sidebar.button("Analyze Diabetic Retinopathy")
            r = st.sidebar.button("Analyze Redness Levels")
            if detect:
                prediction = model.predict(data)
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class1 > class2:
                    st.markdown("DetAll thinks this is a **Cataract** by {:.2f}%".format(class1 * 100) )
                elif class2 > class1:
                    st.markdown("DetAll thinks this is not **Cataract** by {:.2f}%".format(class2 * 100))
                else:
                    st.write("We encountered an ERROR. This should be temporary, please try again with a better quality image. Cheers!")
            if dr:
                model_d = tensorflow.keras.models.load_model('eye_models/dr/model.h5')
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                image = Image.open(image_input)
                size = (224, 224)
                image = ImageOps.fit(image, size,Image.LANCZOS)# Image.ANTIALIAS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                answer = model_d.predict(data)
                class1 = answer[0,0]
                class2 = answer[0,1]
                if class1 > class2:
                    st.write("Diabetic Retinopathy Detected. Confidence: {:.2f}".format(class1 * 100))
                    st.write("-------------------------------")
                elif class2 > class1:
                    st.write("Diabetic Retinopathy Not Detected.")
                    st.write("-------------------------------")
            if r:
                model_r = tensorflow.keras.models.load_model('eye_models/redness/model.h5')
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                image = Image.open(image_input)
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
                image_array = np.asarray(image)
                image.show()
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                answer = model_r.predict(data)
                class1 = answer[0,0]
                class2 = answer[0,1]
                if class1 > class2:
                    st.write("Redness Levels: {:.2f}%".format(class1 * 100))
                    st.write("-------------------------------")
                elif class2 > class1:
                    st.write("No Redness Detected. Confidence: {:.2f}%".format(class2 * 100))
                    st.write("-------------------------------")

    elif menu == "COVID":
        st.sidebar.write("It uses CT Scans to detect whether the patient is likely to have COVID or not. Upload an image to get started.")
        st.write("---------------------------")
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_input = st.sidebar.file_uploader("Choose a file: ", type=['png', 'jpg'])
        if image_input:
            img = image_input.getvalue()
            analyze = st.sidebar.button("Analyze")
            size = st.slider("Adjust image size: ", 300, 1000)
            st.image(img, width=size, height=size)
            st.write("-----------------------------------------")
            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('covid_model/model.h5')
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            if analyze: 
                image = Image.open(image_input)
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                normalized_image_array.resize(data.shape)
                data[0] = normalized_image_array
                prediction = model.predict(data)
                print(prediction)
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class2 > class1:
                    st.markdown("**Possibility of COVID.** Confidence: {:.2f}%".format(class2 * 100))
                elif class1 > class2:
                    st.markdown("**Unlikely to have COVID** Confidence: {:.2f}".format(class1 * 100))
                else:
                    st.write("Error! Please upload a better quality image for accuracy.")
                    
    elif menu == "Skin":
        st.sidebar.write("It detects whether the patient has benign or malignant type of cancer. Further classifications are still under testing. Upload an image to get started.")
        st.write("---------------------------")
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_input = st.sidebar.file_uploader("Choose a file: ", type='jpg')
        if image_input:
            img = image_input.getvalue()
            analyze = st.sidebar.button("Analyze")
            size = st.slider("Adjust image size: ", 300, 1000)
            st.image(img, width=size, height=size)
            st.write("-----------------------------------------")
            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('skin_model/model.h5')
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            if analyze: 
                image = Image.open(image_input)
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                class1 = prediction[0,0]
                class2 = prediction[0,1]
                if class1 - class2 > 0.5:
                    st.markdown("**Benign Detected.** Confidence: {:.2f}%".format(class1 * 100))
                elif class2 - class1 > 0.5:
                    st.markdown("**Malign Detected.** Confidence: {:.2f}".format(class2 * 100))
                else:
                    st.write("Error! Please upload a better quality image for accuracy.")

    elif menu =="Medical Records":
        session_state = SessionState.get(username="", password="")

        session_state.username = st.text_input("Username", value="")
        new_username = session_state.username

        session_state.password = st.text_input("Password", type="password", value="")

        if session_state.username == "admin" and session_state.password == "admin":

            # Load the JSON file
            import json
            with open('/Users/chinmay/Documents/CHINMAY/DEP/Eyes-Diseases-Detctor-main/dep-6th-sem-iit-ropar-a98966aeae57.json') as json_file:
                secrets = json.load(json_file)

            # Extract the credentials
            scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            creds = Credentials.from_service_account_info(secrets, scopes=scope)

            st.sidebar.write("Enter your medical information or view your records.")
            action = st.sidebar.radio("Choose an action: ", ["Enter Information", "View Records"])

            if action == "Enter Information":
                # ... code to enter information ...

                client = gspread.authorize(creds)                
                # Open the Google Spreadsheet by its name (make sure you've shared it with the client email)                
                sheet = client.open("Data").sheet1                
                    # Write data to the Google Spreadsheet   

                client = gspread.authorize(creds)
                sheet = client.open("Data").sheet1
                records = sheet.get_all_records()
                df = pd.DataFrame(records)

                # Medical Information Inputs
                medical_info = {}

                # Personal Information
                medical_info['Name'] = st.text_input('Name: ')
                medical_info['Age'] = st.number_input('Age: ', min_value=0, max_value=120, step=1)
                medical_info['Height'] = st.number_input('Height (in cm): ', min_value=0, max_value=250, step=1)
                medical_info['Weight'] = st.number_input('Weight (in kg): ', min_value=0, max_value=200, step=1)
                medical_info['Gender'] = st.selectbox('Gender: ', ['Male', 'Female', 'Other'])
                medical_info['Blood Sugar Level'] = st.number_input('Blood Sugar Level: ', min_value=0.0, max_value=500.0, step=0.1)
                medical_info['Blood Pressure'] = st.text_input('Blood Pressure (e.g., 120/80): ')
                medical_info['Date of Visit'] = st.date_input('Date of Visit: ')
                medical_info['Is Heart Patient'] = st.selectbox('Is Heart Patient: ', ['True', 'False'])
                medical_info['Is Sugar Patient'] = st.selectbox('Is Sugar Patient: ', ['True', 'False'])

                # Save Information Button
                save_info = st.button("Save Information")

                if save_info:  
                    new_data.update(medical_info)
                    row = list(new_data.values())
                    sheet.append_row(row)
                    st.success("Information saved successfully!")

            elif action == "View Records":
                # Authorize and open the Google Spreadsheet
                client = gspread.authorize(creds)
                sheet = client.open("Data").sheet1

                # Get all records from the sheet
                records = sheet.get_all_records()

                # Convert the records to a pandas DataFrame for easier manipulation
                df = pd.DataFrame(records)

                # Display the DataFrame in the Streamlit app
                st.dataframe(df)

        else:
            st.error("Invalid username or password.")

if __name__ == '__main__':
    main()
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


