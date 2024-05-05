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
            client = gspread.authorize(creds)                
                # Open the Google Spreadsheet by its name (make sure you've shared it with the client email)                
            sheet = client.open("Data").sheet1                
                # Write data to the Google Spreadsheet   

            session_state.username = st.text_input("Username", value="")
            new_username = session_state.username
            client = gspread.authorize(creds)
            sheet = client.open("Data").sheet1
            records = sheet.get_all_records()
            df = pd.DataFrame(records)

            if new_username in df["Username"].values:
                st.error("This username already exists. Please choose another username.")

            else:

                st.success("Unique Username detected, kindly continue")

                session_state.password = st.text_input("Password", type="password", value="")

                # Medical Information Inputs
                medical_info = {}

                # Personal Information
                medical_info['Date of Birth:'] = st.text_input('Date of Birth: ')
                medical_info['Gender:'] = st.selectbox('Gender: ', ['Male', 'Female', 'Other'])
                medical_info['Address:'] = st.text_input('Address: ')
                medical_info['Contact Information:'] = st.text_input('Contact Information: ')

                # Medical History
                medical_info['Medical History'] = st.text_input('Medical History: ')
                medical_info['Allergies'] = st.text_input('Allergies: ')
                medical_info['Current Medications'] = st.text_input('Current Medications: ')
                medical_info['Past Surgeries'] = st.text_input('Past Surgeries: ')
                medical_info['Family History'] = st.text_input('Family History: ')

                # Vital Signs
                medical_info['Blood Pressure'] = st.text_input('Blood Pressure (e.g., 120/80)(Numeric value necessary)')
                medical_info['Heart Rate'] = st.text_input('Heart Rate (beats per minute)(Numeric value necessary)')
                medical_info['Respiratory Rate'] = st.text_input('Respiratory Rate (Numeric value necessary)')
                medical_info['Temperature'] = st.text_input('Temperature (degrees Celsius) (Numeric value necessary)')
                medical_info['Oxygen Saturation'] = st.text_input('Oxygen Saturation (%) (Numeric value necessary)')

                # Laboratory Results
                medical_info['Laboratory Results'] = st.text_input('Laboratory Results: ')

                # Medication List
                medical_info['Medication List'] = st.text_input('Medication List: ')

                # Family Medical History
                medical_info['Family Medical History'] = st.text_input('Family Medical History: ')

                # Social History
                medical_info['Social History'] = st.text_input('Social History: ')

                # Medical Imaging Results
                medical_info['Medical Imaging Results'] = st.text_input('Medical Imaging Results: ')

                # Progress Notes
                medical_info['Progress Notes'] = st.text_input('Progress Notes: ')

                # Specialist Reports
                medical_info['Specialist Reports'] = st.text_input('Specialist Reports: ')

                # Emergency Contact Information
                medical_info['Emergency Contact Information'] = st.text_input('Emergency Contact Information: ')

                # Save Information Button
                save_info = st.button("Save Information")


                if save_info:  

                    new_data = {"Username": session_state.username, "Password": session_state.password}
                    new_data.update(medical_info)
                    row = list(new_data.values())
                    sheet.append_row(row)
        
                    st.success("Information saved successfully!")


        elif action == "View Records":
            session_state.username = st.text_input("Username", value="")
            session_state.password = st.text_input("Password", type="password", value="enter again")
            view_records = st.button("View Records")

            
            new_username = session_state.username
            client = gspread.authorize(creds)
            sheet = client.open("Data").sheet1
            records = sheet.get_all_records()
            df = pd.DataFrame(records)

            if view_records:
                # Authorize using the credentials and open the Google Spreadsheet
                
                # Filter the DataFrame for the entered username and password
                
                user_data = df[(df["Username"] == session_state.username) & (df["Password"] == session_state.password)]
                if not user_data.empty:
                    
                        # Access each column individually
                    dob = user_data["Date of Birth"].values[0]
                    gender = user_data["Gender"].values[0]
                    address = user_data["Address"].values[0]
                    contact_info = user_data["Contact Information"].values[0]
                    medical_history = user_data["Medical History"].values[0]
                    allergies = user_data["Allergies"].values[0]
                    current_medications = user_data["Current Medications"].values[0]
                    past_surgeries = user_data["Past Surgeries"].values[0]
                    family_history = user_data["Family History"].values[0]
                    social_history = user_data["Social History"].values[0]
                    medical_imaging_results = user_data["Medical Imaging Results"].values[0]
                    lab_results = user_data["Laboratory Results"].values[0]
                    specialist_reports = user_data["Specialist Reports"].values[0]
                    emergency_contact_info = user_data["Emergency Contact Information"].values[0]
                    progress_notes = user_data["Progress Notes"].values[0]

                        # Create a DataFrame for the information
                    info_df = pd.DataFrame({
                        'Date of Birth:': [dob],
                        'Gender:': [gender],
                        'Address:': [address],
                        'Contact Information:': [contact_info],
                        'Medical History': [medical_history],
                        'Allergies': [allergies],
                        'Current Medications': [current_medications],
                        'Past Surgeries': [past_surgeries],
                        'Family History': [family_history],
                        'Social History': [social_history],
                        'Medical Imaging Results': [medical_imaging_results],
                        'Laboratory Results': [lab_results],
                        'Specialist Reports': [specialist_reports],
                        'Emergency Contact Information': [emergency_contact_info],
                        'Progress Notes': [progress_notes]
                    })
                        # Transpose the DataFrame
                    #info_df = info_df.transpose()
                        # Display the DataFrame
                    st.write(info_df)
                else:
                    st.error("No records found for this user.")

if __name__ == '__main__':
    main()
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
