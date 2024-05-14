import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras.models
import pandas as pd
import time
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.mixture import GaussianMixture
import SessionState
# pip install st-gsheets-connection
from streamlit_gsheets import GSheetsConnection  # google sheet
#read/write data : google sheet
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="Health Companion",
    page_icon='🏥',
    layout="centered",
)

def main():
    st.sidebar.title("Health Companion🏥")
    super_menu = st.sidebar.selectbox("Menu: ", ["Health Companion Home", "Health Analysis by Computer Vision", "Sales Data Prediction", "Patient Visualisation", "Medical Records"])
    st.title(super_menu)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if super_menu == "Health Companion Home":
        display_home()

    elif super_menu == "Health Analysis by Computer Vision":
        menu = st.radio("Menu: ", ["Eyes-Risk", "Skin", "X-Ray Prediction"])
        if menu == "Eyes-Risk":
            with st.expander("Know more about Eye - Risks in humans: "):
                st.markdown('''<u>**Understanding Eye Risk</u>:**
Eye risk assessment involves the evaluation of various factors that may indicate potential health issues related to the eyes. This assessment often focuses on detecting abnormalities or signs of diseases such as cataracts and retinopathy, which can affect vision and overall eye health.

**<u>Factors Considered in Eye Risk Assessment:</u>**\n
1. **Cloudiness Detection**: Cloudiness or protein buildup in the eyes can be an early indicator of certain eye conditions. By analyzing close-up images of the eyes, AI models can detect cloudiness patterns, providing insights into potential risks.\n
2. **Identification of Abnormalities**: AI-powered analysis helps in identifying abnormalities in eye images, such as irregularities in the retina or lens. These abnormalities may signal the presence of underlying eye diseases that require further evaluation and treatment.''', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image("Cataract.jpg", caption="Eye with cloudiness (Cataract)", width=300)
                with col2:
                    st.image("Clear Eye.jpeg", caption="Clear Eye", width=300)


            st.write("The AI Model detects cloudiness(Protein Buildup) in your closeup eye-image to check for risk of many diseases (including Cataract and Retinopathy)")
    
            st.write("Please ensure that your image contains the eye as the majority subject, and the iris is free from any light reflections")
            st.markdown(''':red[Please note that the AI Models present in our backend are not professional medical advice, only a learning method]''')
            with st.expander("Upload Image: "):
                image_input = st.file_uploader("Choose an eye image: ", type=['png', 'jpg'])
                start_camera = st.checkbox("Start Camera")
            
            if image_input:
                    img = image_input.getvalue()
                    st.image(img, width=300)#, height=300)
                    detect = st.button("Run Analysis using uploaded model")
                    np.set_printoptions(suppress=True)
                    model = tensorflow.keras.models.load_model('model.h5',compile=False)
                    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                    image = Image.open(image_input)
                    size = (224, 224)
                    image = ImageOps.fit(image, size,Image.LANCZOS)  # Image.ANTIALIAS) #PIL.Image.LANCZOS
                    image_array = np.asarray(image)
                    #normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                    #data[0] = normalized_image_array
                    data[0] = image_array
                    st.write("------------------------------------------------------")
                  
                    if detect:
                        prediction = model.predict(data)
                        class1 = prediction[0,0]
                        class2 = prediction[0,1]
                        if class1 > class2:
                            st.markdown("Your Model predicts the eye cloudiness risk is {:.2f}%".format(class1 * 100) )
                        elif class2 > class1:
                            st.markdown("Your model thinks the eyes are clear with confidence {:.2f}%".format(class2 * 100))
                        else:
                            st.write("Please try again with a better quality image.")
        
            elif start_camera:
                picture = st.camera_input("Take a picture", key="eye_photo" ,help="Click a close up photo of your eye so that we can check and analyse it")
                if picture:
                    img = picture.getvalue()
                    st.image(img, width=300)#, height=300)
                    detect = st.button("Run Analysis using uploaded model")
                    np.set_printoptions(suppress=True)
                    model = tensorflow.keras.models.load_model('model.h5',compile=False)
                    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                    image = Image.open(picture)
                    size = (224, 224)
                    image = ImageOps.fit(image, size,Image.LANCZOS)  # Image.ANTIALIAS) #PIL.Image.LANCZOS
                    image_array = np.asarray(image)
                    #normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                    #data[0] = normalized_image_array
                    data[0] = image_array
                    st.write("------------------------------------------------------")
                  
                    if detect:
                        prediction = model.predict(data)
                        class1 = prediction[0,0]
                        class2 = prediction[0,1]
                        if class1 > 3*class2:
                            st.markdown("Your Model predicts the eye cloudiness risk is {:.2f}%".format(class1 * 100) )
                        elif class2 > 3*class1:
                            st.markdown("Your model thinks the eyes are clear with confidence {:.2f}%".format(class2 * 100))
                        else:
                            st.write(" Please try again with a better quality image.")

        elif menu == "Skin":
            with st.expander("Know more about Skin Diseases"):
                st.markdown('''
                    <u>Understanding Skin Diseases</u>:
                    Skin diseases encompass a wide range of conditions that affect the skin's appearance, texture, and overall health. These conditions can be caused by various factors, including genetics, environmental factors, infections, and autoimmune disorders.

                    Various Types of Skin Cancer which our model can detect are:

                    1. **Actinic Keratoses and Intraepithelial Carcinomae (akiec):** Actinic keratoses are rough, scaly patches on the skin caused by excessive sun exposure. Intraepithelial carcinoma refers to abnormal cell growth within the skin's outer layer. Both conditions are considered precursors to skin cancer.

                    2. **Basal Cell Carcinoma (bcc):** Basal cell carcinoma is the most common type of skin cancer, typically appearing as a small, flesh-colored or pink bump on the skin. It often develops on areas exposed to the sun and grows slowly.

                    3. **Benign Keratosis-like Lesions (bkl):** Benign keratosis-like lesions refer to non-cancerous growths on the skin that resemble actinic keratoses but are not associated with a risk of cancer. They may include seborrheic keratosis, solar lentigo, and other benign growths.

                    4. **Dermatofibroma (df):** Dermatofibroma is a benign skin lesion that commonly appears as a small, firm bump on the skin. It is composed of fibrous tissue and often develops on the legs, although it can occur elsewhere on the body.

                    5. **Melanocytic Nevi (nv):** Melanocytic nevi, commonly known as moles, are benign growths on the skin that develop when melanocytes, the cells that produce pigment, cluster together. Moles can vary in size, color, and shape.

                    6. **Pyogenic Granulomas and Hemorrhage (vasc):** Pyogenic granulomas are benign vascular growths that often appear as red, moist bumps on the skin. They can bleed easily and may develop in response to injury or hormonal changes during pregnancy.

                    7. **Melanoma (mel):** Melanoma is a type of skin cancer that develops from melanocytes, the cells that produce pigment. It can occur anywhere on the body and is often characterized by changes in the size, shape, or color of moles or the appearance of new pigmented lesions. Melanoma is aggressive and can spread to other parts of the body if not treated early.
                    ''', unsafe_allow_html=True)
                
                st.write("Some pictorial examples of skin diseases:")
                col1, col2 = st.columns(2)
                with col1:
                    st.image("rednesscarcinoma.png", caption = "Redness/Carcinoma", width=300)
                with col2:
                    st.image("mole.jpeg", caption="class 4 Mole ", width=300)

            st.markdown(''':red[Please note that the AI Models present in our backend are not professional medical advice, only a learning method]''')
            with st.expander("Upload Image: "):
                image_input = st.file_uploader("Choose a CLOSEUP image of the affected skin, with  only the skin present in the image: ", type=['png', 'jpg'])
                start_camera = st.checkbox("Start Camera")

            if image_input:
                    img = image_input.getvalue()
                    
                    st.image(img)
                    st.write("-----------------------------------------")
                    analyze = st.button("Run Analysis using uploaded model")
                    np.set_printoptions(suppress=True)
                    #model = tensorflow.models.load_model('model.h5')
                    model = tensorflow.keras.models.load_model('best_model (1).h5',compile =False)
                    
                    data = np.ndarray(shape=(1, 28, 28, 3), dtype=np.float32)
                    if analyze: 
                        image = Image.open(image_input)
                        size = (28, 28)
                        image = ImageOps.fit(image, size, Image.LANCZOS)
                        image_array = np.asarray(image)
                        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                        data[0] = normalized_image_array[:, :, :3]
                        prediction = model.predict(data)
                        st.write(prediction)
                        st.write("0/1/3/5: ", prediction[0,0]+prediction[0,1]+prediction[0,3]+prediction[0,5], "2: ", prediction[0,2], "4/6: ", prediction[0,4] + prediction[0,6])
                        class1 = prediction[0,0]
                        class2 = prediction[0,1]
                        st.markdown("*0:* Classless-Clear")
                        st.markdown("*1:* Basal Cell Carcinoma (bcc)")
                        st.markdown("*2:* Benign Keratosis-like Lesions (bkl)")
                        st.markdown("*3:* Dermatofibroma (df)")
                        st.markdown("*4:* Melanocytic Nevi (nv)")
                        st.markdown("*5:* Pyogenic Granulomas and Hemorrhage (vasc)")
                        st.markdown("*6:* Melanoma (mel)")
                        st.markdown(''' 
                                :red[NOTE] \n 
                                1. Please note that :blue[akiec, bcc, df and vasc] may look similar in photos, therefore we have combined their probabilities
                                2. Similarly :blue[nv and mel] may look similar in photos, therefore we have combined their probabilities''')
                        

            if(start_camera):
                picture = st.camera_input("Take a picture", key="Lesion_photo" ,help="Click a close up photo of your skin so that we can check and analyse it")
                if picture:
                    img = picture.getvalue()
                    
                
                    st.image(img)
                    st.write("-----------------------------------------")
                    analyze = st.button("Run Analysis using uploaded model")
                    np.set_printoptions(suppress=True)
                    #model = tensorflow.models.load_model('model.h5')
                    model = tensorflow.keras.models.load_model('best_model (1).h5',compile=False)

                    data = np.ndarray(shape=(1, 28, 28, 3), dtype=np.float32)
                    if analyze: 
                        image = Image.open(picture)
                        size = (28, 28)
                        image = ImageOps.fit(image, size, Image.LANCZOS)
                        image_array = np.asarray(image)
                        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                        #data[0] = first 3 channels of normalised image array
                        data[0] = normalized_image_array[:, :, :3]
                        # data[0] = normalized_image_array
                        # data[0] = normalized_image_array
                        prediction = model.predict(data)
                        st.write(prediction)
                        st.write("0/1/3/5: ", prediction[0,0]+prediction[0,1]+prediction[0,3]+prediction[0,5], "2: ", prediction[0,2], "4/6: ", prediction[0,4] + prediction[0,6])
                        class1 = prediction[0,0]
                        class2 = prediction[0,1]
                        st.markdown("*0:* Actinic Keratoses and Intraepithelial Carcinomae (akiec)")
                        st.markdown("*1:* Basal Cell Carcinoma (bcc)")
                        st.markdown("*2:* Benign Keratosis-like Lesions (bkl)")
                        st.markdown("*3:* Dermatofibroma (df)")
                        st.markdown("*4:* Melanocytic Nevi (nv)")
                        st.markdown("*5:* Pyogenic Granulomas and Hemorrhage (vasc)")
                        st.markdown("*6:* Melanoma (mel)")
                        st.markdown(''' 
                                :red[NOTE] \n 
                                1. Please note that :blue[akiec, bcc, df and vasc] may look similar in photos, therefore we have combined their probabilities
                                2. Similarly :blue[nv and mel] may look similar in photos, therefore we have combined their probabilities''')
                        if class1 - class2 > 0.5:
                            st.markdown("**Benign Detected.** Confidence: {:.2f}%".format(class1 * 100))
                        elif class2 - class1 > 0.5:
                            st.markdown("**Malign Detected.** Confidence: {:.2f}".format(class2 * 100))
                        else:
                            st.write("Error! Please upload a better quality image for accuracy.")


        elif menu == "X-Ray Prediction":
            with st.expander("Know more about X-rays & Pneumonia"):
                st.markdown('''**Pneumonia** is lung inflammation often caused by infections. It leads to symptoms like 
                            cough, fever, and difficulty breathing.

**Detection through X-Ray:**
Chest X-rays show areas of increased opacity in the lungs, indicating inflammation 
or fluid buildup. Radiologists analyze these patterns to diagnose pneumonia and determine its severity.''', unsafe_allow_html=True)
                
                # col1, col2 = st.columns(2)
                # with col1:
                #     st.image("C:\\Users\\Lenovo\\Desktop\\Pneumonia.jpeg", caption="Pictorial Representation of Pneunmonia", width=300)
                # with col2:
                #     st.image("C:\\Users\\Lenovo\\Desktop\\Pneumonia X Ray.jpeg", caption="X Ray of Pneumonia affected person", width=300)

            st.markdown(''':red[Please note that the AI Models present in our backend are not professional medical advice, only a learning method]''')
            with st.expander("Upload X-Ray Image: "):
                image_input = st.file_uploader("Choose a clear image of X-Ray: ", type=['png', 'jpg'])
                #start_camera = st.checkbox("Start Camera")

            if image_input:
                    img = image_input.getvalue()
                    st.image(img)
                    st.write("-----------------------------------------")
                    analyze = st.button("Run Analysis using uploaded model")
                    np.set_printoptions(suppress=True)
                    #model = tensorflow.models.load_model('model.h5')
                    model = tensorflow.keras.models.load_model('CheXNet_model.hdf5',compile=False)
                    model_arch= model.to_json()
                    model= tensorflow.keras.models.model_from_json(model_arch)
                    model.load_weights('chex.h5')
                    
                    data = np.ndarray(shape=(1, 320,320,3), dtype=np.float32)
                    if analyze: 
                        image = Image.open(image_input)
                        size = (320, 320)
                        image = ImageOps.fit(image, size, Image.LANCZOS)
                        image_array = np.asarray(image)
                        normalized_image_array = (image_array.astype(np.float32) / 100.0) - 1
                        data[0] = normalized_image_array[:,:,:3]
                        prediction = model.predict(data)
                        # st.write(prediction)
                        
                        # st.write("0: Atelectasis, 1: Cardiomegaly, 2: Effusion, 3: Infiltration, 4: Mass, 5: Nodule, 6: Pneumonia, 7: Pneumothorax, 8: Consolidation, 9: Edema, 10: Emphysema, 11: Fibrosis, 12: Pleural_Thickening, 13: Hernia")
                        #st.write(prediction[0,3])
                        if(prediction[0,3] < 0.88):
                            st.write("No Abnormality Detected")
                        else:
                            st.write("Abnormality in Chest X-ray Detected")

            

    elif super_menu == "Sales Data Prediction":
        menu = st.radio("Menu", ["Visualise overall sales", "Predict Sales Data"])
        if menu == "Visualise overall sales":
            visualize_sales_data()
        elif menu == "Predict Sales Data":
            predict_sales_data()

    elif super_menu == "Patient Visualisation":
        st.subheader("Visualise Patient Data")
        visualize_patient_data()

    elif super_menu == "Medical Records":
        session_state = SessionState.get(username="", password="")

        session_state.username = st.text_input("Username", value="")
        new_username = session_state.username

        session_state.password = st.text_input("Password", type="password", value="")

        if session_state.username == "admin" and session_state.password == "admin":

            # Load the JSON file
            import json
            with open('medical_records.json') as json_file:
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
                medical_info['Blood Sugar Level'] = st.number_input('Blood Sugar Level (mg/dL): ', min_value=50.0, max_value=150.0, step=0.1)
                medical_info['Blood Pressure'] = st.number_input('Systolic Blood Pressure (e.g., 120): ', min_value=80.0, max_value=170.0, step=0.1)
                medical_info['Date of Visit'] = st.date_input('Date of Visit: ').strftime("%Y-%m-%d")
                medical_info['Is Heart Patient'] = st.selectbox('Is Heart Patient: ', ['True', 'False'])
                medical_info['Is Sugar Patient'] = st.selectbox('Is Sugar Patient: ', ['True', 'False'])

                # Save Information Button
                save_info = st.button("Save Information")

                if save_info:  
                    new_data = {}  # Initialize new_data
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
                if st.button("Show All Medical Records"):
                    st.dataframe(df, use_container_width=True)
                    st.download_button(label='📥 Download Patient Data',
                                data=df.to_csv(index=False),
                                file_name= 'patient_data.csv')

                    # st.write(df[df['Name'] == name])
        else:
            st.error("Invalid username or password.")


def display_home():
    st.subheader("Welcome to Health Companion! 🏥 ")
    st.write("Health Companion is your personal healthcare assistant equipped with various features to cater to your health needs.")
    st.write("Here's a brief overview of the available features:")

    st.write("- **Health Analysis by Computer Vision**:", "_Utilize AI models to analyze eye images for cloudiness risk, assess skin conditions, and predict X-Ray results._")
    st.write("- **Sales Data Prediction**: _Visualize and predict sales data trends for pharmaceutical products._")
    st.write("- **Patient Visualization**: _Explore patient data, visualize clusters using Gaussian Mixture Models, and analyze trends._")

    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
    time.sleep(1)
    status_text.success("All Set!")
    st.write("Please explore the sections from the select menu in the sidebar.")

def visualize_sales_data():
    with st.expander("Upload CSV File: "):
        data_file = st.file_uploader("Upload CSV", type=['csv'])
    if data_file is not None:
        data = pd.read_csv(data_file)
        df = pd.DataFrame(data)
        option = st.selectbox("Options: ", ["Show CSV File", "Cumulative Sales Trend", "Sales Trend for Particular Drug", "Total Sales Breakdown", "Sales Variance Breakdown"])
        
        if option == "Show CSV File":
            st.dataframe(df, use_container_width=True)

        elif option == "Cumulative Sales Trend":
            df = pd.DataFrame(data)
            df['datum'] = pd.to_datetime(df['datum'])

            def plot_all_data(df):
                fig = px.line(df, x='datum', y=df.columns[1:], title='Sales Data Across Time')
                # Customization: Adding unique line styles and colors if needed
                line_styles = ['solid', 'dot', 'dash', 'longdash', 'dashdot']
                colors = px.colors.qualitative.Plotly
                # for i, column in enumerate(df.columns[1:]):
                #     fig.add_scatter(x=df['datum'], y=df[column], mode='lines', name=column,
                #                     line=dict(color=colors[i % len(colors)], dash=line_styles[i % len(line_styles)]))

                st.plotly_chart(fig)            

            plot_all_data(df)

        elif option == "Sales Trend for Particular Drug":
            drug = st.selectbox('Select Drug/Medication', df.columns[1:])
            fig = px.line(df, x='datum', y=drug, title=f'Sales Data for {drug}')
            st.plotly_chart(fig)

        elif option == "Total Sales Breakdown":       
            def plot_sales_distribution_with_variance(df):
                medication_sums = df.drop(columns=['datum']).sum()
                medication_variance = df.drop(columns=['datum']).var()
                
                # Create the pie chart with hover data showing variance
                fig = px.pie(values=medication_sums, names=medication_sums.index,
                            title='Total Sales Distribution among Medications',
                            hover_data={'Variance': medication_variance.round(2)})
                st.plotly_chart(fig)

            plot_sales_distribution_with_variance(df)
        
        elif option == "Sales Variance Breakdown":
            def plot_variance_bar_chart(df):
                # Calculate variance for each medication column
                medication_variance = df.drop(columns=['datum']).var()/df.drop(columns=['datum']).sum()

                # Create the bar chart to display the variance
                fig = px.bar(x=medication_variance.index, y=medication_variance,
                            labels={'x': 'Medication', 'y': 'Normalised Variance'},
                            title='Variance of Sales for Each Medication')
                
                st.plotly_chart(fig)
            
            plot_variance_bar_chart(df)

def predict_sales_data():
    with st.expander("Upload CSV File: "):
        data_file = st.file_uploader("Upload Here: ", type=['csv'])
    
    def forecast_demand_safety_stock(df):
        # User inputs
        drug = st.selectbox('Select the drug for prediction:', df.columns[1:])  # Exclude 'datum'
        lead_time = st.number_input('Enter the lead time in days:', min_value=0, max_value=30, value=10)
        confidence_level = st.radio('Select the confidence level:', [0, 90, 95, 99], horizontal=True)
        lead_time = lead_time/30  # Convert lead time to months
        # Prepare data
        x = np.array(range(len(df))).reshape(-1, 1)  # Time as a simple linear feature
        y = df[drug].values  # Values of the drug sales
        x = x[-20:]
        y = y[-20:]
        # Linear Regression model
        poly_feature = PolynomialFeatures(degree=3)
        x = poly_feature.fit_transform(x)

        model = LinearRegression()
        model.fit(x, y)
        next_three_months = np.array([len(df), len(df) + 1, len(df) + 2]).reshape(-1, 1)
        next_three_months = poly_feature.fit_transform(next_three_months)
        predictions = model.predict(next_three_months)
        
        # Calculate the variance of predictions
        medication_variance = df.drop(columns=['datum']).var()
        drug_variance = medication_variance[drug]
        #st.write(drug_variance)
        # Calculate Safety Stock
        z_score = {0:0, 90: 1.645, 95: 1.96, 99: 2.576}.get(confidence_level, 0)  # Default to 0% if not found
        safety_stock = z_score * np.sqrt(drug_variance)*np.sqrt(lead_time) 

        # Calculate Risk
        average_demand = np.mean(df[drug].values)
        risk_ratio = safety_stock / average_demand
        risk_status = "Risky" if risk_ratio > 0.30   else "Normal"  # Threshold of 0.5 for demonstration

        # Display results
        predictions=[-1*i if i < 0 else i for i in predictions]

        #st.write(f"Predicted demands for the next three months: {predictions}")
        st.write(f"Safety Stock-(Reorder Point): {safety_stock.round(0)}")
        st.write(f"Risk Status: {risk_status} (Safety Stock to Average Demand Ratio: {risk_ratio:.2f})")

        return predictions, safety_stock, average_demand,drug
    
    def plot_drug_sales_with_forecast(df, predictions, safety_stock, average_demand,drug):
        # User input for drug selection
        #drug = st.selectbox('Select the drug to plot:', df.columns[1:])  # Exclude 'datum'
        
        # Prepare data for plotting
        x = pd.date_range(start=df['datum'].min(), periods=len(df), freq='M')
        x_future = pd.date_range(start=x[-1], periods=4, freq='M')[1:]  # Start from the next month

        # Combine historical and predicted data
        y_hist = df[drug].values
        y_future = predictions
        #changing predictions to be postve by multipling with -1 for negative values
        y_future = [-1*i if i < 0 else i for i in y_future]

        # Create the plot
        fig = go.Figure()

        # Add historical data trace
        fig.add_trace(go.Scatter(x=x, y=y_hist, mode='markers', name='Historical Data', line=dict(color='blue')))

        # Add predicted data trace
        fig.add_trace(go.Scatter(x=x_future, y=y_future, mode='markers', name='Predicted Data', line=dict(color='red')))

        # Add a line for decision boundary based on safety stock to average demand ratio
        # decision_boundary = safety_stock / average_demand
        # y_boundary = [decision_boundary * average_demand] * len(x_future)  # Repeat for the length of future predictions

        # fig.add_trace(go.Scatter(x=x_future, y=y_boundary, mode='lines', name='Decision Boundary (Risky if above)', line=dict(color='green', dash='dash')))

        # Update layout with titles and annotations
        fig.update_layout(title=f'Sales Forecast and Risk Analysis for {drug}',
                        xaxis_title='Date',
                        yaxis_title='Sales',
                        legend_title='Legend')

        st.plotly_chart(fig)

    if data_file is not None:
        data_file = pd.read_csv(data_file)
        # print(data_file)
        df = pd.DataFrame(data_file)
        predictions, safety_stock, average_demand,drug = forecast_demand_safety_stock(df)
        plot_drug_sales_with_forecast(df[-10:], predictions, safety_stock, average_demand,drug)

    else:
        st.write("Please upload the CSV file to predict the sales data accordingly.")

def visualize_patient_data():
    with st.expander("Upload CSV File: "):
        data_file = st.file_uploader("Upload CSV", type=['csv'])
    if data_file is not None:
        data = pd.read_csv(data_file)

        option = st.selectbox("Options: ", ["Show CSV File and Description", "Show Individual Data", "Show Histograms", "Show Class Visualisation","Show GMM Graph Clusters", "2D Sideview of the GMMs"])
        
        if option == "Show Individual Data":
            patient_id = st.selectbox("Select Patient ID", data['Name'].unique())
            patient_data = data[data['Name'] == patient_id]
            # Plotting the data blood sugar and blood pressure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=patient_data['Date of Visit'], y=patient_data['Blood Sugar Level'], mode='markers', name='Blood Sugar Level', marker=dict(color='red')))
            fig.add_trace(go.Scatter(x=patient_data['Date of Visit'], y=patient_data['Blood Pressure'], mode='markers', name='Blood Pressure', marker=dict(color='blue')))
            fig.update_layout(title=f'Patient {patient_id} Data', xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig)

        elif option == "Show CSV File and Description":
            st.dataframe(data, use_container_width=True)
            
            if st.checkbox("Show Summary"):
                st.write("Description of the data:")
                st.write(data.describe())

        elif option == "Show Histograms":
            fig, ax = plt.subplots()
            data.hist(ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

        elif option == "Show Class Visualisation":
            def plot_patient_data(df):
                # User input for selecting axes
                x_axis = st.selectbox('Select X-axis:', df.columns.drop(['Name', 'Date of Visit', 'Is Heart Patient', 'Is Sugar Patient']), index=1)
                y_axis = st.selectbox('Select Y-axis:', df.columns.drop(['Name', 'Date of Visit', 'Is Heart Patient', 'Is Sugar Patient']), index=2)
                
                # User selection for patient condition to highlight
                condition = st.selectbox('Highlight condition:', ['None', 'Is Heart Patient', 'Is Sugar Patient'])

                # Define color based on condition
                if condition == 'Is Heart Patient':
                    color = 'Is Heart Patient'
                elif condition == 'Is Sugar Patient':
                    color = 'Is Sugar Patient'
                else:
                    color = None  # No specific coloring based on condition
                
                # Plotting
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color,
                                title=f'Patient Data: {x_axis} vs {y_axis}',
                                labels={color: condition},
                                hover_data=['Name', 'Is Heart Patient', 'Is Sugar Patient'])
                
                # Enhance marker appearance for better visibility
                fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                                selector=dict(mode='markers'))
                
                st.plotly_chart(fig)

            plot_patient_data(data)
        
        if option == "Show GMM Graph Clusters":
            def plot_gmm_3d(df):
                # User selects axes for the GMM
                columns = df.columns.drop(['Name', 'Date of Visit', 'Is Heart Patient', 'Is Sugar Patient', 'Gender'])
                col1, col2 = st.columns(2)

                with col1:    
                    x_axis = st.selectbox('Select X-axis for GMM:', columns, index=1)
                    y_axis = st.selectbox('Select Y-axis for GMM:', columns, index=2)
                    z_axis = st.selectbox('Select Z-axis for GMM:', columns, index=3)
                    
                    # Extract data for fitting
                    data = df[[x_axis, y_axis, z_axis]].dropna()
                    
                    # Fit GMM
                    gmm = GaussianMixture(n_components=2, random_state=42)
                    gmm.fit(data)
                    labels = gmm.predict(data)
                    
                    # Add labels to data for coloring
                    data['Cluster'] = labels
                    df['Cluster'] = labels  
                    # 3D Scatter plot
                    fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis,               color='Cluster',                      
                                        title=f'3D GMM Clusters: {x_axis}, {y_axis}, {z_axis}',
                                        labels={'Cluster': 'GMM Cluster'})#,symbol='Blood Sugar Level')#hover_name='Name')  
                    st.plotly_chart(fig)

            plot_gmm_3d(data)

        elif option == "2D Sideview of the GMMs":
            def plot_two_peaks_plotly():
                # Data generation: two Gaussian distributions
                x1 = np.linspace(-10, 10, 400)
                y1 = np.exp(-(x1 - 1)**2 / 2)  # Gaussian peak with mean = 1, std = 1
                
                x2 = np.linspace(-10, 10, 400)
                y2 = np.exp(-(x2 + 1)**2 / 2)  # Gaussian peak with mean = -1, std = 1

                # Create the figure
                fig = go.Figure()

                # Add traces for each peak
                fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='Peak 1 (Mean = 1)', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='Peak 2 (Mean = -1)', line=dict(color='red', dash='dash')))

                # Add titles and labels
                fig.update_layout(title='Plot of Two Gaussian Peaks',
                                xaxis_title='X-axis',
                                yaxis_title='Density',
                                legend_title='Legend')

                # Show plot
                st.plotly_chart(fig)

            plot_two_peaks_plotly()




if __name__ == "__main__":
    main()
