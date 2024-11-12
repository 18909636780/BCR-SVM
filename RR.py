#Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('SVM.pkl')

# Define feature options
Level_of_Education_options = {    
    0: 'Primary (0)',    
    1: 'Secondary (1)',    
    2: 'Certificate (2)',    
    3: 'Diploma (3)',
    4: 'Degree (4)'
}

Tumor_Grade_options = {       
    1: 'Grade1 (1)',    
    2: 'Grade2 (2)',    
    3: 'Grade3 (3)',
    4: 'Grade4 (4)'
}

# Define feature names
feature_names = ["Level_of_Education", "Tumor_Size_2_Years_after_Surgery", "Tumor_Grade", "Lymph_Node_Metastasis", "Numbe_of_Lymph_Nodes", "Marital_Status_Unmarried", "Marital_Status_Married", "Marital_Status_Divorced"]

# Streamlit user interface
st.title("Breast Cancer Recurrence Predictor")

# Level_of_Educations
Level_of_Education = st.selectbox("Level of Education:", options=list(Level_of_Education_options.keys()), format_func=lambda x: Level_of_Education_options[x])

# Tumor_Size_2_Years_after_Surgery
Tumor_Size_2_Years_after_Surgery = st.number_input("Tumor Size 2 Years after Surgery(mm):", min_value=0, max_value=100, value=50)

# Tumor_Grade
Tumor_Grade = st.selectbox("Tumor Grade:", options=list(Tumor_Grade_options.keys()), format_func=lambda x: Tumor_Grade_options[x])

# Lymph_Node_Metastasis
Lymph_Node_Metastasis = st.selectbox("Lymph Node Metastasis:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Numbe_of_Lymph_Nodes
Numbe_of_Lymph_Nodes = st.number_input("Numbe of Lymph Nodes:", min_value=0, max_value=50, value=25)

# Marital_Status_Unmarried
Marital_Status_Unmarried = st.selectbox("Marital Status Unmarried:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Marital_Status_Married
Marital_Status_Married = st.selectbox("Marital Status Married:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Marital_Status_Divorced
Marital_Status_Divorced = st.selectbox("Marital Status Divorced:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Process inputs and make predictions
feature_values = [ Level_of_Education, Tumor_Size_2_Years_after_Surgery, Tumor_Grade, Lymph_Node_Metastasis, Numbe_of_Lymph_Nodes, Marital_Status_Unmarried, Marital_Status_Married, Marital_Status_Divorced]
features = np.array([feature_values])

if st.button("Predict"):   
    # Predict class and probabilities   
    predicted_class = model.predict(standardized_features)[0]    
    predicted_proba = model.predict_proba(standardized_features)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")(1: Disease, 0: No Disease)")     
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

 # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:  
        advice = (  
            f"According to our model, you have a high risk of breast cancer recurrence. "  
            f"The model predicts that your probability of having breast cancer recurrence is {probability:.1f}%. "  
            "While this is just an estimate, it suggests that you may be at significant risk. "  
            "I recommend that you consult a doctor as soon as possible for further evaluation and "  
            "to ensure you receive an accurate diagnosis and necessary treatment."  
        )  
    else:  
        advice = (  
            f"According to our model, you have a low risk of breast cancer recurrence. "  
            f"The model predicts that your probability of not having breast cancer recurrence is {100 - probability:.1f}%. "  
            "However, maintaining a healthy lifestyle is still very important. "  
            "I recommend regular check-ups to monitor your health, "  
            "and to seek medical advice promptly if you experience any symptoms."  
        )  
  
    st.write(advice)  

# Calculate SHAP values and display force plot   
   st.subheader("SHAP Force Plot Explanation")
   explainer = shap.KernelExplainer(model)    
   shap_values = explainer.shap_values(pd.DataFrame(feature_values, columns=features_names))
# Display the SHAP force plot for the predicted class    
    if predicted_class == 1:        
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame(features, columns=feature_names)
, matplotlib=True)    
    else:        
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame(features, columns=feature_names)
, matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')