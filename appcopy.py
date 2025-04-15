import streamlit as st
import torch
import timm
from torchvision import transforms as T
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from disease_info import DISEASE_INFO, HEALTHY_CROP_CONDITIONS, ENVIRONMENTAL_THRESHOLDS
import time

# Page configuration
st.set_page_config(
    page_title="Crop Disease Detection & Management",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .stProgress {
        margin-top: 1rem;
    }
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #f0f2f6
    }
    </style>
""", unsafe_allow_html=True)

# Model Loading
@st.cache_resource
def load_model():
    classes = {
        0: 'Corn_CommonRust', 1: 'CornGray_Leaf_Spot', 2: 'CornHealthy',
        3: 'CornNorthern_Leaf_Blight', 4: 'Potato_EarlyBlight', 5: 'PotatoHealthy',
        6: 'Potato_LateBlight', 7: 'RiceBrown_Spot', 8: 'RiceHealthy',
        9: 'RiceLeaf_Blast', 10: 'Rice_Neck_Blast', 11: 'Sugarcane_Bacterial Blight',
        12: 'Sugarcane_Healthy', 13: 'SugarcaneRed Rot', 14: 'WheatBrown_Rust',
        15: 'WheatHealthy', 16: 'WheatYellow_Rust'
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load("crop_best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model, classes, device

# Image Transformation
def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Disease Prediction
def predict_disease(image, model, classes, device, transform):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.max(probabilities, dim=1)[1]
        probability = probabilities[0][predicted_idx].item()
        predicted_class = classes[predicted_idx.item()]
    
    return predicted_class, probability

# Risk Assessment Visualization
def create_risk_gauge(risk_score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Disease Risk Level"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

# Environmental Risk Calculator
def calculate_environmental_risk(temp, humidity, rainfall, disease):
    if disease not in ENVIRONMENTAL_THRESHOLDS:
        return 50  # Default medium risk
    
    thresholds = ENVIRONMENTAL_THRESHOLDS[disease]
    risk_score = 0
    
    # Temperature risk
    if thresholds['temperature']['min'] <= temp <= thresholds['temperature']['max']:
        risk_score += 33
        if abs(temp - thresholds['temperature']['optimal']) <= 5:
            risk_score += 17
    
    # Humidity risk
    if humidity >= thresholds['humidity']['min']:
        risk_score += 33
        if humidity >= thresholds['humidity']['optimal']:
            risk_score += 17
    
    return risk_score

# Main Application
def main():
    st.title("🌾 Crop Disease Detection & Management System")
    
    # Load model
    model, classes, device = load_model()
    transform = get_transform()
    
    # Sidebar
    st.sidebar.header("Upload Image & Environmental Data")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload crop image", type=['jpg', 'jpeg', 'png'])
    camera_image = st.sidebar.camera_input("Or take a photo instead")
    
    # Environmental inputs
    st.sidebar.subheader("Environmental Conditions")
    temperature = st.sidebar.slider("Temperature (°F)", 40, 100, 75)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 65)
    rainfall = st.sidebar.slider("Recent Rainfall (inches)", 0.0, 10.0, 2.0)
    
    # Main content area
    if uploaded_file is not None or camera_image is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file if uploaded_file is not None else camera_image).convert('RGB')
            st.image(image, caption="Uploaded Crop Image", use_column_width=True)
            
            # Prediction with progress bar
            with st.spinner("Analyzing image..."):
                predicted_class, probability = predict_disease(image, model, classes, device, transform)
                time.sleep(1)  # Simulate processing time
                st.success("Analysis Complete!")
            
            # Display prediction results
            st.subheader("Prediction Results")
            st.write(f"**Detected Condition:** {predicted_class}")
            st.write(f"**Confidence:** {probability*100:.2f}%")
            
            # Environmental risk assessment
            risk_score = calculate_environmental_risk(temperature, humidity, rainfall, predicted_class)
            st.plotly_chart(create_risk_gauge(risk_score))
        
        with col2:
            # Disease information and management
            if predicted_class in DISEASE_INFO:
                disease_data = DISEASE_INFO[predicted_class]
                
                # Create tabs for different information categories
                tabs = st.tabs(["Symptoms", "Management", "Prevention", "Environmental Impact"])
                
                with tabs[0]:
                    st.subheader("Disease Symptoms")
                    for symptom in disease_data['symptoms']:
                        st.write(f"• {symptom}")
                
                with tabs[1]:
                    st.subheader("Management Strategies")
                    for strategy in disease_data['management']:
                        st.write(f"• {strategy}")
                    
                    if risk_score > 66:
                        st.warning("⚠️ High Risk - Immediate action recommended!")
                
                with tabs[2]:
                    st.subheader("Prevention Methods")
                    for method in disease_data['prevention']:
                        st.write(f"• {method}")
                
                with tabs[3]:
                    st.subheader("Environmental Conditions")
                    st.write(f"**Optimal Temperature:** {disease_data['environmental_conditions']['temperature']}")
                    st.write(f"**Required Humidity:** {disease_data['environmental_conditions']['humidity']}")
                    st.write(f"**Rainfall Conditions:** {disease_data['environmental_conditions']['rainfall']}")
                
                # Yield impact warning
                st.error(f"**Potential Yield Impact:** {disease_data['yield_impact']}")
            
            elif 'Healthy' in predicted_class:
                st.success("✅ Healthy Crop Detected!")
                if predicted_class in HEALTHY_CROP_CONDITIONS:
                    healthy_data = HEALTHY_CROP_CONDITIONS[predicted_class]
                    
                    st.subheader("Optimal Growing Conditions")
                    st.write(f"**Temperature:** {healthy_data['optimal_conditions']['temperature']}")
                    st.write(f"**Humidity:** {healthy_data['optimal_conditions']['humidity']}")
                    st.write(f"**Water Requirements:** {healthy_data['optimal_conditions']['rainfall']}")
                    
                    st.subheader("Maintenance Tips")
                    for tip in healthy_data['maintenance_tips']:
                        st.write(f"• {tip}")
    
    else:
        # Display welcome message and instructions
        st.info("👋 Welcome to the Crop Disease Detection System!")
        st.write("""
        ### How to use:
        1. Upload a clear image of your crop
        2. Set the environmental conditions
        3. Get instant disease detection and management advice
        
        ### Features:
        - Disease Detection
        - Risk Assessment
        - Management Strategies
        - Prevention Guidelines
        """)

if __name__ == "__main__":
    main()