import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time

# Define the ANN model class
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.PReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.model(x)

# Set page configuration
st.set_page_config(
    page_title="Flight Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideIn {
    from { transform: translateX(-50px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.fadeIn {
    animation: fadeIn 1.5s ease-in-out;
}

.slideIn {
    animation: slideIn 1s ease-in-out;
}

.stProgress .st-bo {
    background-color: #4CAF50;
}

.prediction-box {
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    animation: fadeIn 1.5s ease-in-out;
}

.satisfied {
    background-color: rgba(212, 237, 218, 0.8);
    border: 1px solid rgba(195, 230, 203, 0.8);
    backdrop-filter: blur(10px);
}

.dissatisfied {
    background-color: rgba(248, 215, 218, 0.8);
    border: 1px solid rgba(245, 198, 203, 0.8);
    backdrop-filter: blur(10px);
}

.model-comparison {
    padding: 15px;
    border-radius: 5px;
    margin-top: 10px;
    background-color: rgba(233, 236, 239, 0.8);
    backdrop-filter: blur(10px);
    animation: slideIn 1s ease-in-out;
}

.header-container {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 20px;
}

.header-text {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E88E5;
    text-align: center;
    animation: fadeIn 2s ease-in-out;
}

.subheader {
    font-size: 1.5rem;
    color: #424242;
    text-align: center;
    margin-bottom: 30px;
    animation: fadeIn 2.5s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Header with animation
st.markdown('<div class="header-container"><h1 class="header-text">✈️ Flight Satisfaction Predictor</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Enter flight details to predict passenger satisfaction</p>', unsafe_allow_html=True)

# Function to train models if not already saved
def train_models():
    # This is a placeholder function that would normally load your dataset and train models
    # For this example, we'll create simple models based on the notebook
    
    # Create dummy data similar to your notebook
    np.random.seed(42)
    X_dummy = np.random.rand(1000, 22)  # 22 features as seen in the notebook
    y_dummy = np.random.randint(0, 2, 1000)  # Binary classification
    
    # Train models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dummy)
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_scaled, y_dummy)
    
    dt = DecisionTreeClassifier()
    dt.fit(X_dummy, y_dummy)
    
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_dummy, y_dummy)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(log_reg, 'models/logistic_regression.pkl')
    joblib.dump(dt, 'models/decision_tree.pkl')
    joblib.dump(rf, 'models/random_forest.pkl')
    
    return scaler, log_reg, dt, rf

# Load or train models
def load_models():
    try:
        scaler = joblib.load('models/scaler.pkl')
        log_reg = joblib.load('models/logistic_regression.pkl')
        dt = joblib.load('models/decision_tree.pkl')
        rf = joblib.load('models/random_forest.pkl')
        xgb_model = joblib.load('models/xgboost.pkl')
    except:
        st.info("Training models for first use...")
        scaler, log_reg, dt, rf = train_models()
        xgb_model = None  # XGBoost will be None if not found
    
    return scaler, log_reg, dt, rf, xgb_model

# Load ANN model
ann_model = ANN(input_size=22)
# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ann_model.load_state_dict(torch.load('models/ann_model.pth', map_location=device))
ann_model = ann_model.to(device)
ann_model.eval()

# Create sidebar for inputs
st.sidebar.markdown("## Passenger Information")

# Categorical features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.sidebar.selectbox("Type of Travel", ["Personal Travel", "Business travel"])
travel_class = st.sidebar.selectbox("Class", ["Eco", "Eco Plus", "Business"])

# Numerical features
st.sidebar.markdown("## Flight Details")
age = st.sidebar.slider("Age", 0, 85, 30)
flight_distance = st.sidebar.slider("Flight Distance", 0, 4000, 1000)
departure_delay = st.sidebar.slider("Departure Delay (minutes)", 0, 90, 15)
arrival_delay = st.sidebar.slider("Arrival Delay (minutes)", 0, 90, 15)

# Service ratings
st.sidebar.markdown("## Service Ratings (1-5)")
wifi = st.sidebar.slider("Inflight WiFi Service", 1, 5, 3)
time_convenient = st.sidebar.slider("Departure/Arrival Time Convenience", 1, 5, 3)
online_booking = st.sidebar.slider("Ease of Online Booking", 1, 5, 3)
gate_location = st.sidebar.slider("Gate Location", 1, 5, 3)
food_drink = st.sidebar.slider("Food and Drink", 1, 5, 3)
online_boarding = st.sidebar.slider("Online Boarding", 1, 5, 3)
seat_comfort = st.sidebar.slider("Seat Comfort", 1, 5, 3)
entertainment = st.sidebar.slider("Inflight Entertainment", 1, 5, 3)
onboard_service = st.sidebar.slider("On-board Service", 1, 5, 3)
leg_room = st.sidebar.slider("Leg Room Service", 1, 5, 3)
baggage_handling = st.sidebar.slider("Baggage Handling", 1, 5, 3)
checkin_service = st.sidebar.slider("Check-in Service", 1, 5, 3)
inflight_service = st.sidebar.slider("Inflight Service", 1, 5, 3)
cleanliness = st.sidebar.slider("Cleanliness", 1, 5, 3)

# Encode categorical features
def encode_features(gender, customer_type, travel_type, travel_class):
    # Gender encoding: Male = 1, Female = 0
    gender_encoded = 1 if gender == "Male" else 0
    
    # Customer Type encoding: Loyal = 1, Disloyal = 0
    customer_type_encoded = 1 if customer_type == "Loyal Customer" else 0
    
    # Travel Type encoding: Business = 1, Personal = 0
    travel_type_encoded = 1 if travel_type == "Business travel" else 0
    
    # Class encoding: Business = 2, Eco Plus = 1, Eco = 0
    if travel_class == "Business":
        class_encoded = 2
    elif travel_class == "Eco Plus":
        class_encoded = 1
    else:
        class_encoded = 0
    
    return gender_encoded, customer_type_encoded, travel_type_encoded, class_encoded

# Predict function
def predict(input_data, scaler, log_reg, dt, rf, ann_model, xgb_model=None):
    # Scale the data for logistic regression
    input_scaled = scaler.transform([input_data])
    
    # Set device for PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.FloatTensor(input_scaled).to(device)
    
    # Make predictions
    log_reg_pred = log_reg.predict_proba([input_data])[0]
    dt_pred = dt.predict_proba([input_data])[0]
    rf_pred = rf.predict_proba([input_data])[0]
    
    # ANN prediction
    ann_model.eval()
    with torch.no_grad():
        ann_outputs = ann_model(input_tensor)
        ann_probs = torch.softmax(ann_outputs, dim=1)
        ann_pred = ann_probs[0].cpu().numpy()
    
    # XGBoost prediction (if model is available)
    if xgb_model is not None:
        xgb_pred = xgb_model.predict_proba([input_data])[0]
        xgb_class = xgb_model.predict([input_data])[0]
    else:
        xgb_pred = np.array([0.5, 0.5])  # Default if model not available
        xgb_class = 0
    
    # Get the final predictions (1 = satisfied, 0 = dissatisfied)
    log_reg_class = log_reg.predict([input_data])[0]
    dt_class = dt.predict([input_data])[0]
    rf_class = rf.predict([input_data])[0]
    ann_class = ann_pred.argmax()
    
    # Ensemble prediction (majority voting)
    ensemble_class = 1 if (log_reg_class + dt_class + rf_class + ann_class + xgb_class) >= 3 else 0
    
    return {
        'logistic_regression': {'class': log_reg_class, 'proba': log_reg_pred},
        'decision_tree': {'class': dt_class, 'proba': dt_pred},
        'random_forest': {'class': rf_class, 'proba': rf_pred},
        'ann': {'class': ann_class, 'proba': ann_pred},
        'xgboost': {'class': xgb_class, 'proba': xgb_pred},
        'ensemble': {'class': ensemble_class}
    }

# Main prediction button
if st.sidebar.button("Predict Satisfaction", key="predict_button"):
    # Show a spinner while processing
    with st.spinner("Processing your flight data..."):
        # Encode categorical features
        gender_encoded, customer_type_encoded, travel_type_encoded, class_encoded = encode_features(
            gender, customer_type, travel_type, travel_class
        )
        
        # Create input data array
        input_data = [
            gender_encoded, customer_type_encoded, age, travel_type_encoded, 
            class_encoded, flight_distance, wifi, time_convenient, online_booking,
            gate_location, food_drink, online_boarding, seat_comfort, entertainment,
            onboard_service, leg_room, baggage_handling, checkin_service,
            inflight_service, cleanliness, departure_delay, arrival_delay
        ]
        
        # Load models
        scaler, log_reg, dt, rf, xgb_model = load_models()
        
        # Make predictions
        predictions = predict(input_data, scaler, log_reg, dt, rf, ann_model, xgb_model)
        
        # Add a small delay for animation effect
        time.sleep(1)
    
    # Display results with animations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Overall prediction result with animation
        ensemble_result = "satisfied" if predictions['ensemble']['class'] == 1 else "neutral or dissatisfied"
        result_class = "satisfied" if predictions['ensemble']['class'] == 1 else "dissatisfied"
        
        st.markdown(f"""
        <div class="prediction-box {result_class}">
            <h2>Overall Prediction: {ensemble_result.title()}</h2>
            <p>Based on the majority vote of all models</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model comparison
        st.markdown("<h3 class='slideIn'>Model Predictions</h3>", unsafe_allow_html=True)
        
        # Create a progress bar for each model's confidence
        for model_name, result in predictions.items():
            if model_name != 'ensemble':
                prediction_class = result['class']
                prediction_label = "Satisfied" if prediction_class == 1 else "Dissatisfied"
                confidence = result['proba'][prediction_class] * 100
                
                st.markdown(f"""
                <div class="model-comparison">
                    <h4>{model_name.replace('_', ' ').title()}</h4>
                    <p>Prediction: <strong>{prediction_label}</strong> with {confidence:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Animated progress bar
                st.progress(confidence/100)
    
    with col2:
        # Create a pie chart for model agreement
        fig, ax = plt.subplots(figsize=(5, 5))
        model_results = [predictions[m]['class'] for m in ['logistic_regression', 'decision_tree', 'random_forest']]
        satisfied_count = sum(model_results)
        dissatisfied_count = len(model_results) - satisfied_count
        
        labels = ['Satisfied', 'Dissatisfied']
        sizes = [satisfied_count, dissatisfied_count]
        colors = ['#4CAF50', '#F44336']
        explode = (0.1, 0) if satisfied_count > dissatisfied_count else (0, 0.1)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Model Agreement')
        st.pyplot(fig)
        
        # Feature importance visualization (simplified)
        st.markdown("<h4 class='slideIn'>Key Factors</h4>", unsafe_allow_html=True)
        
        # Create a simplified feature importance chart
        # In a real app, you would extract this from the models
        features = ['Seat Comfort', 'Cleanliness', 'Food & Drink', 'Wifi', 'Delays']
        importance = [0.25, 0.20, 0.18, 0.15, 0.22]  # Dummy values
        
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(features, importance, color='#1E88E5')
        ax.set_xlim(0, 0.3)
        ax.set_xlabel('Relative Importance')
        plt.tight_layout()
        st.pyplot(fig)

# Display instructions if no prediction has been made
else:
    st.markdown("""
    <div style="text-align: center; padding: 20px; animation: fadeIn 2s ease-in-out;">
        <img src="https://cdn-icons-png.flaticon.com/512/6245/6245301.png" width="150">
        <h3>How to use this predictor:</h3>
        <ol style="text-align: left; display: inline-block;">
            <li>Enter passenger information in the sidebar</li>
            <li>Adjust flight details and service ratings</li>
            <li>Click the "Predict Satisfaction" button</li>
            <li>View predictions from multiple models</li>
            <li>Compare model results and key factors</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample visualization
    st.markdown("<h3 class='slideIn'>Sample Satisfaction Factors</h3>", unsafe_allow_html=True)
    
    # Create sample data
    factors = ['Seat Comfort', 'Cleanliness', 'Food & Drink', 'Inflight Entertainment', 
               'Staff Service', 'Baggage Handling', 'Departure/Arrival Time']
    satisfied_scores = [4.2, 4.5, 3.8, 4.0, 4.3, 3.9, 3.5]
    dissatisfied_scores = [2.1, 2.3, 1.9, 2.5, 2.0, 2.2, 1.8]
    
    # Create a comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(factors))
    width = 0.35
    
    ax.bar(x - width/2, satisfied_scores, width, label='Satisfied Passengers', color='#4CAF50')
    ax.bar(x + width/2, dissatisfied_scores, width, label='Dissatisfied Passengers', color='#F44336')
    
    ax.set_xticks(x)
    ax.set_xticklabels(factors, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 5)
    ax.set_ylabel('Average Rating')
    ax.set_title('Comparison of Ratings: Satisfied vs. Dissatisfied Passengers')
    plt.tight_layout()
    
    st.pyplot(fig)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #e0e0e0;">
    <p>This application predicts passenger satisfaction based on flight experience factors.</p>
    <p>Models: Logistic Regression, Decision Tree, Random Forest, and Neural Network</p>
</div>
""", unsafe_allow_html=True)