# Flight Satisfaction Predictor - User Guide

## Getting Started

### Installation

1. Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

2. Extract the models from the notebook (optional, the app will create dummy models if none exist):

```bash
python extract_models.py
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## Using the Application

### Input Parameters

The sidebar on the left contains all the input parameters you can adjust:

1. **Passenger Information**:
   - Gender (Male/Female)
   - Customer Type (Loyal/Disloyal)
   - Type of Travel (Personal/Business)
   - Class (Eco/Eco Plus/Business)

2. **Flight Details**:
   - Age (0-85)
   - Flight Distance (0-4000)
   - Departure Delay (0-90 minutes)
   - Arrival Delay (0-90 minutes)

3. **Service Ratings** (all on a scale of 1-5):
   - Inflight WiFi Service
   - Departure/Arrival Time Convenience
   - Ease of Online Booking
   - Gate Location
   - Food and Drink
   - Online Boarding
   - Seat Comfort
   - Inflight Entertainment
   - On-board Service
   - Leg Room Service
   - Baggage Handling
   - Check-in Service
   - Inflight Service
   - Cleanliness

### Making Predictions

1. Adjust all the parameters according to the flight experience you want to analyze
2. Click the "Predict Satisfaction" button at the bottom of the sidebar
3. Wait for the models to process the data (this should only take a moment)

### Understanding the Results

After clicking the prediction button, you'll see:

1. **Overall Prediction**: Shows whether the passenger is likely to be satisfied or dissatisfied based on the majority vote of all models

2. **Model Predictions**: Individual predictions from each model:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   
   Each model shows its prediction and confidence level with a progress bar

3. **Model Agreement**: A pie chart showing how many models predicted "Satisfied" vs "Dissatisfied"

4. **Key Factors**: A chart showing which factors had the most influence on the prediction

## Tips for Best Results

- Try extreme values to see how they affect the prediction
- Compare business vs personal travel scenarios
- See how delays impact satisfaction across different service quality levels
- Experiment with different combinations of high and low ratings

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that the Streamlit server is running (look for the URL in the terminal)
3. If models fail to load, the application will create dummy models automatically
4. Restart the application if it becomes unresponsive

## Understanding the Models

The application uses three different machine learning models to predict passenger satisfaction:

1. **Logistic Regression**: A linear model that works well for binary classification problems
2. **Decision Tree**: A tree-based model that makes decisions based on feature thresholds
3. **Random Forest**: An ensemble of decision trees that typically provides better accuracy

The final prediction is determined by majority voting among these three models.