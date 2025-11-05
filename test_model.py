import joblib
import pandas as pd
import numpy as np

def test_model():
    try:
        # Load the model and features
        model = joblib.load("models/hurricane_model.pkl")
        features = joblib.load("models/hurricane_features.pkl")
        
        print(f"Model features: {features}")
        
        # Test with some sample data
        test_data = [
            [1013, 25.0, -70.0],  # Typical values
            [950, 20.0, -80.0],   # Stronger storm
            [980, 30.0, -60.0]     # Another test case
        ]
        
        # Create DataFrame with correct feature order
        test_df = pd.DataFrame(test_data, columns=features)
        
        # Make predictions
        predictions = model.predict(test_df)
        
        # Print results
        for i, (data, pred) in enumerate(zip(test_data, predictions)):
            print(f"\nTest {i+1}:")
            print(f"  Input: Pressure={data[0]} mbar, Lat={data[1]}, Lon={data[2]}")
            print(f"  Predicted Wind Speed: {pred:.1f} kts")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_model()
