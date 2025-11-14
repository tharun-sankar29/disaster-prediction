import pandas as pd
import joblib
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from typing import List, Union, Optional, Dict, Any

# -------------------------
# SETUP & HELPER FUNCTIONS
# -------------------------

def save_artifacts(model: Union[Pipeline, RandomForestRegressor], 
                  features: List[str], 
                  prefix: str) -> None:
    """Save trained model and feature list to disk."""
    joblib.dump(model, f"models/{prefix}_model.pkl")
    joblib.dump(features, f"models/{prefix}_features.pkl")

def clean_lat_lon(value: Union[str, float]) -> float:
    """Converts latitude/longitude string formats (e.g., '28.0N') to decimal floats."""
    val = str(value).strip()
    if 'N' in val:
        return float(val.replace('N', ''))
    elif 'S' in val:
        return -float(val.replace('S', ''))
    elif 'W' in val:
        return -float(val.replace('W', ''))
    elif 'E' in val:
        return float(val.replace('E', ''))
    else:
        # Assumes decimal format if no cardinal direction is present
        return float(val)

# -------------------------
# TRAINING FUNCTIONS
# -------------------------

def train_hurricane_model(atlantic_file: str = 'atlantic.csv', pacific_file: str = 'pacific.csv'):
    """
    Trains the hurricane Maximum Wind Speed prediction model.
    Input Features: Minimum Pressure, Latitude, Longitude.
    Target: Maximum Wind.
    """
    atlantic_df = pd.read_csv(atlantic_file)
    pacific_df = pd.read_csv(pacific_file)
    combined_df = pd.concat([atlantic_df, pacific_df], ignore_index=True)
    combined_df.columns = combined_df.columns.str.strip()
    combined_df = combined_df.replace(-999, np.nan)
    combined_df = combined_df.dropna(subset=['Minimum Pressure', 'Maximum Wind', 'Latitude', 'Longitude'])

    X = combined_df[['Minimum Pressure', 'Latitude', 'Longitude']].copy()
    y = combined_df['Maximum Wind']

    X.loc[:, 'Latitude'] = X['Latitude'].apply(clean_lat_lon)
    X.loc[:, 'Longitude'] = X['Longitude'].apply(clean_lat_lon)

    features = list(X.columns)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    save_artifacts(model, features, "hurricane")
    print("✅ Hurricane model trained and saved.")

def train_volcano_model(events_file: str = 'datasets/volcano-events.csv'):
    """
    Trains the Volcanic Explosivity Index (VEI) prediction model pipeline.
    Uses RandomForestRegressor as a robust general-purpose regressor.
    """
    data = pd.read_csv(events_file)
    data = data.dropna(how="all").reset_index(drop=True)
    data = data[data["VEI"].notna()].reset_index(drop=True)
    
    features = ["Year", "Month", "Day", "Latitude", "Longitude", "Elevation (m)", "Country", "Type", "Agent"]
    target = "VEI"
    available_features = [f for f in features if f in data.columns]
    X = data[available_features]
    y = data[target].astype(float)
    
    numeric_features = [c for c in available_features if X[c].dtype != "object"]
    categorical_features = [c for c in available_features if c not in numeric_features]
    
    # Pre-processing for categorical columns to ensure consistent string format
    for c in categorical_features:
        X.loc[:, c] = X[c].fillna('').astype(str).str.strip()
        
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    model = Pipeline(steps=[
        ("preprocessor", preprocessor), 
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)
    save_artifacts(model, available_features, "volcano") # Saving the Pipeline and feature list
    print("✅ Volcano model pipeline trained and saved.")

# -------------------------
# CORE PREDICTION FUNCTIONS (Load and Predict)
# -------------------------

def predict_hurricane(min_pressure: float, latitude: float, longitude: float) -> float:
    """Predicts hurricane Maximum Wind Speed using raw numeric inputs."""
    import logging
    import numpy as np
    logger = logging.getLogger(__name__)
    
    try:
        # Log the function call with all parameters
        logger.info("=== Starting predict_hurricane ===")
        logger.info(f"Input - Pressure: {min_pressure:.2f} mbar, "
                  f"Lat: {latitude:.4f}, Lon: {longitude:.4f}")
        
        # Load model and features
        logger.info("Loading model and features...")
        model = joblib.load("models/hurricane_model.pkl")
        features = joblib.load("models/hurricane_features.pkl")
        logger.info(f"Model features: {features}")
        
        # Ensure correct data types
        min_pressure = float(min_pressure)
        latitude = float(latitude)
        longitude = float(longitude)
        
        # Create input data in the exact format expected by the model
        input_data = {
            'Minimum Pressure': [min_pressure],
            'Latitude': [abs(latitude)],  # Use absolute value as in training
            'Longitude': [abs(longitude) * -1 if longitude < 0 else longitude]
        }
        
        # Create DataFrame with exact feature names and order
        input_df = pd.DataFrame(input_data, columns=features)
        
        # Log the prepared input data
        logger.info("Prepared input data:")
        logger.info(f"  - Minimum Pressure: {input_df['Minimum Pressure'].values[0]:.2f} mbar")
        logger.info(f"  - Latitude: {input_df['Latitude'].values[0]:.4f}")
        logger.info(f"  - Longitude: {input_df['Longitude'].values[0]:.4f}")
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(input_df)[0]
        logger.info(f"Raw model prediction: {prediction}")
        
        # Ensure the prediction is a valid number
        if not np.isfinite(prediction):
            logger.warning(f"Invalid prediction: {prediction}. Using 0.")
            return 0.0
            
        # Round to nearest integer and ensure it's within reasonable bounds
        prediction = max(0, min(200, round(prediction)))
        logger.info(f"Final predicted wind speed: {prediction} kts")
        
        # Log feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            logger.info("Feature importances:")
            for name, importance in zip(features, importances):
                logger.info(f"  - {name}: {importance:.4f}")
        
        logger.info("=== End predict_hurricane ===")
        return float(prediction)
        
    except Exception as e:
        logger.error(f"Error in predict_hurricane: {str(e)}", exc_info=True)
        # Return a default value that will be noticeable if something goes wrong
        return 99.9

def predict_volcano(year: int, month: int, day: int, latitude: float, longitude: float, 
                    elevation_m: float, country: str, type_str: str, agent: str) -> dict:
    """
    Predicts VEI using the 9 required features from the dataset.
    
    Args:
        year: Year of the event
        month: Month of the event (1-12)
        day: Day of the event
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        elevation_m: Elevation in meters
        country: Country where the volcano is located
        type_str: Type of volcano (e.g., 'Stratovolcano', 'Shield volcano')
        agent: Type of activity (e.g., 'T' for tephra, 'P' for pyroclastic flows)
        
    Returns:
        dict: Contains prediction and available context from the dataset
    """
    pipeline = joblib.load("models/volcano_model.pkl")
    features = joblib.load("models/volcano_features.pkl")
    
    # Prepare input data using only the available features from the dataset
    input_data = {
        'Year': year, 
        'Month': month, 
        'Day': day, 
        'Latitude': latitude, 
        'Longitude': longitude, 
        'Elevation (m)': elevation_m, 
        'Country': country, 
        'Type': type_str, 
        'Agent': agent
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Ensure categorical inputs are string objects for the pipeline
    for col in input_df.columns:
        if col not in ['Year', 'Month', 'Day', 'Latitude', 'Longitude', 'Elevation (m)']:
            input_df.loc[:, col] = input_df[col].astype(str).str.strip()

    # Make prediction
    vei_prediction = round(pipeline.predict(input_df)[0], 1)
    
    # Return prediction with context
    return {
        'vei': vei_prediction,
        'volcano_type': type_str,
        'elevation_m': elevation_m,
        'activity_type': agent,
        'location': {
            'country': country,
            'coordinates': {
                'latitude': latitude,
                'longitude': longitude
            }
        },
        'date': f"{year}-{month:02d}-{day:02d}"
    }

# -------------------------
# LIVE DATA INTEGRATION FUNCTIONS
# -------------------------

def setup_and_train_all_models() -> None:
    """Train and save all models to disk."""
    logger = logging.getLogger(__name__)
    try:
        logger.info("Starting model training...")
        train_hurricane_model()
        train_volcano_model()
        logger.info("All models trained and saved successfully!")
    except Exception as e:
        logger.error(f"Error training models: {e}", exc_info=True)
        raise

def predict_live_hurricane(storm_id: str) -> dict:
    """
    Fetch live hurricane data and make a prediction.
    
    Args:
        storm_id: The storm identifier (e.g., 'AL022025')
        
    Returns:
        dict: Contains prediction results and metadata
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"=== Starting prediction for storm: {storm_id} ===")
        from live_data_fetcher import LiveDataFetcher
        
        # Initialize the data fetcher
        fetcher = LiveDataFetcher()
        
        # Get live data
        logger.info(f"Fetching data for storm: {storm_id}")
        live_data = fetcher.get_hurricane_data(storm_id)
        logger.debug(f"Raw live data: {live_data}")
        
        # Validate and extract min_pressure
        if 'min_pressure_mbar' not in live_data and 'min_pressure' not in live_data:
            raise ValueError("No minimum pressure data found in live data")
            
        min_pressure = float(live_data.get('min_pressure_mbar') or live_data.get('min_pressure'))
        logger.info(f"Using min_pressure: {min_pressure} mbar")
        
        # Get and validate coordinates
        if 'latitude' not in live_data or 'longitude' not in live_data:
            raise ValueError("Latitude and longitude are required in live data")
            
        latitude = float(live_data['latitude'])
        longitude = float(live_data['longitude'])
        logger.info(f"Using coordinates: lat={latitude}, lon={longitude}")
        
        # Log the input values being sent to predict_hurricane
        logger.info(f"Calling predict_hurricane with: pressure={min_pressure}, lat={latitude}, lon={longitude}")
        
        # Call predict_hurricane with the validated values
        predicted_wind_speed = predict_hurricane(
            min_pressure=min_pressure,
            latitude=latitude,
            longitude=longitude
        )
        
        logger.info(f"Predicted wind speed: {predicted_wind_speed} kts")
        
        # Prepare the result with detailed information
        result = {
            'prediction': predicted_wind_speed,
            'data': {
                'storm_name': live_data.get('name', f"Storm {storm_id}"),
                'min_pressure': min_pressure,
                'latitude': latitude,
                'longitude': longitude,
                'lat_str': live_data.get('lat_str', f"{abs(latitude):.1f}°{'N' if latitude >= 0 else 'S'}"),
                'lon_str': live_data.get('lon_str', f"{abs(longitude):.1f}°{'E' if longitude >= 0 else 'W'}"),
                'data_source': live_data.get('data_source', 'NHC ATCF'),
                'timestamp': live_data.get('timestamp', datetime.utcnow().isoformat()),
                'storm_id': storm_id,
                'raw_data': live_data.get('raw_data', 'N/A')
            },
            'status': 'success'
        }
        
        logger.info(f"Successfully completed prediction for storm: {storm_id}")
        return result
        
    except Exception as e:
        error_msg = f"Error in predict_live_hurricane for storm {storm_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'status': 'error',
            'error': error_msg,
            'prediction': None,
            'data': {
                'storm_id': storm_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

def predict_live_volcano(volcano_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch live volcano data and make a prediction.
    
    Args:
        volcano_id: Optional volcano identifier. If None, gets a random active volcano.
        
    Returns:
        dict: Contains prediction results and metadata with status
    """
    logger = logging.getLogger(__name__)
    try:
        from live_data_fetcher import LiveDataFetcher
        
        logger.info(f"Fetching data for volcano: {volcano_id or 'random'}")
        fetcher = LiveDataFetcher()
        live_data = fetcher.get_volcano_data(volcano_id)
        
        prediction = predict_volcano(
            year=live_data['Year'],
            month=live_data['Month'],
            day=live_data['Day'],
            latitude=live_data['Latitude'],
            longitude=live_data['Longitude'],
            elevation_m=live_data['Elevation (m)'],
            country=live_data['Country'],
            type_str=live_data['Type'],
            agent=live_data['Agent']
        )
        
        result = {
            'prediction': prediction,
            'volcano': live_data.get('Volcano', 'Unknown'),
            'data': {k: v for k, v in live_data.items() 
                    if k not in ['recent_earthquakes', 'data_source', 'timestamp']},
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Successfully predicted for volcano: {result['volcano']}")
        return result
        
    except Exception as e:
        error_msg = f"Error in predict_live_volcano: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'status': 'error',
            'error': error_msg,
            'timestamp': datetime.utcnow().isoformat()
        }
