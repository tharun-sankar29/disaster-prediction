from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from models import predict_live_hurricane, predict_live_volcano
from live_data_fetcher import LiveDataFetcher
import logging
import requests
from datetime import datetime, timezone

# Initialize data fetcher
data_fetcher = LiveDataFetcher()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ------------------ ROUTES ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hurricane')
def hurricane_page():
    return render_template('hurricane.html')

@app.route('/volcano')
def volcano_page():
    return render_template('volcano.html')

# Redirect any other disaster routes to the home page
@app.route('/flood')
@app.route('/landslide')
@app.route('/forestfire')
@app.route('/tsunami')
def redirect_to_home():
    return redirect(url_for('index'))

# ------------------ HURRICANE ROUTES ------------------

@app.route('/api/hurricane/live/<storm_id>', methods=['GET'])
def get_live_hurricane_data(storm_id):
    """Fetch live hurricane data by storm ID."""
    try:
        logger.info(f"Fetching live data for storm: {storm_id}")
        data = data_fetcher.get_hurricane_data(storm_id)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching live hurricane data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict-hurricane', methods=['POST'])
def hurricane_prediction():
    """Handle manual prediction with user-provided data."""
    try:
        if request.is_json:
            data = request.get_json(force=True)
        else:
            data = request.form.to_dict()
            for key in data:
                try:
                    data[key] = float(data[key])
                except (ValueError, TypeError):
                    pass

        min_pressure = data.get('min_pressure') or data.get('Minimum Pressure')
        latitude = data.get('latitude') or data.get('Latitude')
        longitude = data.get('longitude') or data.get('Longitude')

        if None in (min_pressure, latitude, longitude):
            return jsonify({
                'status': 'error',
                'error': f'Missing required parameters. Got: {data}'
            }), 400

        from models import predict_hurricane
        prediction = predict_hurricane(float(min_pressure), float(latitude), float(longitude))

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'input': {
                'min_pressure': min_pressure,
                'latitude': latitude,
                'longitude': longitude
            }
        })

    except Exception as e:
        logger.error(f"Error in hurricane prediction: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 400


@app.route('/api/hurricane/predict/live', methods=['POST'])
def live_hurricane_prediction():
    """
    Handle prediction using live data from NHC.
    Expected JSON: { "storm_id": "AL022025" }
    """
    start_time = datetime.now(timezone.utc)
    request_id = f"req_{start_time.strftime('%Y%m%d%H%M%S')}"
    
    try:
        logger.info(f"[{request_id}] Received prediction request")

        # Parse and validate JSON
        data = request.get_json()
        if not data or not isinstance(data, dict):
            error_msg = "Invalid request: Expected JSON data"
            raise ValueError(error_msg)
        
        storm_id = data.get('storm_id')
        if not storm_id or not isinstance(storm_id, str):
            raise ValueError("Missing or invalid 'storm_id' in request")

        logger.info(f"[{request_id}] Processing prediction for storm: {storm_id}")

        # Run prediction
        result = predict_live_hurricane(storm_id)
        logger.debug(f"[{request_id}] Raw prediction result: {result}")

        if not result or 'status' not in result:
            raise ValueError("Invalid response from prediction service")

        if result.get('status') != 'success':
            error_msg = result.get('error', 'Prediction failed')
            raise ValueError(error_msg)

        # Prepare response
        response_data = {
            'status': 'success',
            'request_id': request_id,
            'storm_id': storm_id,
            'prediction': result['prediction'],
            'metadata': {
                'data_source': result.get('data', {}).get('data_source', 'NHC ATCF'),
                'timestamp': result.get('data', {}).get('timestamp', datetime.now(timezone.utc).isoformat()),
                'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            },
            'input_parameters': {
                'min_pressure': result.get('data', {}).get('min_pressure'),
                'latitude': result.get('data', {}).get('latitude'),
                'longitude': result.get('data', {}).get('longitude'),
                'lat_str': result.get('data', {}).get('lat_str'),
                'lon_str': result.get('data', {}).get('lon_str'),
                'storm_name': result.get('data', {}).get('storm_name', f"Storm {storm_id}")
            },
            'raw_data': result.get('data', {})
        }

        logger.info(f"[{request_id}] ✅ Successfully completed prediction for storm {storm_id}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'request_id': request_id,
            'storm_id': storm_id if 'storm_id' in locals() else 'unknown',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        }), 500


@app.route('/api/storms/active/names', methods=['GET'])
def get_active_storm_names():
    """Get list of active storm names and IDs only (lightweight)."""
    try:
        logger.info("Fetching active storm names (lightweight)")
        storms = data_fetcher.get_active_storm_names()
        return jsonify({
            'status': 'success',
            'data': storms,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching active storm names: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/storms/active', methods=['GET'])
def get_active_storms():
    """Get list of currently active tropical storms/hurricanes with full data."""
    try:
        logger.info("Fetching active storms with full data")
        storms = data_fetcher.get_active_storms()
        return jsonify({
            'status': 'success',
            'data': storms,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching active storms: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ------------------ VOLCANO ROUTES ------------------

@app.route('/api/volcanoes/active', methods=['GET'])
def get_active_volcanoes():
    """Get list of active volcanoes from USGS Volcano API."""
    try:
        logger.info("Fetching active volcanoes from USGS API")
        
        # USGS Volcano API endpoint for all volcanoes
        usgs_url = "https://volcanoes.usgs.gov/vsc/api/volcanoApi/volcanoesGVP"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        # Fetch volcano data
        response = requests.get(usgs_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        volcanoes = response.json()
        
        if not isinstance(volcanoes, list):
            raise ValueError("Invalid data format from USGS API")
        
        # Format the response
        active_volcanoes = []
        for volcano in sorted(volcanoes, key=lambda x: x.get('vName', '').lower()):  # Sort alphabetically by name
            try:
                # Get the volcano status from the observatory code
                obs_code = volcano.get('obsAbbr', '')
                status = 'Active' if obs_code else 'Inactive'
                
                # Get the volcano type from the name or use a default
                volcano_type = 'Volcano'
                if 'caldera' in volcano.get('vName', '').lower():
                    volcano_type = 'Caldera'
                elif 'shield' in volcano.get('vName', '').lower():
                    volcano_type = 'Shield Volcano'
                
                active_volcanoes.append({
                    'id': volcano.get('vnum', '').lower(),
                    'name': volcano.get('vName', 'Unnamed Volcano').strip(),
                    'country': volcano.get('country', 'Unknown'),
                    'region': volcano.get('subregion', 'Unknown'),
                    'latitude': float(volcano.get('latitude', 0)),
                    'longitude': float(volcano.get('longitude', 0)),
                    'elevation': float(volcano.get('elevation_m', 0)),
                    'type': volcano_type,
                    'status': status,
                    'observatory': obs_code,
                    'details': f"Volcano ID: {volcano.get('vnum', 'N/A')}",
                    'webpage': volcano.get('webpage', '')
                })
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing volcano data: {e}")
                continue
        
        return jsonify({
            'status': 'success',
            'count': len(active_volcanoes),
            'data': active_volcanoes,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_source': 'USGS Volcano Hazards Program',
            'note': 'Showing up to 50 active volcanoes',
            'total_volcanoes': len(volcanoes)
        })
        
    except requests.RequestException as e:
        logger.error(f"Error fetching active volcanoes: {str(e)}")
        # Fallback to sample data if the API fails
        current_time = datetime.now(timezone.utc)
        return jsonify({
            'status': 'success',  # Still return success but with sample data
            'count': 1,
            'data': [{
                'id': 'sample-1',
                'name': 'Kīlauea',
                'country': 'United States',
                'region': 'Hawaiian Islands',
                'latitude': 19.421,
                'longitude': -155.288,
                'elevation': 1247,
                'type': 'Shield Volcano',
                'status': 'Erupting',
                'activity': 'Lava lake in Halemaʻumaʻu crater',
                'details': 'Sample data - could not fetch live data',
                'note': 'Using sample data due to API error'
            }],
            'timestamp': current_time.isoformat(),
            'data_source': 'Sample Data',
            'warning': 'Using sample data due to API error',
            'error': str(e)
        })
    except Exception as e:
        logger.error(f"Unexpected error in get_active_volcanoes: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500
        
    except requests.RequestException as e:
        logger.error(f"Error fetching active volcanoes: {str(e)}")
        # Fallback to sample data if the API fails
        return jsonify({
            'status': 'success',  # Still return success but with sample data
            'count': 1,
            'data': [{
                'id': 'sample-1',
                'name': 'Sample Volcano',
                'country': 'Sample Country',
                'region': 'Sample Region',
                'latitude': 0,
                'longitude': 0,
                'elevation': 0,
                'type': 'Stratovolcano',
                'last_eruption': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'status': 'Erupting',
                'vei': 3,
                'activity': 'Strombolian activity',
                'note': 'Sample data - could not fetch live data'
            }],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_source': 'Sample Data',
            'warning': 'Using sample data due to API error',
            'error': str(e)
        })
    except Exception as e:
        logger.error(f"Unexpected error in get_active_volcanoes: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/api/volcano/live/<volcano_id>', methods=['GET'])
def get_live_volcano_data(volcano_id):
    """Fetch live volcano data by volcano ID."""
    try:
        logger.info(f"Fetching live data for volcano: {volcano_id}")
        data = data_fetcher.get_volcano_data(volcano_id)
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching live volcano data: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/predict-volcano', methods=['POST'])
def volcano_prediction():
    """Handle manual volcano prediction with user-provided data."""
    try:
        data = request.get_json(force=True)
        required_fields = [
            'Year', 'Month', 'Day', 'Latitude', 'Longitude',
            'Elevation (m)', 'Country', 'Type', 'Agent'
        ]

        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

        from models import predict_volcano
        prediction = predict_volcano(
            data['Year'], data['Month'], data['Day'],
            data['Latitude'], data['Longitude'], data['Elevation (m)'],
            data['Country'], data['Type'], data['Agent']
        )
        return jsonify({'prediction': prediction})

    except Exception as e:
        logger.error(f"Error in volcano prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/predict-volcano-live', methods=['POST'])
def live_volcano_prediction():
    """Handle prediction using live volcano data."""
    try:
        data = request.get_json(force=True)
        volcano_id = data.get('volcano_id')
        result = predict_live_volcano(volcano_id)

        if result['status'] != 'success':
            return jsonify({'error': result.get('error', 'Unknown error')}), 400

        return jsonify({
            'prediction': result['prediction'],
            'data': result['data'],
            'data_source': 'USGS/Smithsonian GVP Live Data',
            'volcano': result['volcano'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error in live volcano prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400


# ------------------ MAIN ENTRY ------------------

if __name__ == '__main__':
    app.run(debug=True)
