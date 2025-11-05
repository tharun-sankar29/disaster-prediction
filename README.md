# 🌋🌀 Disaster Prediction App

A Flask-based web application for predicting natural disasters, featuring hurricane and volcano prediction models with real-time data integration.

## ✨ Features

- **🌪️ Hurricane Prediction**
  - Real-time tracking and path prediction
  - Intensity forecasting
  - Historical data analysis

- **🌋 Volcano Prediction**
  - Volcanic Explosivity Index (VEI) prediction
  - Activity monitoring
  - Historical eruption analysis

- **🌐 Real-time Data**
  - Live data from USGS and other reliable sources
  - Automatic updates and alerts
  - Interactive visualization

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/tharun-sankar29/disaster-prediction.git
   cd disaster-prediction
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:5000`

## 🛠️ Project Structure

```
.
├── app.py                 # Main Flask application
├── models.py             # ML models and prediction logic
├── live_data_fetcher.py  # Real-time data fetching
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── templates/           # HTML templates
    ├── index.html       # Home page
    ├── hurricane.html   # Hurricane prediction interface
    └── volcano.html     # Volcano prediction interface
```

## 🌐 API Endpoints

- `GET /` - Home page
- `GET /hurricane` - Hurricane prediction interface
- `GET /volcano` - Volcano prediction interface
- `GET /api/volcanoes/active` - Get active volcanoes data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or suggestions, please open an issue or contact the maintainers.

1. Access the web interface at `http://localhost:5000`
2. Select the type of prediction (Hurricane or Volcano)
3. Enter the required parameters
4. View the prediction results

## Project Structure

- `app.py` - Main Flask application
- `models.py` - Machine learning models and prediction logic
- `live_data_fetcher.py` - Fetches real-time data from various APIs
- `templates/` - HTML templates for the web interface
  - `index.html` - Home page
  - `hurricane.html` - Hurricane prediction interface
  - `volcano.html` - Volcano prediction interface

## License

This project is licensed under the MIT License.
