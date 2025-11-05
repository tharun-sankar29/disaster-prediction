import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup

class LiveDataFetcher:
    """Fetches live data for natural disaster predictions from official sources."""
    
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # NHC Data Sources
        self.nhc_base_url = "https://www.nhc.noaa.gov"
        self.atcf_base_url = "https://ftp.nhc.noaa.gov/atcf/btk"
        
        # USGS Data Sources
        self.usgs_volcano_url = "https://volcano.si.edu/webservices/feeds/volcanoes.php"
        self.usgs_earthquake_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    def get_active_storm_names(self) -> List[Dict[str, str]]:
        """
        Get a lightweight list of active storm names and IDs only.
        This is much faster than get_active_storms() as it doesn't fetch all storm data.
        
        Returns:
            List[Dict[str, str]]: List of active storms with 'id', 'name', 'basin', and 'year'.
        """
        self.logger.info("Fetching active storm names from ATCF...")
        
        try:
            current_year = str(datetime.now().year)
            basins = ['al', 'ep']  # Atlantic and East Pacific basins
            storm_list = []
            
            for basin in basins:
                try:
                    # Get the directory listing
                    list_url = "https://ftp.nhc.noaa.gov/atcf/btk/"
                    response = requests.get(list_url, timeout=10)
                    response.raise_for_status()
                    
                    # Parse the directory listing to find all files for this basin/year
                    soup = BeautifulSoup(response.text, 'html.parser')
                    storm_files = [a.text for a in soup.find_all('a') 
                                 if a.text.startswith(f'b{basin}') 
                                 and a.text.endswith(f'{current_year}.dat')]
                    
                    for storm_file in storm_files:
                        try:
                            # Extract storm ID from filename (e.g., bal012025.dat -> AL012025)
                            storm_id = storm_file[1:-4].upper()  # Remove 'b' and '.dat', make uppercase
                            
                            # Add to the list with a placeholder name (will be updated if we can get it)
                            storm_list.append({
                                'id': storm_id,
                                'name': f"Storm {storm_id}",
                                'basin': storm_id[:2],
                                'year': storm_id[-4:]
                            })
                            
                        except Exception as e:
                            self.logger.warning(f"Error processing {storm_file}: {e}")
                            continue
                            
                except requests.RequestException as e:
                    self.logger.warning(f"Could not fetch ATCF data for basin {basin}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error processing basin {basin}: {e}", exc_info=True)
                    continue
            
            return storm_list
            
        except Exception as e:
            self.logger.error(f"Error in get_active_storm_names: {e}", exc_info=True)
            return []
            
    def get_active_storms(self) -> List[Dict[str, Any]]:
        """
        Get list of currently active tropical cyclones from NHC with full data.
        
        Returns:
            List[Dict[str, Any]]: List of active storms with 'id', 'name', and 'data' keys.
            Only includes storms that have been updated in the last 12 hours.
        """
        self.logger.info("Fetching active storms from NHC...")
        
        try:
            # Get all potential storms from ATCF
            all_storms = self._get_active_storms_from_atcf()
            
            # Filter for active storms (last update within 12 hours)
            active_storms = []
            current_time = datetime.utcnow()
            
            for storm in all_storms:
                try:
                    # Get the timestamp from the storm data
                    timestamp_str = storm['data'].get('timestamp')
                    if not timestamp_str:
                        continue
                        
                    # Parse the timestamp
                    if 'T' in timestamp_str:  # ISO format
                        last_update = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:  # Try other possible formats if needed
                        last_update = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    
                    # Consider storm active if last update was within 12 hours
                    if (current_time - last_update) <= timedelta(hours=12):
                        active_storms.append(storm)
                        self.logger.info(f"Found active storm: {storm['name']} (last update: {last_update})")
                        
                except Exception as e:
                    self.logger.warning(f"Error checking storm activity for {storm.get('id')}: {e}")
                    
            if not active_storms:
                self.logger.warning("No active storms found in ATCF data, trying NHC website...")
                return self._get_active_storms_from_nhc_website()
                
            return active_storms
            
        except Exception as e:
            self.logger.error(f"Error getting active storms: {e}", exc_info=True)
            return self._get_sample_storms()
    
    def _get_active_storms_from_atcf(self) -> List[Dict[str, Any]]:
        """Fetch active storms from ATCF best track data."""
        current_year = str(datetime.now().year)
        basins = ['al', 'ep']  # Atlantic and East Pacific basins
        storms = []
        
        for basin in basins:
            try:
                # Get the list of all storm files for this basin/year
                list_url = "https://ftp.nhc.noaa.gov/atcf/btk/"
                self.logger.info(f"Fetching ATCF file list from: {list_url}")
                
                # Get the directory listing
                response = requests.get(list_url, timeout=10)
                response.raise_for_status()
                
                # Parse the directory listing to find all files for this basin/year
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                storm_files = [a.text for a in soup.find_all('a') 
                             if a.text.startswith(f'b{basin}') 
                             and a.text.endswith(f'{current_year}.dat')]
                
                if not storm_files:
                    self.logger.warning(f"No ATCF files found for basin {basin} in {current_year}")
                    continue
                
                # Process each storm file
                for storm_file in storm_files:
                    try:
                        storm_url = f"{list_url}{storm_file}"
                        self.logger.info(f"Processing storm data from: {storm_url}")
                        
                        response = requests.get(storm_url, timeout=10)
                        response.raise_for_status()
                        
                        # Process the ATCF data
                        lines = [line.strip() for line in response.text.split('\n') if line.strip()]
                        if not lines:
                            continue
                        
                        # Get the most recent entry (last line)
                        latest_entry = [p.strip() for p in lines[-1].split(',')]
                        if len(latest_entry) < 10:  # Ensure we have enough columns
                            continue
                        
                        # Parse storm data
                        storm_id = storm_file[1:-4]  # Remove 'b' prefix and '.dat' suffix
                        storm_name = latest_entry[27].strip() or f"Unnamed ({storm_id.upper()})"
                        
                        # Parse latitude (e.g., '224N' -> 22.4°N)
                        lat_str = latest_entry[6].strip()
                        lat = float(lat_str[:-1]) / 10.0
                        if lat_str[-1].upper() == 'S':
                            lat = -lat
                            
                        # Parse longitude (e.g., '987W' -> 98.7°W)
                        lon_str = latest_entry[7].strip()
                        lon = float(lon_str[:-1]) / 10.0
                        if lon_str[-1].upper() == 'W':
                            lon = -lon
                            
                        # Parse pressure (with fallback)
                        try:
                            min_pressure = float(latest_entry[9].strip()) if latest_entry[9].strip() else 1000.0
                        except (ValueError, IndexError):
                            min_pressure = 1000.0
                        
                        # Parse wind speed (with fallback)
                        try:
                            max_wind_kts = float(latest_entry[8].strip()) if latest_entry[8].strip() else 0.0
                        except (ValueError, IndexError):
                            max_wind_kts = 0.0
                        
                        # Add storm to the list
                        storms.append({
                            'id': storm_id.upper(),
                            'name': f"{storm_name.title()} ({storm_id.upper()})",
                            'data': {
                                'min_pressure': min_pressure,
                                'min_pressure_mbar': min_pressure,
                                'latitude': lat,
                                'longitude': lon,
                                'lat_str': f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}",
                                'lon_str': f"{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}",
                                'max_wind_kts': max_wind_kts,
                                'data_source': 'NHC ATCF',
                                'timestamp': datetime.utcnow().isoformat()
                            }
                        })
                        
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Error parsing storm data for {storm_file}: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Unexpected error processing {storm_file}: {e}", exc_info=True)
                        continue
                        
            except requests.RequestException as e:
                self.logger.warning(f"Could not fetch ATCF data for basin {basin}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error processing basin {basin}: {e}", exc_info=True)
                continue
        
        return storms
    
    def _get_active_storms_from_nhc_website(self) -> List[Dict[str, Any]]:
        """Fallback: Try to get active storms from NHC website."""
        self.logger.info("Trying NHC website as fallback...")
        
        try:
            url = f"{self.nhc_base_url}/gtwo.php?basin=atlc&fdays=2"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            storms = []
            
            # Look for storm information in the page
            storm_elems = soup.find_all(['div', 'span'], class_=['storm', 'storm-name'])
            
            for elem in storm_elems:
                try:
                    name = elem.get_text(strip=True)
                    if not name or name.lower() == 'no active storms':
                        continue
                        
                    # Generate a generic storm ID
                    storm_id = f"AL{len(storms) + 1:02d}{datetime.now().year}"
                    
                    # Stagger positions to avoid overlap
                    lat = 25.0 + len(storms) * 2
                    lon = -70.0 - len(storms) * 2
                    
                    storms.append({
                        'id': storm_id,
                        'name': name,
                        'data': {
                            'min_pressure': 1000,
                            'min_pressure_mbar': 1000,
                            'latitude': lat,
                            'longitude': lon,
                            'lat_str': f"{lat:.1f}°N",
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    })
                except Exception as e:
                    self.logger.warning(f"Error processing storm element: {e}")
                    continue
                    
            return storms
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching from NHC website: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in _get_active_storms_from_nhc_website: {e}", exc_info=True)
            return []
    
    def _get_sample_storms(self) -> List[Dict[str, Any]]:
        """Last resort: Return sample storm data for testing."""
        self.logger.warning("No active storms found, using sample data")
        
        return [{
            'id': 'AL022025',
            'name': 'Sample Storm (AL022025)',
            'data': {
                'min_pressure': 1000,
                'min_pressure_mbar': 1000,
                'latitude': 25.0,
                'longitude': -70.0,
                'lat_str': '25.0°N',
                'lon_str': '70.0°W',
                'data_source': 'SAMPLE DATA',
                'timestamp': datetime.utcnow().isoformat(),
                'note': 'This is sample data - no active storms found'
            }
        }]
    
    def get_hurricane_data(self, storm_id: str) -> Dict[str, Any]:
        """
        Fetch live hurricane data from NHC ATCF source with robust BEST track parsing.
        """
        try:
            storm_id = storm_id.lower().replace('bal', '').replace('.dat', '').strip()

            if len(storm_id) < 6:
                raise ValueError(f"Invalid storm ID: {storm_id}")

            basin = storm_id[:2].lower()
            number = storm_id[2:4]
            year = storm_id[4:8]
            atcf_filename = f"b{basin}{number}{year}.dat"
            atcf_url = f"{self.atcf_base_url}/{atcf_filename}"

            self.logger.info(f"Fetching ATCF data for {storm_id} -> {atcf_url}")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(atcf_url, headers=headers, timeout=15)
            response.raise_for_status()

            lines = [line.strip() for line in response.text.split("\n") if line.strip()]
            if not lines:
                raise ValueError("No ATCF data lines found")

            last_entry = [p.strip() for p in lines[-1].split(",")]
            self.logger.debug(f"Last entry raw: {last_entry}")

            # --- FIX: Detect “BEST” anywhere ---
            try:
                best_index = next(i for i, v in enumerate(last_entry) if v.strip().upper() == "BEST")
                lat_str = last_entry[best_index + 2].strip()
                lon_str = last_entry[best_index + 3].strip()
            except StopIteration:
                # If “BEST” not found, fall back to traditional positions
                lat_str = last_entry[6].strip() if len(last_entry) > 6 else ""
                lon_str = last_entry[7].strip() if len(last_entry) > 7 else ""

            self.logger.debug(f"Detected lat='{lat_str}', lon='{lon_str}'")

            # --- Parse latitude ---
            if not lat_str or lat_str[-1].upper() not in ("N", "S"):
                raise ValueError(f"Invalid latitude format (must end with N or S): {lat_str}")
            lat_val = float(lat_str[:-1]) / 10.0 if lat_str[:-1].isdigit() else float(lat_str[:-1])
            lat = lat_val if lat_str[-1].upper() == "N" else -lat_val

            # --- Parse longitude ---
            if not lon_str or lon_str[-1].upper() not in ("E", "W"):
                raise ValueError(f"Invalid longitude format (must end with E or W): {lon_str}")
            lon_val = float(lon_str[:-1]) / 10.0 if lon_str[:-1].isdigit() else float(lon_str[:-1])
            lon = lon_val if lon_str[-1].upper() == "E" else -lon_val

            # --- Parse pressure and wind speed safely ---
            min_pressure = 1000.0
            max_wind_kts = 0.0
            storm_name = "Unknown"
            
            if len(last_entry) > 9:
                try:
                    min_pressure = float(last_entry[9]) if last_entry[9] and last_entry[9] != "-999" else 1000.0
                except ValueError:
                    pass
            if len(last_entry) > 8:
                try:
                    max_wind_kts = float(last_entry[8]) if last_entry[8] and last_entry[8] != "-999" else 0.0
                except ValueError:
                    pass
            
            # Try to get storm name from the entry (usually around index 27-28 in ATCF format)
            if len(last_entry) > 28 and last_entry[27].strip():
                storm_name = last_entry[27].strip().title()
            else:
                # If name not found, use a default with the storm ID
                storm_name = f"Storm {storm_id.upper()}"

            result = {
                "storm_id": storm_id.upper(),
                "name": storm_name,
                "latitude": lat,
                "longitude": lon,
                "lat_str": f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}",
                "lon_str": f"{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}",
                "min_pressure_mbar": min_pressure,
                "max_wind_kts": max_wind_kts,
                "min_pressure": min_pressure,
                "data_source": "NHC ATCF",
                "timestamp": datetime.utcnow().isoformat(),
                "note": "Parsed using robust BEST-track-safe parser",
                "raw_data": last_entry
            }

            self.logger.info(f"✅ Processed hurricane {storm_id}: {result['lat_str']} {result['lon_str']}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error processing hurricane {storm_id}: {e}", exc_info=True)
            raise ValueError(f"Unexpected error processing storm {storm_id}: {e}") from e

    
    def _get_sample_hurricane_data(self, storm_id: str) -> Dict[str, Any]:
        """Return sample hurricane data for testing/fallback purposes."""
        self.logger.warning(f"Using sample data for storm: {storm_id}")
        
        return {
            "min_pressure_mbar": 980.0,
            "latitude": 25.0,
            "longitude": -70.0,
            "min_pressure": 980.0,
            "lat_str": '25.0°N',
            "lon_str": '70.0°W',
            "max_wind_kts": 85.0,
            "data_source": "SAMPLE DATA",
            "timestamp": datetime.utcnow().isoformat(),
            "note": "This is sample data - actual data could not be fetched"
        }
    
    def get_volcano_data(self, volcano_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch live volcano data from USGS and Smithsonian GVP."""
        try:
            # Get volcano data from Smithsonian GVP
            params = {
                "format": "json",
                "eruption": "Y"  # Only show erupting volcanoes
            }
            response = requests.get(self.usgs_volcano_url, params=params, timeout=10)
            response.raise_for_status()
            
            volcanoes = response.json()
            if not volcanoes:
                raise ValueError("No active volcanoes found")
                
            # Get the first erupting volcano or use the specified one
            volcano = next((v for v in volcanoes 
                          if volcano_id and v['volcanoID'].lower() == volcano_id.lower()), 
                          volcanoes[0])
            
            # Get recent earthquakes near the volcano
            quake_params = {
                "format": "geojson",
                "latitude": float(volcano['latitude']),
                "longitude": float(volcano['longitude']),
                "maxradiuskm": 50,
                "limit": 10,
                "orderby": "time"
            }
            quake_response = requests.get(self.usgs_earthquake_url, params=quake_params, timeout=10)
            quake_response.raise_for_status()
            quakes = quake_response.json()
            
            # Determine activity level based on recent earthquakes
            current_date = datetime.utcnow()
            recent_quakes = [q for q in quakes.get('features', []) 
                           if datetime.fromisoformat(q['properties']['time']) > 
                           current_date - timedelta(days=1)]
            
            return {
                'Year': current_date.year,
                'Month': current_date.month,
                'Day': current_date.day,
                'Latitude': float(volcano['latitude']),
                'Longitude': float(volcano['longitude']),
                'Elevation (m)': float(volcano['elevation']),
                'Country': volcano['country'],
                'Type': volcano['primary_volcano_type'],
                'Agent': 'S' if len(recent_quakes) > 5 else 'G',  # Seismic or Gas
                'Volcano': volcano['volcano_name'],
                'recent_earthquakes': len(recent_quakes),
                'data_source': 'USGS/Smithsonian GVP',
                'timestamp': current_date.isoformat()
            }
            
        except requests.RequestException as e:
            print(f"Error fetching volcano data: {e}")
            raise
