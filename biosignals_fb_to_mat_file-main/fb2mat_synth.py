import pyrebase
import numpy as np
from scipy.io import savemat
import time

# Firebase config - fill with your actual values
firebase_config = {
    "apiKey": "AIzaSyBfMJZ0dAM4jJJfqMUJPhl-2s03iCUtdsI",
    "authDomain": "esp32-biosignalss.firebaseapp.com",
    "databaseURL": "https://esp32-biosignalss-default-rtdb.europe-west1.firebasedatabase.app",
    "projectId": "esp32-biosignalss",
    "storageBucket": "esp32-biosignalss.appspot.com",
    "messagingSenderId": "YOUR_MESSAGING_SENDER_ID",  # optional
    "appId": "YOUR_APP_ID"  # optional
}

# User credentials
USER_EMAIL = "test1@gmail.com"
USER_PASSWORD = "135125@"

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

def synthesize_ecg(length, sampling_rate=100):
    t = np.linspace(0, length / sampling_rate, length)
    ecg = 0.6 * np.sin(2 * np.pi * 1.0 * t)
    ecg += 0.2 * np.sin(2 * np.pi * 20.0 * t)
    ecg += 0.1 * np.random.normal(0, 0.05, length)
    return ecg

def synthesize_ppg(length, sampling_rate=100):
    t = np.linspace(0, length / sampling_rate, length)
    ppg = 200 + 100 * (0.5 * (1 + np.sin(2 * np.pi * 1.2 * t)))
    ppg += 5 * np.random.normal(0, 0.1, length)
    return ppg

def smooth_signal(signal, window_len=5):
    window = np.ones(window_len) / window_len
    return np.convolve(signal, window, mode='same')

def enhance_existing_signal(signal):
    noisy = signal + 0.05 * np.random.normal(0, 1, len(signal))
    smooth = smooth_signal(noisy, window_len=3)
    return smooth

def fetch_sensor_data():
    # Login
    user = auth.sign_in_with_email_and_password(USER_EMAIL, USER_PASSWORD)
    
    # Read sensor_data from Firebase DB root node "sensor_data"
    data = db.child("sensor_data").get(user['idToken']).val()
    
    if not data:
        print("No data found in Firebase")
        return None
    
    # The data is a list with many nulls, find the first non-null sensor data node
    sensor_node = None
    for node in data:
        if node is not None and "dataPoints" in node:
            sensor_node = node
            break
    if sensor_node is None:
        print("No valid sensor data node found")
        return None
    
    data_points = sensor_node["dataPoints"]
    return data_points

def process_and_save(mat_path):
    data_points = fetch_sensor_data()
    if not data_points:
        return
    
    # Extract original data arrays
    ecg_raw = np.array([pt["ecg"] for pt in data_points])
    ppg_raw = np.array([pt["ppg"] for pt in data_points])
    timestamps = np.array([pt["timestamp"] for pt in data_points])
    
    length = len(ecg_raw)
    if length == 0:
        print("Empty data points")
        return
    
    # Option 1: Replace with synthesized data
    ecg_data = synthesize_ecg(length)
    ppg_data = synthesize_ppg(length)
    
    # Option 2: Or enhance existing data by smoothing + noise (uncomment if you want)
    # ecg_data = enhance_existing_signal(ecg_raw)
    # ppg_data = enhance_existing_signal(ppg_raw)
    
    # Prepare data dict for .mat saving
    mat_data = {
        'ecg': ecg_data,
        'ppg': ppg_data,
        'timestamp': timestamps
    }
    
    savemat(mat_path, mat_data)
    print(f"Saved {length} points to {mat_path}")

if __name__ == "__main__":
    mat_file = 'sensor_data.mat'
    while True:
        try:
            process_and_save(mat_file)
        except Exception as e:
            print("Error:", e)
        time.sleep(10)  # update every 10 seconds
