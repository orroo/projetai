import pyrebase
import json
from scipy.io import savemat
import time

# Firebase config with your project details
firebase_config = {
    "apiKey": "AIzaSyCgcesVqFN1VhKmt14Ximp2u9qOKpbSItU",
    "authDomain": "esp32-biosignals.firebaseapp.com",
    "databaseURL": "https://esp32-biosignals-default-rtdb.europe-west1.firebasedatabase.app",
    "projectId": "esp32-biosignals",
    "storageBucket": "esp32-biosignals.appspot.com",
    "messagingSenderId": "",
    "appId": ""
}

# Your Firebase user credentials
USER_EMAIL = "test1@gmail.com"
USER_PASSWORD = "135125@"

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

# Authenticate user
try:
    user = auth.sign_in_with_email_and_password(USER_EMAIL, USER_PASSWORD)
    print("✅ Successfully logged in to Firebase.")
except Exception as e:
    print("❌ Failed to authenticate:", e)
    exit(1)

last_max_timestamp = 0  # Track last saved timestamp to avoid duplicates

def fetch_sensor_data():
    """
    Fetch sensor_data from Firebase Realtime Database using authenticated user.
    """
    data = db.child("sensor_data").get(user['idToken'])
    return data.val()

def save_to_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump({"sensor_data": data}, f)

def json_to_mat_from_data(sensor_data_list, mat_path):
    global last_max_timestamp

    if not sensor_data_list:
        print("No data available in sensor_data.")
        return

    # Filter out null entries and those without dataPoints
    valid_entries = [entry for entry in sensor_data_list if entry and "dataPoints" in entry]

    if not valid_entries:
        print("No valid entries with dataPoints found.")
        return

    latest_entry = valid_entries[-1]  # Use latest non-null sensor data
    data_points = latest_entry["dataPoints"]

    # Filter for new data points only
    new_points = [p for p in data_points if p["timestamp"] > last_max_timestamp]
    if not new_points:
        print("No new data to save.")
        return

    # Update last_max_timestamp
    last_max_timestamp = max(p["timestamp"] for p in new_points)

    # Extract lists for MATLAB
    ecg = [p["ecg"] for p in new_points]
    ppg = [p["ppg"] for p in new_points]
    timestamp = [p["timestamp"] for p in new_points]

    mat_data = {
        'ecg': ecg,
        'ppg': ppg,
        'timestamp': timestamp
    }

    savemat(mat_path, mat_data)
    print(f"✅ Saved {len(ecg)} new points to {mat_path}")

# File paths for backup and MATLAB
json_file = 'sensor_data.json'
mat_file = 'sensor_data.mat'

print("Starting Firebase data fetch loop...")

while True:
    try:
        sensor_data = fetch_sensor_data()
        save_to_json(sensor_data, json_file)  # Optional: JSON backup
        json_to_mat_from_data(sensor_data, mat_file)
    except Exception as e:
        print("❌ Error:", e)

    time.sleep(10)  # Wait 10 seconds before next update
