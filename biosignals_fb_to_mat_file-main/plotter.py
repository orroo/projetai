import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def plot_sensor_data(mat_path):
    # Load data from .mat file
    data = loadmat(mat_path)
    
    # Extract arrays (matlab arrays may come with extra dimensions)
    ecg = np.ravel(data['ecg'])
    ppg = np.ravel(data['ppg'])
    timestamp = np.ravel(data['timestamp'])

    # Convert timestamps to relative time (assuming milliseconds or something)
    time_sec = (timestamp - timestamp[0]) / 1000.0  # seconds from start
    
    plt.figure(figsize=(12, 6))
    
    # Plot ECG
    plt.subplot(2, 1, 1)
    plt.plot(time_sec, ecg, label='ECG', color='red')
    plt.title('ECG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Plot PPG
    plt.subplot(2, 1, 2)
    plt.plot(time_sec, ppg, label='PPG', color='blue')
    plt.title('PPG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_sensor_data('sensor_data.mat')
