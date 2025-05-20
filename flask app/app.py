from flask import Flask, jsonify, request
import pandas as pd
import json
import os
import joblib
import torch
import torch.nn as nn
import numpy as np  
from datetime import datetime , date

app = Flask(__name__)


class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=2, num_layers=1):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)  # output: (batch_size, sequence_length, hidden_size)
        out = self.dropout(out)
        # out = self.fc1(out)
        # out = self.dropout1(out)
        out = self.fc(out)
        return out


F_D_model = TimeSeriesModel(44)  # rebuild the model architecture exactly
F_D_model.load_state_dict(torch.load('pkl\model_weights.pt'))
F_D_model.eval()

F_D_scalerX =  joblib.load("pkl\scalerX.pkl")
F_D_scalerY =  joblib.load("pkl\scalerY.pkl")




def predict(model, input_df, device='cpu'):
    model.eval()  # Set model to evaluation mode
    
    # Convert to NumPy and apply scaler
    input_scaled = F_D_scalerX.transform(input_df.values.astype(np.float32))
    
    # Convert to Tensor and move to device
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
        output = F_D_scalerY.inverse_transform(output)
        # print(output)

    # If output is a tuple/list of two (duration and session), process each
    # if isinstance(output, (tuple, list)):
        duration_pred = output[0][0]
        session_pred = output[0][1]
        # Inverse transform each prediction independently
        # duration_pred = F_D_scalerY.inverse_transform(duration_pred)
        # session_pred = F_D_scalerY.inverse_transform(session_pred)
        return duration_pred, session_pred
    # else:
        # output_np = output.numpy()
        # output_inv = F_D_scalerY.inverse_transform(output_np)
        # return -1,-1



# Load CSV once (if it’s not huge)
CSV_PATH = os.path.join('pkl', 'samples_of_a_patient.csv')
sample_df = pd.read_csv(CSV_PATH)

@app.route('/random_sample', methods=['GET'])
def get_random_sample():
    sample_json = sample_df.sample(1).to_json(orient='records')
    data = json.loads(sample_json)   # Extract single dict
    return jsonify(data)





@app.route('/predict', methods=['POST'])
def predict_duration():
    try:
        input_features = request.get_json()

        treatment = pd.DataFrame([input_features])

        # Feature Engineering
        treatment["dialysisstart"] = pd.to_datetime(treatment["dialysisstart"])
        treatment["Ds_hour"] = treatment.dialysisstart.dt.hour
        treatment["Ds_minutes"] = treatment.dialysisstart.dt.minute

        treatment["keyindate"] = pd.to_datetime(treatment["keyindate"])
        treatment["session_year"] = treatment.keyindate.dt.year
        treatment["session_month"] = treatment.keyindate.dt.month
        treatment["session_dayofweek"] = treatment.keyindate.dt.dayofweek
        treatment["session_is_weekend"] = treatment.session_dayofweek.apply(lambda x: 1 if x >= 5 else 0)

        treatment["first_dialysis"] = pd.to_datetime(treatment["first_dialysis"])
        treatment["fd_year"] = treatment.first_dialysis.dt.year
        treatment["fd_month"] = treatment.first_dialysis.dt.month

        datatime = datetime.now()
        treatment["datatime"] = pd.to_datetime(datatime)

        treatment["year"] = treatment.datatime.dt.year
        treatment["month"] = treatment.datatime.dt.month
        treatment["dayofweek"] = treatment.datatime.dt.dayofweek
        treatment["is_weekend"] = treatment.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        treatment["hour"] = treatment.datatime.dt.hour
        treatment["minutes"] = treatment.datatime.dt.minute

        # Features expected by the model
        feature_cols = ['sbp', 'dbp', 'temperature', 'conductivity', 'uf',
           'blood_flow', 'gender', 'birthday', 'DM',
            'weightstart', 'weightend', 'dryweight',
           'pulse', 'respiratory_rate', 'blood_oxygen_lvl', 'glucose_lvl',
           'hypotension', 'age', 'dialyzer', 'bath', 'technique', 'gain',
           'bath_temperature', 'replacement_Volume', 'kt', 'Bath_Flow',
           'bicarbonate_conductivity', 'arterial_Pressure', 'Venous_Pressure',
           'transmembrane_Pressure', 
            'Ds_hour', 'Ds_minutes', 'session_year',
           'session_month', 'session_dayofweek', 'session_is_weekend', 'year',
           'month', 'dayofweek', 'is_weekend', 'hour', 'minutes', 'fd_year',
           'fd_month']

        X = treatment[feature_cols]

        # Model prediction
        duration, freq = predict(F_D_model, X)

        return jsonify({
            "duration": float(duration),
            "frequency": float(freq)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)





from scipy.io import loadmat


from scipy.interpolate import interp1d

# Load CSV once (if it’s not huge)
# CSV_PATH = os.path.join('pkl', 'samples_of_a_patient.csv')
# mat = loadmat("../biosignals_fb_to_mat_file-main/sensor_data.mat")
mat_path = "../biosignals_fb_to_mat_file-main/sensor_data.mat"


# print("Upsampled shape:", upsampled_data[0,0])  # (1, 2, 125)
# print("Upsampled shape:", upsampled_data[0,1])  # (1, 2, 125)

# print("Upsampled shape:", upsampled_data.shape)  # (1, 2, 125)
@app.route('/mat_data', methods=['GET'])
def get_mat_data():
    mat_data = loadmat(mat_path)
  
    combined_data = np.stack((mat_data['ppg'],mat_data['ecg']), axis=1)  # shape (1, 2, 50)
    print("Combined shape:", combined_data.shape)  # Should be (1, 2, 50)

    x_original = np.linspace(0, 1, 50)
    
    x_new = np.linspace(0, 1, 125)  

    upsampled_data = np.zeros((1, 2, 125))  

    for i in range(2):  # For ECG (i=0) and PPG (i=1)
        f = interp1d(x_original, combined_data[0, i, :], kind='cubic')  # 'linear' or 'cubic'
        upsampled_data[0, i, :] = f(x_new)
    json_data = json.dumps(upsampled_data.tolist())
    data = json.loads(json_data)   # Extract single dict
    print(data)
    return jsonify(data)








if __name__ == '__main__':
    app.run(debug=True)



