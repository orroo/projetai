from django.shortcuts import render
import torch
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# Create your views here.
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync


async def async_send_broadcast_notification(message):
    channel_layer = get_channel_layer()
    await channel_layer.group_send(
        "notifications",  # The group name
        {
            "type": "send_notification",
            "message": message
        }
    )


# channel_layer = get_channel_layer()
# async_to_sync(channel_layer.group_send)(
#     "notifications",
#     {
#         "type": "send_notification",
#         "message": "New notification received!"
#     }
# )




# @csrf_exempt
# def send_notification(request):
#     if request.method == "POST":
#         data = json.loads(request.body)
#         message = data.get("message", "Default message")

#         channel_layer = get_channel_layer()
#         async_to_sync(channel_layer.group_send)(
#             "notifications",
#             {
#                 "type": "send_notification",
#                 "message": message,
#             }
#         )
#         return JsonResponse({"status": "sent"})





import torch.nn as nn


def about(request):
    return render(request,"about-us.html")


def blog(request):
    return render(request,"blog.html")


def contact(request):
    return render(request,"contact.html")


def department(request):
    return render(request,"department.html")

    
def doctors(request):
    return render(request,"doctors.html")


    
def element(request):
    return render(request,"element.html")


    
def index(request):
    return render(request,"index.html")


    
def singleblog(request):
    return render(request,"single-blog.html")


# def predict(model, input_tensor, device='cpu'):
#     model.eval()  # Set model to evaluation mode
#     input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
#     input_tensor = F_D_scalerX.transform(input_tensor) 

#     with torch.no_grad():
#         output = model(input_tensor)
#         output = F_D_scalerY.inverse_transform(output)

#     # If output is a tuple/list of two (duration and session), unpack it
#     if isinstance(output, (tuple, list)):
#         duration_pred = output[0].cpu().numpy()
#         session_pred = output[1].cpu().numpy()
#         return duration_pred, session_pred
#     else:
#         return output.cpu().numpy()



# def F_D_pred(request):
#     input= [101.0,53.0,36.8,14.5,0.46,200.0,1.0,1932.0,1.0,65.1,63.5,63.5,71.0,15.0,98.0,79.0,1.0,49.0,2.0,1.0,1.0,1.4,0.0,26.13,67.2,809.0,0.0,-185.0,190.0,185.0,7,5,2015,11,3,0,2015,11,3,0,17,46,2013,9]
#     input_df = pd.DataFrame([input])  # shape: (1, n_features)

#     predicted_dur, predicted_freq = predict(F_D_model, input_df)

#     # For example purposes
#     true_dur = 240.0
#     true_freq = 2.693548387096774

#     context = {
#         "dur": predicted_dur,
#         "freq": predicted_freq,
#         "act_dur": true_dur,
#         "act_freq": true_freq
#     }
    
#     # ws.on_open = lambda ws: ws.send(json.dumps({
#     #     "input": [101.0,53.0,36.8,14.5,0.46,200.0,1.0,1932.0,1.0,65.1,63.5,63.5,
#     #             71.0,15.0,98.0,79.0,1.0,49.0,2.0,1.0,1.0,1.4,0.0,26.13,67.2,
#     #             809.0,0.0,-185.0,190.0,185.0,7,5,2015,11,3,0,2015,11,3,0,
#     #             17,46,2013,9]
#     # }))


#     return render(request, "test.html", context)




from django.shortcuts import render

def live_prediction_view(request):
    return render(request, "test.html")



# import websocket
# import json

# def on_message(ws, message):
#     print("Received:", message)

# ws = websocket.WebSocketApp(
#     "ws://localhost:8000/ws/predict/",
#     on_message=on_message
# )

# ws.on_open = lambda ws: ws.send(json.dumps({
#     "input": [101.0,53.0,36.8,14.5,0.46,200.0,1.0,1932.0,1.0,65.1,63.5,63.5,
#               71.0,15.0,98.0,79.0,1.0,49.0,2.0,1.0,1.0,1.4,0.0,26.13,67.2,
#               809.0,0.0,-185.0,190.0,185.0,7,5,2015,11,3,0,2015,11,3,0,
#               17,46,2013,9]
# }))

# ws.run_forever()
