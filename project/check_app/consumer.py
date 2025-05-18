# import json
# from channels.generic.websocket import AsyncWebsocketConsumer
# import pandas as pd
# from sklearn.preprocessing import OrdinalEncoder
# from datetime import datetime , date

# from channels.layers import get_channel_layer
# import torch.nn as nn
# # from .views import F_D_model, F_D_scalerX, F_D_scalerY  # Adjust import path
# from asgiref.sync import async_to_sync ,asyncio
# import requests
# import joblib
# import torch





# # class PredictConsumer(AsyncWebsocketConsumer):
# #     async def connect(self):
# #         await self.accept()
# #         await self.send(text_data=json.dumps({"message": "Connected to prediction server."}))

# #     async def receive(self, text_data):

# #         channel_layer = get_channel_layer()
        
# #         # Broadcast to group, excluding a specific user (if needed)
# #         # async_to_sync(
# #         await channel_layer.group_send(
# #             "notifications",  # Group name
# #             {
# #                 "type": "send_notification",
# #                 "message": 'test call:D',
# #             }
# #         )



# #         data = json.loads(text_data)
# #         input_features = data.get("input")
# #         if input_features is None:
# #             await self.send(text_data=json.dumps({"error": "No input data provided."}))
# #             return

# #         print(input_features)
   

# #         try:
    
# #             response = requests.post(predict_url, json=input_features)

# #             if response.status_code != 400:
# #                 response_data = response.json()
# #                 duration = response_data.get("duration")
# #                 freq = response_data.get("frequency")
# #             else:
# #                 duration = None
# #                 freq = None
# #             print( duration)
# #             print( freq)
# #             await self.send(text_data=json.dumps({
# #                 "duration": float(duration),
# #                 "frequency": float(freq)
# #             }))
# #         except Exception as e:
            
# #             # print( "in not 2")
# #             await self.send(text_data=json.dumps({"error": str(e)}))

# #     async def disconnect(self, close_code):
# #         pass
