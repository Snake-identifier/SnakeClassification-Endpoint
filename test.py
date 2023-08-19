# Description: This file is used to test the model by sending a POST request to the server.
import requests
import base64
from PIL import Image
import io

# Load your image using PIL or any other image processing library
image_path = 'images.jpg'
image = Image.open(image_path)

# Convert image to bytes and then to base64 encoded string
image_bytes = io.BytesIO()
image.save(image_bytes, format='JPEG')
encoded_image = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

payload = {
    "image": encoded_image
}

server_url = "http://localhost:8080"

try:
    
    response = requests.post(server_url, json=payload, headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        response_data = response.json()
        print("Predicted Class:", response_data['predicted_class'])
        print("Predicted Probability:", response_data['predicted_probability'])
        print("Class Label:", response_data['class_label'])
    else:
        print("Request failed with status code:", response.status_code)

except Exception as e:
    print("An error occurred:", str(e))
