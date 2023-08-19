from flask import Flask, request, jsonify
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
import base64

app = Flask(__name__)

trained_model = torch.load('TrainedMobileNetV2Model.pth')
trained_model.eval()

# Transformation for incoming images

transform = transforms.Compose([
    transforms.Resize((240, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
@app.route("/", methods=['POST'])
def predict():
            
    if request.method == 'POST':
        try:
            # Get the uploaded image data from the JSON payload
            data = request.json
            if 'image' not in data:
                return jsonify({'error': 'no image data'})

            encoded_image = data['image']
            image_bytes = base64.b64decode(encoded_image)

            # Convert bytes to PIL image and apply transformations
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            # Make predictions
            with torch.no_grad():
                output = trained_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_class = probabilities.topk(1)
                predicted_class = top_class.item()
                predicted_prob = top_prob.item()
            class_labels = ['Cobra', 'Common Krait', 'Russell\'s Viper', 'hump - nosed pit viper', 'krait Bungarus ceylonicus',
                            'non venomous', 'saw- scaled viper']
            # Prepare JSON response
            response = {
                'predicted_class': predicted_class,
                'predicted_probability': predicted_prob,
                'class_label': class_labels[predicted_class] 
            }

            print(response)
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', port=5000)