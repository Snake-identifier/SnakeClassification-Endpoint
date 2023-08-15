from flask import Flask, request, jsonify
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
from dotenv import load_dotenv

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        imgname=request.files['image'].filename
        # print(file)
        if imgname is None or imgname == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(imgname):
            return jsonify({'error': 'format not supported'})
        try:
            # Get the uploaded image
            uploaded_image = request.files['image'].read()
            image = Image.open(io.BytesIO(uploaded_image)).convert('RGB')
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
                'class_label': class_labels[predicted_class]  # Replace with your class labels
            }

            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=False, port=5000)