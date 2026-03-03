
from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
from model import GenderAgeModel
import io

app = Flask(__name__)

# ✅ This fixes ngrok blank page issue
@app.after_request
def add_header(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

model = GenderAgeModel()
model.load_state_dict(torch.load(
    '/content/model.pth',
    map_location='cpu',
    weights_only=False
))
model.eval()
print("✅ Model loaded!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file   = request.files['image']
        img    = Image.open(io.BytesIO(file.read())).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            gender_pred, age_pred = model(tensor)
        gender = 'Male' if gender_pred.item() > 0.5 else 'Female'
        age    = round(age_pred.item())
        result = {'gender': gender, 'age': age}
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
