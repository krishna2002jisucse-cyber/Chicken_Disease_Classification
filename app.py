from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys
import json
from flask_cors import CORS, cross_origin

# Add src to Python path so cnnClassifier can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"

@app.route("/metrics", methods=['GET'])
@cross_origin()
def metricsRoute():
    metrics_path = os.path.join("artifacts", "evaluation_results", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Metrics not found. Run evaluation first."}), 404

@app.route("/evaluation_images/<filename>", methods=['GET'])
@cross_origin()
def evaluation_image(filename):
    img_dir = os.path.join(os.getcwd(), "artifacts", "evaluation_results")
    return send_from_directory(img_dir, filename)

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) # local host
