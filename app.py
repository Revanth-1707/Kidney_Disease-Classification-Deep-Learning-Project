import os
import time
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US_UTF-8')

app = Flask(__name__)
CORS(app)

# ---------------- Logging ----------------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("kidney_classifier_api")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler("logs/api.log", maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ---------------- Prometheus metrics ----------------
REQUEST_COUNT = Counter(
    "kidney_api_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
PREDICTION_COUNT = Counter(
    "kidney_predictions_total", "Total predictions made", ["predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "kidney_prediction_latency_seconds", "Time taken to run a prediction"
)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    REQUEST_COUNT.labels(endpoint="/", method="GET", status="200").inc()
    return render_template('index.html')


@app.route("/health", methods=['GET'])
@cross_origin()
def health():
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status="200").inc()
    return jsonify({"status": "healthy"}), 200


@app.route("/metrics", methods=['GET'])
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    logger.info("Training triggered via /train by %s", request.remote_addr)
    os.system("dvc repro")
    REQUEST_COUNT.labels(endpoint="/train", method=request.method, status="200").inc()
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    start_time = time.time()
    try:
        body = request.json
        if not body or 'image' not in body:
            logger.warning("Predict called with missing 'image' field from %s", request.remote_addr)
            REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="400").inc()
            return jsonify({"error": "Missing 'image' field (base64 string) in request body"}), 400

        image = body['image']
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        predicted_class = result[0].get("image", "unknown") if isinstance(result, list) else str(result)
        PREDICTION_COUNT.labels(predicted_class=str(predicted_class)).inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()

        logger.info(
            "Prediction served | result=%s | latency=%.4fs | ip=%s",
            result, latency, request.remote_addr
        )
        return jsonify(result)

    except Exception as e:
        logger.error("Prediction failed | error=%s", str(e))
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        return jsonify({"error": "Internal prediction error", "details": str(e)}), 500


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)