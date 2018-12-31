import os

from flask import Flask
from flask_restful import Api

from modules.predict import ImageClassifier
from modules.resources import Index, Classify


if 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI' in os.environ:
    del os.environ['AWS_CONTAINER_CREDENTIALS_RELATIVE_URI']
    del os.environ['ECS_CONTAINER_METADATA_URI']
# Initialize application.
app = Flask(__name__)
api = Api(app)
# Add Resources.
api.add_resource(Index, '/')
api.add_resource(Classify, '/classify')
# Instantiate ImageClassifier for warm start.
model_name = os.environ['MODEL_NAME']
batch_size = int(os.environ['BATCH_SIZE'])
device = os.environ['DEVICE']
ImageClassifier(model_name, batch_size, device)
# Start server.
app.run(host='0.0.0.0', port=5000, debug=True)
