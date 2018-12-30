import os

from flask import Flask
from flask_restful import Api

from modules.predict import ImageClassifier
from modules.resources import Index, Classify


# Initialize application.
app = Flask(__name__)
api = Api(app)
# Add Resources.
api.add_resource(Index, '/')
api.add_resource(Classify, '/classify')
# Instantiate ImageClassifier for warm start.
model_name = os.environ['MODEL_NAME']
batch_size = int(os.environ['BATCH_SIZE'])
ImageClassifier(model_name, batch_size)
# Start server.
app.run(host='0.0.0.0', port=5000, debug=True)
