import mxnet as mx
import numpy as np
from werkzeug import FileStorage
from flask_restful import Resource, reqparse

from .predict import ImageClassifier


class Index(Resource):
    """
    Resource for health check.
    """

    def get(self):
        return ''


class Classify(Resource):
    """
    The resource class for image classification.
    """

    def __init__(self):
        # Prepare request parser.
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('images', required=True, type=FileStorage, action='append', location='files')
        # Get classifier.
        self.classifier = ImageClassifier._instance

    def post(self):
        # Parse arguments.
        args = self.parser.parse_args()
        image_files = args['images']
        # Convert input images to mx.ndarray.NDArray.
        images = []
        for image_file in image_files:
            raw_image = np.fromstring(image_file.read(), np.uint8)
            image = mx.img.imdecode(raw_image)
            images.append(image)
        # Return classification result.
        prod = self.classifier(images)
        return prod
