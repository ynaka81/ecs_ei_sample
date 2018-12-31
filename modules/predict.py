import typing

import mxnet as mx
import numpy as np
from singleton_decorator import singleton

from .downloader import ModelDownloader


@singleton
class ImageClassifier:
    """
    The image classifier with multiple pre-trained models.

    :param model_name: (str) The pre-trained model name.
    :param batch_size: (int) The batch size of model.
    :param device: (str) The device name such as cpu or ei.
    """

    def __init__(self, model_name: str, batch_size: int, device: str) -> None:
        # Prepare context.
        if device == 'ei':
            ctx = mx.eia()
        else:
            ctx = mx.cpu()
        # Download model.
        downloader = ModelDownloader()
        model_path = downloader(model_name)
        # Load model.
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
        self.mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3, 224, 224))], label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)

    def preprocess(self, image: mx.ndarray.NDArray) -> mx.ndarray.NDArray:
        # Convert into format (batch, RGB, width, height).
        image = mx.image.imresize(image, 224, 224)  # resize
        image = image.transpose((2, 0, 1))  # Channel first
        image = image.expand_dims(axis=0)  # batchify
        image = image.astype('float32')  # for gpu context
        return image

    def __call__(self, images: typing.List[mx.ndarray.NDArray]) -> typing.List[typing.List[typing.Dict[str, float]]]:
        # Pre-process.
        images = [self.preprocess(image) for image in images]
        # Compute the predict probabilities.
        data = mx.ndarray.concat(*images, dim=0)
        batch = mx.io.DataBatch(data=(data, ))
        self.mod.forward(batch)
        prob = self.mod.get_outputs()[0].asnumpy()
        # Return the top-5.
        top5s = np.argsort(prob)[:, ::-1]
        top5s = [dict([(str(j), float(prob[i, j])) for j in top5[:5]]) for i, top5 in enumerate(top5s)]
        return top5s
