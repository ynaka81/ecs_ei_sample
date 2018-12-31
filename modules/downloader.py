import os
import requests

from tqdm import tqdm


class ModelDownloader:
    """
    The downloader of pre-trained model.
    """

    __URL = {
        'caffenet': ('http://data.dmlc.ml/mxnet/models/imagenet/caffenet/caffenet-symbol.json', 'http://data.dmlc.ml/models/imagenet/caffenet/caffenet-0000.params'),
        'squeezenet': ('http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.1-symbol.json', 'http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.1-0000.params'),
        'resnet': ('http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-symbol.json', 'http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-0000.params'),
        'resnext': ('http://data.dmlc.ml/models/imagenet/resnext/101-layers/resnext-101-64x4d-symbol.json', 'http://data.dmlc.ml/models/imagenet/resnext/101-layers/resnext-101-64x4d-0000.params')
    }

    def __call__(self, model_name: str) -> str:
        """
        Download pre-trained model.

        :param model_name: (str) The pre-trained model name.
        """
        os.makedirs(f'models/{model_name}', exist_ok=True)
        for url in self.__URL[model_name]:
            file_size = int(requests.head(url).headers['Content-Length'])
            response = requests.get(url, stream=True)
            filename = os.path.basename(url)
            downloaded_model_name = '-'.join(filename.split('-')[:-1])
            filename = f'models/{model_name}/{filename}'
            if os.path.isfile(filename):
                continue
            with tqdm(total=file_size, unit='B', unit_scale=True) as bar:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                        bar.update(len(chunk))
        return f'models/{model_name}/{downloaded_model_name}'
