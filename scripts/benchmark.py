import os
import requests
import sys
from io import BytesIO

from benchmarker import Benchmarker


base_url = os.environ['BASE_URL']
N = int(sys.argv[1])
filename = 'cat.jpg'
with open(filename, 'rb') as f:
    image = f.read()
    with Benchmarker(cycle=10, width=25) as bench:
        @bench('test')
        def _(bm):
            for _ in bm:
                files = [('images', (filename, BytesIO(image), 'image/jpeg')) for _ in range(N)]
                response = requests.post(
                    f'{base_url}/classify',
                    files=files
                )
                response.raise_for_status()
