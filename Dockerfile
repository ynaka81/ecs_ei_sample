FROM python:3.6

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
RUN pip install --no-cache-dir awscli && \
    AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
    aws s3 cp s3://amazonei-apachemxnet/amazonei_mxnet-1.3.0-py2.py3-none-manylinux1_x86_64.whl /tmp && \
    pip install --no-cache-dir /tmp/amazonei_mxnet-1.3.0-py2.py3-none-manylinux1_x86_64.whl && \
    rm /tmp/amazonei_mxnet-1.3.0-py2.py3-none-manylinux1_x86_64.whl

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY modules /app/modules
COPY main.py /app/main.py
CMD python /app/main.py
