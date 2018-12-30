FROM python:3.6

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY modules /app/modules
COPY main.py /app/main.py
CMD python /app/main.py
