FROM  python:3.11.4-slim
WORKDIR /app
COPY . .
RUN apt-get update
RUN pip install --upgrade pip
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python","app.py"]