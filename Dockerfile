FROM python:3.9-slim


WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 80


ENV GOOGLE_API_KEY=AIzaSyCP1kveVOTOIMyzvEY6Xdwpq18567ETBPU


CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.enableCORS=false"]
