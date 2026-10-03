FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow<3"

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///data/mlflow.db", \
     "--default-artifact-root", "/data/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
