FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY switchyard ./switchyard
RUN pip install --no-cache-dir .

COPY config.example.yaml ./config.example.yaml
ENV SWITCHYARD_CONFIG=/app/config.yaml \
    PORT=8080
EXPOSE 8080

RUN useradd --create-home app && chown -R app /app
USER app

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/healthz')"
CMD ["python", "-m", "switchyard"]
