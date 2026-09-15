FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a dedicated non-root application user.
RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home appuser

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY alembic.ini .
COPY alembic ./alembic
COPY docker-entrypoint.sh .

RUN chmod +x docker-entrypoint.sh \
    && chown -R appuser:appuser /app

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]