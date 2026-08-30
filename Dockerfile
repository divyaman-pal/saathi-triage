# Hugging Face Spaces deployment.
# HF removed the native Streamlit SDK, so the app ships as a container. This is
# also closer to how SAATHI is meant to run in a hospital: a single edge box on
# the ED network, no cloud dependency on the decision path.
FROM python:3.11-slim

# HF Spaces runs containers as uid 1000. The app writes its audit store at
# runtime, so the working tree must be owned by that user.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# XSRF protection is disabled only because HF terminates TLS at its own proxy
# and rewrites the origin; the app holds no credentials and writes no real data.
CMD ["streamlit", "run", "saathi/ui/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
