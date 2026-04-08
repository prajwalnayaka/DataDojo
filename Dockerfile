FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . .

RUN pip install --no-cache-dir .

EXPOSE 7860

ENV ENABLE_WEB_INTERFACE=True

CMD ["python", "server/app.py", "--host", "0.0.0.0", "--port", "7860"]