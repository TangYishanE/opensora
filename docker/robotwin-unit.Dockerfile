ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.11-slim@sha256:e9e0bd902ae57c30d3ffdb7019b33785cee3c7860dc1eda2dfd740354911a82c
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
COPY opensora/requirements/requirements-robotwin-test.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps --no-build-isolation colossalai==0.5.0
WORKDIR /workspace
ENV PYTHONPATH=/workspace/opensora:/workspace/Rlinf
CMD ["python", "-m", "pytest", "opensora/tests", "Rlinf/tests/unit_tests/test_world_model.py", "-q", "--import-mode=importlib"]
