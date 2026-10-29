FROM opensora-robotwin-unit:local
COPY opensora/requirements/requirements-robodojo-test.txt /tmp/robodojo-test.txt
RUN pip install --no-cache-dir -r /tmp/robodojo-test.txt
CMD ["python", "-m", "pytest", "opensora/tests", "Rlinf/tests/unit_tests/test_world_model.py", "Rlinf/tests/unit_tests/test_robodojo.py", "-q", "--import-mode=importlib"]
