# Container image that runs your code
FROM python:3.8

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY src /src

RUN pip install -r src/ci/requirements.txt

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["python", "src/ci/helloworld.py"]
