# Container image that runs your code
FROM python:3.8

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY helloworld.py /helloworld.py

RUN pip install numpy scipy pandas sklearn coverage

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["python", "/helloworld.py"]
