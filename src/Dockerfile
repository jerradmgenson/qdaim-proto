# Container image that runs your code
FROM ubuntu:20.04

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY ./ /src

RUN apt update
RUN apt install git -y
RUN apt install python3-pip -y
RUN pip3 install -r src/ci/requirements.txt

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["python", "src/tests/run_tests.py"]
