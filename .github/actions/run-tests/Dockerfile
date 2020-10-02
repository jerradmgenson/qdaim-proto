FROM ubuntu:20.04

RUN apt update
RUN apt install git -y
RUN apt install python3-pip -y
RUN pip3 install -r $GITHUB_WORKSPACE/src/ci/requirements.txt

ENTRYPOINT ["python3", "$GITHUB_WORKSPACE/src/tests/run_tests.py"]
