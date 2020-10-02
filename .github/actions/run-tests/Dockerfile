FROM ubuntu:20.04

RUN apt update
RUN apt install git -y
RUN apt install python3-pip -y
RUN pip3 install -r /github/workspace/src/ci/requirements.txt

ENTRYPOINT ["python3", "/github/workspace/src/tests/run_tests.py"]
