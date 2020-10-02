FROM ubuntu:20.04

RUN echo "GITHUB_WORKSPACE: $GITHUB_WORKSPACE"
RUN apt update
RUN apt install git -y
RUN apt install python3-pip -y
RUN git checkout https://github.com/jerradmgenson/qdaim.git
RUN pip3 install -r qdaim/src/ci/requirements.txt

ENTRYPOINT ["python3", "qdaim/src/tests/run_tests.py"]
