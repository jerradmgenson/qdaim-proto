FROM ubuntu:20.04
COPY $GITHUB_WORKSPACE /

RUN echo "GITHUB_WORKSPACE: $GITHUB_WORKSPACE"
RUN apt update
RUN apt install git -y
RUN apt install python3-pip -y
RUN pip3 install -r src/ci/requirements.txt

ENTRYPOINT ["python3", "src/tests/run_tests.py"]
