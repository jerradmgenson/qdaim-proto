FROM ubuntu:20.04
COPY . /

RUN apt update
RUN apt install git -y
RUN apt install python3-pip -y
RUN pip3 install -r requirements.txt

ENTRYPOINT ["scons"]
