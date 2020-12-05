FROM ubuntu:20.04
COPY . /

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install git python3-pip -y
RUN pip3 install -r requirements.txt
ENV PYTHONPATH=/src
RUN apt install r-base build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev cloc -y
RUN R -e "install.packages('rcmdcheck')"
RUN R -e "install.packages('devtools')"
RUN R -e "devtools::install_version('argparser', version='0.6')"
RUN R -e "devtools::install_version('lintr', version='2.0.1')"
RUN R -e "devtools::install_version('mice', version='3.12.0')"

ENTRYPOINT ["scons"]
