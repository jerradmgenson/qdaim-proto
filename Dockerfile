FROM ubuntu:20.04
COPY . /

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install git python3-pip=20.0.2-5ubuntu1.1 r-base=3.6.3-2 build-essential=12.8ubuntu1 libcurl4-gnutls-dev=7.68.0-1ubuntu2.4 libxml2-dev=2.9.10+dfsg-5 libssl-dev=1.1.1f-1ubuntu2.2 cloc=1.82-1 libgit2-dev=0.28.4+dfsg.1-2 -y --no-install-recommends
RUN rm -fr /var/lib/apt/lists/*
RUN pip3 install -r requirements.txt
ENV PYTHONPATH=/src
RUN R -e "install.packages('rcmdcheck')"
RUN R -e "install.packages('tibble')"
RUN R -e "install.packages('devtools')"
RUN R -e "devtools::install_version('argparser', version='0.6')"
RUN R -e "devtools::install_version('lintr', version='2.0.1')"
RUN R -e "devtools::install_version('mice', version='3.12.0')"
RUN R -e "devtools::install_version('naniar', version='0.6.0')"

ENTRYPOINT ["scons"]
