FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install git=1:2.25.1-1ubuntu3.1 python3-pip=20.0.2-5ubuntu1.1 r-base=3.6.3-2 build-essential=12.8ubuntu1 libcurl4-gnutls-dev=7.68.0-1ubuntu2.5 libxml2-dev=2.9.10+dfsg-5 libssl-dev=1.1.1f-1ubuntu2.3 cloc=1.82-1 libgit2-dev=0.28.4+dfsg.1-2 -y --no-install-recommends && rm -fr /var/lib/apt/lists/*

COPY .git /.git/
RUN git reset --hard
RUN git rev-parse --verify HEAD > build/commit_hash
RUN rm -fr /.git
RUN git init /

RUN pip3 install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/src
RUN R -e "install.packages('rcmdcheck')"
RUN R -e "install.packages('tibble')"
RUN R -e "install.packages('devtools')"
RUN R -e "devtools::install_version('argparser', version='0.6')"
RUN R -e "devtools::install_version('mice', version='3.12.0')"
RUN R -e "devtools::install_version('naniar', version='0.6.0')"

ENTRYPOINT ["scons"]
