FROM ubuntu:20.04

LABEL maintainer="cefl-project"
LABEL description="Defects4J image with required AWT/font native libs for CEFL"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN apt-get update -y && \
    apt-get install -y \
      openjdk-11-jdk \
      git \
      build-essential \
      subversion \
      perl \
      curl \
      unzip \
      cpanminus \
      make \
      libfreetype6 \
      libfontconfig1 \
      libxrender1 \
      libxext6 \
      libx11-6 && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="/defects4j/framework/bin:${PATH}"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY defects4j /defects4j

WORKDIR /defects4j
RUN cpanm --installdeps .
RUN ./init.sh

RUN mkdir -p /workspace /gzoltar
WORKDIR /workspace

# Smoke-test: verify defects4j CLI works
RUN defects4j info -p Lang >/dev/null

CMD ["/bin/bash"]
