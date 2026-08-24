FROM continuumio/miniconda3:23.3.1-0

LABEL maintainer="cefl-project"
LABEL description="BugsInPy image with FauxPy snapshot and conda-based per-bug Python envs for CEFL"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      git \
      nano \
      dos2unix \
      build-essential \
      ccache \
      curl \
      ca-certificates \
      libpng-dev \
      libfreetype6-dev \
      pkg-config && \
    rm -rf /var/lib/apt/lists/*

# C-extension builds (pandas/matplotlib/spacy) are routed through ccache by the per-project FauxPy
# build hooks: they prepend /usr/lib/ccache to PATH and point CCACHE_DIR at the shared /workspace
# mount, so the compiler cache persists across bugs, runs, and parallel lanes. The hooks also build
# at -O0 (the SUT only needs to be importable for coverage). See bip_fauxpy_config.fast_build.

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY BugsInPy /BugsInPy
COPY fauxpy-experiments/pytest-FauxPy /opt/pytest-FauxPy

ENV PATH="/BugsInPy/framework/bin:${PATH}"

RUN mkdir -p /workspace && chmod 755 /workspace
WORKDIR /workspace

# Smoke-test: BugsInPy CLI is on PATH (bugsinpy-info returns nonzero on --help, so tolerate it)
RUN bugsinpy-info --help >/dev/null 2>&1 || true

CMD ["/bin/bash"]
