# ===================================================================================
# A container which builds everything needed for the queueserver so that 
# ecosystem so that respective client-side bluesky-adaptive can work.
#
# The dependencies from beamline-bayesian-playground[container] are installed on build so that
# so that it doesn't take forever on each docker-compose with the mounted volume.
# ===================================================================================

ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION} AS developer

# Add any system dependencies for the developer/build environment here
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libqt5gui5 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.ipython/profile_default/startup/

# Copy the queueserver directory
COPY ./src/queueserver /queueserver

# Copy the uv dependencies and install so that it's cached.
COPY uv.lock ./uv.lock
RUN pip install uv
RUN uv export --group container --no-dev --frozen --no-hashes --no-emit-project -o /requirements.txt
RUN rm /uv.lock
RUN uv pip install --system -r /requirements.txt
