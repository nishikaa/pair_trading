# Base Arguements to leverage across build stages
ARG APP_DIR=/app

#############
# Build (Prepare Environment) image
#############
FROM python:3.11-slim AS build
ARG APP_DIR

# install security updates
RUN apt-get update \
    && apt-get install -y \
         curl \
         build-essential \
         libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# install poetry and add to path
ENV POETRY_VERSION=1.6.1
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH /root/.local/bin:$PATH

# change work directory for where our app will live
WORKDIR ${APP_DIR}
COPY pyproject.toml poetry.lock ./

# Copy over the venv including any symbolic links, do not install development/testing libraries when install poetry dependencies
RUN python -m venv --copies ${APP_DIR}/venv
RUN . ${APP_DIR}/venv/bin/activate && poetry install --all-extras

#############
# Deployment image
#############
FROM python:3.11-slim as run
ARG APP_DIR

COPY --from=build ${APP_DIR}/venv ${APP_DIR}/venv/
ENV PATH ${APP_DIR}/venv/bin:$PATH

WORKDIR ${APP_DIR}/
COPY  . ./

# Run service
CMD ["uvicorn", "mlapi.main:app", "--host", "0.0.0.0", "--port", "8000"]