VERSION 0.6

base-python:
    FROM python:3.8-slim
    ENV PIP_CACHE_DIR /pip-cache

requirements:
    FROM +base-python
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        pip install poetry
    COPY pyproject.toml poetry.lock .
    RUN poetry export -f requirements.txt -E tf -o requirements.txt
    RUN poetry export -f requirements.txt -E tf --with dev -o dev-requirements.txt
    RUN poetry export -f requirements.txt -E tf --with docs -o docs-requirements.txt
    SAVE ARTIFACT requirements.txt /requirements.txt
    SAVE ARTIFACT dev-requirements.txt /dev-requirements.txt
    SAVE ARTIFACT docs-requirements.txt /docs-requirements.txt

build:
    FROM +base-python
    
    WORKDIR /app

    COPY +requirements/requirements.txt .
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        pip install -r requirements.txt
    
    COPY tfga tfga
    ENV PYTHONPATH /app:$PYTHONPATH

test:
    FROM +build
    
    COPY +requirements/dev-requirements.txt .
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        pip install -r dev-requirements.txt

    COPY tests tests
    RUN pytest tests

publish:
    FROM +base-python

    ARG --required REPOSITORY

    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        pip install poetry
    RUN poetry config repositories.testpypi https://test.pypi.org/legacy/
    COPY pyproject.toml poetry.lock README.md .
    COPY tfga tfga
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        --secret PYPI_TOKEN=+secrets/PYPI_TOKEN \
        poetry publish \
            --build --skip-existing -r $REPOSITORY \
            -u __token__ -p $PYPI_TOKEN

docs:
    FROM +build

    COPY +requirements/docs-requirements.txt .
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        pip install -r docs-requirements.txt

    COPY docs .

    RUN sphinx-apidoc -o _build tfga
    RUN sphinx-build -M html . _build

    SAVE ARTIFACT _build/html /html
