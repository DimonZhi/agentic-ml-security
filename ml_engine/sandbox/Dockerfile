# Образ песочницы ML Engine: один и тот же для обучения и для оценки.
# Версии зафиксированы в requirements.lock (полный pip freeze). Агент пакеты не ставит:
# контейнер запускается без сети и с файловой системой только на чтение.
#
# Сборка:  docker build -t ml-engine-sandbox:1 sandbox/
# Если Docker Hub недоступен: --build-arg BASE_IMAGE=public.ecr.aws/docker/library/python:3.11
# Если медленный PyPI:        --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1

# libgomp нужен LightGBM; в полном образе python он уже есть
RUN if ! ldconfig -p | grep -q libgomp.so.1; then \
      apt-get update && apt-get install -y --no-install-recommends libgomp1 \
      && rm -rf /var/lib/apt/lists/*; fi

ARG PIP_INDEX_URL
COPY requirements.lock /tmp/requirements.lock
# Секрет pipcert нужен только за корпоративным прокси со своим CA; без него шаг работает как обычно.
RUN --mount=type=secret,id=pipcert,required=false \
    if [ -f /run/secrets/pipcert ]; then export PIP_CERT=/run/secrets/pipcert; fi; \
    pip install -r /tmp/requirements.lock && rm /tmp/requirements.lock

RUN useradd --uid 10001 --no-create-home sandbox
USER 10001
WORKDIR /work
