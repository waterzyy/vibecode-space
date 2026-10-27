#!/usr/bin/env bash
# Однократная команда запуска: создаёт .env и venv, ставит зависимости и поднимает сервер.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Создан файл .env. Впишите в него GEMINI_COOKIE_1PSID (и GEMINI_COOKIE_1PSIDTS), затем запустите ./run.sh снова."
    exit 1
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

exec .venv/bin/python server.py
