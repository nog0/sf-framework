FROM python:3.11.10-bullseye

ENV TZ=Etc/UTC
ENV PYTHONPATH=/usr/lib/libreoffice/program:/usr/lib/python3/dist-packages

RUN sed -i 's#http://deb.debian.org/debian-security#https://archive.debian.org/debian-security#g' /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libreoffice-calc libreoffice-script-provider-python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openpyxl numpy pandas matplotlib xlsxwriter tqdm

# Each task/scoring container has its own writable /tmp and a read-only root.
ENV UserInstallation=file:///tmp/spreadsheetbench2-lo-profile
