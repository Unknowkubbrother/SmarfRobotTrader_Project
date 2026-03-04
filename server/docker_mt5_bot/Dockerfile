FROM ghcr.io/linuxserver/baseimage-kasmvnc:debianbookworm

# set version label
ARG BUILD_DATE
ARG VERSION
LABEL build_version="Metatrader Docker:- ${VERSION} Build-date:- ${BUILD_DATE}"
LABEL maintainer="gmartin"

ENV TITLE=Metatrader5
ENV WINEPREFIX="/config/.wine"
ENV WINEDEBUG=-all

# Install all packages in a single layer to reduce image size.
# Keep Debian wine packages for qemu/Apple Silicon compatibility.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    xdotool \
    gnupg2 \
    software-properties-common \
    ca-certificates \
    && dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install --install-recommends -y wine wine64 libwine unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python dependencies for Linux environment
# Pin rpyc to 5.2.3 to match mt5linux 0.2.4 on Windows side
RUN pip3 install --break-system-packages --no-cache-dir --no-deps mt5linux==0.2.4 \
    && pip3 install --break-system-packages --no-cache-dir rpyc==5.2.3 plumbum==1.7.0 "pyparsing>=3.1.0,<4" numpy pyxdg pyzmq

# Pre-download installers to cache them in the image
RUN mkdir -p /defaults/installers \
    && curl -o /defaults/installers/wine-mono-10.3.0-x86.msi https://dl.winehq.org/wine/wine-mono/10.3.0/wine-mono-10.3.0-x86.msi \
    && curl -o /defaults/installers/python-3.11.9-embed-amd64.zip https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip \
    && curl -o /defaults/installers/mt5setup.exe https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe

COPY /Metatrader /Metatrader
RUN chmod +x /Metatrader/start.sh
COPY /root /

# Fix KasmVNC auth generation when PASSWORD starts with '-' (treat as value, not option).
RUN sed -i 's/openssl passwd -apr1 ${PASSWORD}/openssl passwd -apr1 -- "${PASSWORD}"/' /etc/s6-overlay/s6-rc.d/init-nginx/run

EXPOSE 3000 8001
VOLUME /config
