# hash:sha256:d2f2a1c27203b19cd9c377aef7e8d4161a5641062668ee5ea693d8bf349d8a15
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab-py39:1.0

ARG DEBIAN_FRONTEND=noninteractive
ARG API_KEY
ARG API_SECRET
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib==3.8.2 \
    pandas==2.0.0 \
    scikit-learn==1.3.2 \
    seaborn==0.13.0

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
