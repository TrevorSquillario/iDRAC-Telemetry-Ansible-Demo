FROM registry.access.redhat.com/ubi8/python-311
USER root
RUN mkdir "/data" && \
  chown -R default:root "/data"

USER default
WORKDIR /opt/app-root/src
ADD ml/requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt
ADD ml/log_config ./log_config
#ENTRYPOINT ["python", "ml.py"]
CMD ["fastapi", "dev", "ml.py", "--host", "0.0.0.0", "--port", "80"]