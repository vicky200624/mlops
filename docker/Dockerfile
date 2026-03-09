# base image
FROM python:3.11-slim-buster

# workdir
WORKDIR /app

#copy
COPY . /app

#run
RUN pip install -r requirements.txt

#port
EXPOSE 8080

# commands
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]