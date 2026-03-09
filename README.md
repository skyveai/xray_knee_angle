Download repository
Build docker image : docker build --no-cache -t xray-angle-api .
Run docker container: docker run --rm -p 8877:8877 --gpus all -v $(pwd):/app -w /app -d xray-angle-api
