1. Download repository
2. Build docker image : docker build --no-cache -t xray-angle-api .
3. Run docker container: docker run --rm -p 8877:8877 --gpus all -v $(pwd):/app -w /app -d xray-angle-api
