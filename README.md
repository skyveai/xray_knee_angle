# Quick Start
1. Download repository

2. Build docker image
  ```bash
  docker build --no-cache -t xray-angle-api .
  ```

3. Run docker container
  ```bash
  docker run --rm -p 8877:8877 --gpus all -v $(pwd):/app -w /app -d xray-angle-api
  ```
