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

### SHH Local Port Forwarding
 ```bash
 ssh -L 8877:192.168.0.12:8877 gpuadmin@220.76.132.178
 ```

## How to run juputer lab via docker on the server
1. Mount external disk with data to /mnt/xray2_disk
2. Go to /data/igor/projects/2d_landmark
3. Run docker run --rm -p 8889:8889 --gpus all -v $(pwd):/app -w /app -v /mnt/xray2_disk/dataset:/dataset -d torchlab
4. Setup local port forwarding ssh -L 8889:192.168.0.13:8889 gpuadmin@220.76.132.178
5. Run docker logs <container_id_or_name>
6. Now you can open juputer lab in web browser

## Notebooks usage

* Use extract_to_pickle[M] notebook to extract image paths and keypoint coordinates
* Use train_roi[xray_2] notebook to train ROI model
* Use train_hip[xray_2] notebook to train HIP model
* Use train_knee[xray_2] notebook to train KNEE model
* Use train_ankle[xray_2] notebook to train ANKLE model
* Use calc_angle[MPTA-low] notebook to calculate MPTA angle with low points
* Use calc_angle[MPTA-mid] notebook to calculate MPTA angle with mid points
* Use calc_angle[MPTA-edge] notebook to calculate MPTA angle with edge points
* Use calc_angle[MPTA-6-points] notebook to calculate MPTA angle with 6-points
* Use calc_angle[MPTA-8-points] notebook to calculate MPTA angle with 8-points
