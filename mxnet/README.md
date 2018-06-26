# [MXNet](https://mxnet.apache.org/)
## How to use
1. Install [docker](https://docs.docker.com/install/)
2. Install [docker-compose](https://docs.docker.com/compose/install/)
3. Run with docker-compose
```bash
$ docker-compose up -d
```
4. Open `http://localhost:8888` in web browser
5. If you want to run on GPU, please install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and [nvidia-docker-compose](https://github.com/eywalker/nvidia-docker-compose).
  * modify `docker-compose.yml`
```diff
-    image: okwrtdsh/mxnet:latest
+    image: okwrtdsh/mxnet:gpu
```
  * Run with nvidia-docker-compose
```bash
$ nvidia-docker-compose up -d
```
