docker
```bash

sudo docker run -it -v /home/chaowu/00_large/01_RDK_YOLO/06_YOLO11/ultralytics:/open_explorer hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8

docker run -it -v /home/chaowu/00_Large_Storage/01_RDK_YOLO/02_YOLOv8/ultralytics/horizon_bpu:/open_explorer hub.hobot.cc/aitools/ai_toolchain_ubuntu_22_j6_gpu:v3.0.19

```

挂载本文件夹
```bash
mkdir -p ~/nj
sshfs chaowu@10.64.62.34:/home/chaowu/00_large/01_RDK_YOLO/06_YOLO11/ultralytics ~/nj
```

临时使用阿里pip源
```
-i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```
