# Face detection models

[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) by Shiqi Yu, MIT License (see `LICENSE`).
The script uses these models on the Pi to skip frames that have no face, so only useful frames are uploaded to Immich.

| File | Used with |
|---|---|
| `face_detection_yunet_2022mar.onnx` | OpenCV 4.5.4 – 4.7 (Debian 12 / Raspberry Pi OS Bookworm ships 4.6) |
| `face_detection_yunet_2023mar.onnx` | OpenCV 4.8 and newer |
