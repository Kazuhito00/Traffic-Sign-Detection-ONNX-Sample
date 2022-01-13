# Traffic-Sign-Detection-ONNX-Sample
[aarcosg/traffic-sign-detection](https://github.com/aarcosg/traffic-sign-detection) の Faster RCNN ResNet50をONNXに変換して推論するサンプルです。<br>
ONNX変換は[Traffic-Sign-Detection-ONNX-Sample](Traffic-Sign-Detection-ONNX-Sample)をColaboratoryで実行しています。<br><br>

このモデルはGTSDB(German Traffic Sign Detection Benchmark)で訓練されたモデルのため、ドイツ向けです。<br>
以下は日本の道路ですが、当然検出されているクラスIDはあっていません。<br>

https://user-images.githubusercontent.com/37477845/149354209-843c6077-0484-4aee-ab83-dddf5b6f9dfa.mp4

# Requirement 
opencv-python 4.5.3.56 or later<br>
onnxruntime-gpu 1.9.0 or later<br>
※onnxruntime-gpuはonnxruntimeでも動作しますが、推論時間がかかるためGPUを推奨します<br>

# ONNX Model
[faster_rcnn_resnet50.onnx](https://drive.google.com/u/3/uc?id=1L8XrIwZsaz4F_jt1GG_v3-Sbft9bPSzB&export=download)をダウンロードしてmodelディレクトリに置いてください。

# Demo
デモの実行方法は以下です。
#### 動画：動画に対し標識検出した結果を動画出力します
```bash
python demo_video_onnx.py
```
以下のオプション指定が可能です。
* --use_debug_window<br>
動画書き込み時に書き込みフレームをGUI表示するか否か<br>
デフォルト：指定なし
* --model<br>
ByteTrackのONNXモデル格納パス<br>
デフォルト：model/faster_rcnn_resnet50.onnx
* --video<br>
入力動画の格納パス<br>
デフォルト：sample.mp4
* --output_dir<br>
動画出力パス<br>
デフォルト：output
* --score_th<br>
検出のスコア閾値<br>
デフォルト：0.3

#### Webカメラ：Webカメラ画像に対し標識検出した結果をGUI表示します
```bash
python demo_webcam_onnx.py
```
以下のオプション指定が可能です。
* --model<br>
ByteTrackのONNXモデル格納パス<br>
デフォルト：model/faster_rcnn_resnet50.onnx
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --score_th<br>
検出のスコア閾値<br>
デフォルト：0.3

# Reference
* [traffic-sign-detection](https://github.com/aarcosg/traffic-sign-detection)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Traffic-Sign-Detection-ONNX-Sample is under [MIT License](LICENSE).
