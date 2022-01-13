import copy
import time
import argparse

import cv2
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        default='model/faster_rcnn_resnet50.onnx',
    )
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.3,
    )

    args = parser.parse_args()

    return args


def main():
    # 引数取得
    args = get_args()

    model_path = args.model

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    score_th = args.score_th

    # モデル読み込み
    session = onnxruntime.InferenceSession(model_path)

    # 入出力名取得
    input_name = session.get_inputs()[0].name

    output_name_boxes = session.get_outputs()[1].name
    output_name_classes = session.get_outputs()[3].name
    output_name_scores = session.get_outputs()[2].name
    output_name_num = session.get_outputs()[0].name

    # カメラ準備
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    while True:
        start_time = time.time()

        # フレーム読み出し
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 前処理
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('uint8')

        # 推論
        result = session.run(
            [
                output_name_num, output_name_boxes, output_name_scores,
                output_name_classes
            ],
            {input_name: input_image},
        )

        num_detections = result[0]
        boxes = result[1]
        scores = result[2]
        classes = result[3]

        elapsed_time = time.time() - start_time

        # 検出情報描画
        debug_image = draw_info(
            debug_image,
            num_detections,
            boxes,
            scores,
            classes,
            score_th,
            elapsed_time,
        )

        # キー処理(ESC：終了)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映
        cv2.imshow('Traffic Sign Detection ONNX Sample', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def draw_info(
    image,
    num_detections,
    boxes,
    scores,
    classes,
    score_th,
    elapsed_time,
):
    class_name = [
        'prohibitory',
        'mandatory',
        'danger',
    ]

    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)

    for bbox, score, class_id in zip(boxes[0], scores[0], classes[0]):
        if score < score_th:
            continue

        x1, y1 = int(bbox[1] * image_width), int(bbox[0] * image_height)
        x2, y2 = int(bbox[3] * image_width), int(bbox[2] * image_height)

        cv2.putText(
            debug_image,
            str(class_name[int(class_id) - 1]) + '{:.2f}'.format(score),
            (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            cv2.LINE_AA)
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 推論時間
    text = 'elapsed time: %.0fms ' % (elapsed_time * 1000)
    cv2.putText(
        debug_image,
        text,
        (30, 30),
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (0, 255, 0),
        thickness=1,
    )
    return debug_image


if __name__ == '__main__':
    main()
