"""A demo which runs object detection on camera frames.

export TEST_DATA=/usr/lib/python3/dist-packages/edgetpu/test_data

Run face detection model:
python3 -m edgetpuvision.detect \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 -m edgetpuvision.detect \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

Press Q key to exit.

"""
import cv2
from PIL import Image
import argparse
import os
from pose_engine import PoseEngine
import time

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    args = parser.parse_args()

    default_model = 'models/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model, mirror=args.mirror)


    cap = cv2.VideoCapture(0)

    last_time = time.monotonic()
    n = 0
    sum_fps = 0
    sum_process_time = 0
    sum_inference_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        pil_im = Image.fromarray(cv2_im)

        start_time = time.monotonic()
        outputs, inference_time = engine.DetectPosesInImage(pil_im)
        end_time = time.monotonic()
        n += 1
        sum_fps += 1.0 / (end_time - last_time)
        sum_process_time += 1000 * (end_time - start_time) - inference_time
        sum_inference_time += inference_time
        last_time = end_time

        text_line = 'PoseNet: %.1fms Frame IO: %.2fms TrueFPS: %.2f Nposes %d' % (
            sum_inference_time / n, sum_process_time / n, sum_fps / n, len(outputs)
        )
        print(text_line)

        #cv2_im = append_objs_to_img(cv2_im, objs, labels)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
