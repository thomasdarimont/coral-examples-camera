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
import argparse
from pose_engine import PoseEngine
import time
import numpy as np

class Cv2PosenetExample:

    def __init__(self, args):
        self.args = args

        default_model = 'models/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
        if args.res == '480x360':
            self.src_size = (640, 480)
            self.appsink_size = (480, 360)
            self.model = args.model or default_model % (353, 481)
        elif args.res == '640x480':
            self.src_size = (640, 480)
            self.appsink_size = (640, 480)
            self.model = args.model or default_model % (481, 641)
        elif args.res == '1280x720':
            self.src_size = (1280, 720)
            self.appsink_size = (1280, 720)
            self.model = args.model or default_model % (721, 1281)

        # see https://www.pyimagesearch.com/2016/04/25/watermarking-images-with-opencv-and-python/
        self.noseImage = self.load_image("media/Nose.png")

    def load_image(self, imagePath):
        img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
        (B, G, R, A) = cv2.split(img)
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)
        convertedImage = cv2.merge([B, G, R, A])
        return convertedImage

    def run(self):
        print('Loading model: ', self.model)
        engine = PoseEngine(self.model, mirror=self.args.mirror)

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

            # pil_im = Image.fromarray(cv2_im)

            start_time = time.monotonic()
            outputs, inference_time = engine.DetectPosesInImage(cv2_im)
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

            for pose in outputs:
                self.draw_pose(cv2_im, pose)

            cv2.imshow('frame', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_pose(self, cv2TargetImage, pose, color=(0,255,255), threshold=0.2):

        (h, w) = cv2TargetImage.shape[:2]

        image = np.dstack([cv2TargetImage, np.ones((h, w), dtype="uint8") * 255])

        # construct an overlay that is the same size as the input
        # image, (using an extra dimension for the alpha transparency),
        # then add the watermark to the overlay in the bottom-right
        # corner
        overlay = np.zeros((h, w, 4), dtype="uint8")
        (wH, wW) = self.noseImage.shape[:2]
        overlay[h - wH - 10:h - 10, w - wW - 10:w - 10] = self.noseImage

        # blend the two images together using transparent overlays
        cv2TargetImage = cv2TargetImage.copy()
        cv2.addWeighted(overlay, 1.0, cv2TargetImage, 1.0, 0, cv2TargetImage)

        xys = {}
        for label, keypoint in pose.keypoints.items():
            if keypoint.score < threshold: continue
            xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))

            cv2.circle(cv2TargetImage, (int(keypoint.yx[1]), int(keypoint.yx[0])), radius=5, color=color, thickness=-1)
            # dwg.add(dwg.circle(center=(int(keypoint.yx[1]), int(keypoint.yx[0])), r=5, fill='cyan', fill_opacity=keypoint.score, stroke=color))

        xysNose = xys.get("nose")
        xysLeftEar = xys.get("left ear")
        xysRightEar = xys.get("right ear")
        if not (xysNose is None or xysLeftEar is None or xysRightEar is None):
            xLeftEar = xysLeftEar[1]
            xRightEar = xysRightEar[1]
            dxEars = abs(xLeftEar - xRightEar)
            cv2.circle(cv2TargetImage, (xysNose[0], xysNose[1]), radius=int(dxEars/1.4), color=color, thickness=-1)


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    args = parser.parse_args()

    cv2Posenet = Cv2PosenetExample(args)
    cv2Posenet.run()

if __name__ == '__main__':
    main()
