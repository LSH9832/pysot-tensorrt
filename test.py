import camera as cv2
import pysot
import time
import argparse

"""
    source = '0'
    source = '/home/lsh/Videos/car6.avi'
    source = ['/home/lsh/Pictures/car6/*1.jpg', '/home/lsh/Pictures/car6/*6.jpg']
"""

# source = "E:/videos/test.avi"
source = "E:/dataset/UAV123/data_seq/UAV123/car6/*.jpg"
model_type = "siamrpn_mobilev2_l234_dwxcorr"
double_eye = False
fps_limit = 90
use_trt = False


def get_args():
    parser = argparse.ArgumentParser("SiamRPN test parser")
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--name", type=str, default=model_type)
    parser.add_argument("--max-fps", type=int, default=fps_limit, help="max display fps")
    parser.add_argument("--trt", action="store_true", help="use tensorrt model")
    parser.add_argument("--bbox", type=int, nargs="+", default=None)
    parser.add_argument("--test-frame", type=int, default=None)

    return parser.parse_args()

# test on RTX3060, using images in dataset UAV123 car6

# |             Name              | FPS tensorrt | FPS pytorch |
# |:------------------------------|:-------------|:------------|
# | siamrpn_alex_dwxcorr_otb      | 128.49 FPS   |  78.54 FPS  |
# | siamrpn_mobilev2_l234_dwxcorr | 80.59  FPS   |  72.91 FPS  |
# | siamrpn_r50_l234_dwxcorr      |        FPS   |        FPS  |(not good, skip)

def main(args):
    dt_limit = 1. / args.max_fps

    cam = cv2.OpenSource(
        source=args.source,
        double_eyes=double_eye
    )
    cam.setsize(1280, 720) if double_eye else None

    tracker = pysot.Tracker(
        model_name=args.name,
        trt=args.trt
    )

    first_frame = True

    max_fps = 0
    all_dt = []
    delay = 1

    while cam.isOpened():

        t0 = time.time()
        if double_eye:
            success, frame_left, frame_right = cam.readall()
            """
            success, frame_left = cam.readleft()
            success, frame_right = cam.readright()
            """
            frame = frame_left  # frame_right
        else:
            success, frame = cam.read()

        if success:
            if first_frame:
                first_frame = False

                if args.bbox is not None and len(args.bbox) == 4:
                    bbox = args.bbox
                else:
                    bbox = cv2.selectROI(
                        windowName='select roi',
                        img=frame,
                        showCrosshair=True,
                        fromCenter=False
                    )

                # bbox = (508, 180, 139, 110)
                print("init bbox:", bbox)

                tracker.set_boundingbox(
                    image=frame,
                    boundingbox=bbox
                )
                cv2.destroyWindow('select roi')

            else:
                cv2.fps(1)
                output = tracker.update(frame.copy())
                fps2 = cv2.fps(1)
                # print(output)

                frame_show = tracker.draw_boundingbox(frame, output, color=(128, 255, 30))

                cv2.imshow('result', frame_show)
                key = cv2.waitKey(delay)
                if key == 27:
                    print()
                    break

                elif key == ord(" "):
                    delay = 1 - delay

                fps1 = cv2.fps(0)
                max_fps = max(max_fps, fps2)
                all_dt.append(1./fps2)

                if args.test_frame is not None:
                    if len(all_dt) % args.test_frame == 0:
                        delay = 1 - delay

                print('\rReal FPS:%.2f  Inference FPS:%.2f  Confidence:%.3f' % (fps1, fps2, output['best_score']), end='')
                while time.time() - t0 < dt_limit:
                    pass
        else:
            break

    print(f"\n\nmax fps: {max_fps}, average fps: {len(all_dt) / sum(all_dt)}")

    cv2.destroyAllWindows()
    cam.release()


if __name__ == '__main__':
    main(get_args())
