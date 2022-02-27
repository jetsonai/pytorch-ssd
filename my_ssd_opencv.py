# jetsonai.learning@gmail.com
# 20220120

gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)224, height=(int)224, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

import cv2
import numpy as np
#1. 시스템, opencv, 그리고 하위 디렉토리 vision에 있는 ssd 패키지 임포트
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

#2.  opencv, system 패키지 임포트
import cv2
import sys

#3. 객체인식 결과를 이미지에 표시하는 함수
def imageProcessing(frame, predictor, class_names):
    # 추론을 위해 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 모델에 이미지 파일을 입력하여 추론
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    # 추론 결과인 박스와 클래스 명을 이미지에 표시
    for i in range(boxes.size(0)):
        # 신뢰도 0.5 이상의 박스만 표시
        if(probs[i]>0.5):
            # 바운딩박스 표시			
            box = boxes[i, :]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(frame, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 폰트크기
                (255, 0, 255),
                2)  # 선의 유형


    return frame

#4. 영상 딥러닝 프로세싱 함수
def videoProcess(openpath, net_type, model_path, label_path):
    # 라벨 파일을 읽어 클래스 이름들 세팅
    class_names = [name.strip() for name in open(label_path).readlines()]

    # 모델 파일을 읽어 모델 로딩
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)

    net.load(model_path)

    # 네트워크 지정
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

    # 카메라나 영상으로부터 이미지를 갖고오기 위해 연결 열기
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # 영상보여주기 위한 opencv 창 생성
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)

    try:
        while cap.isOpened():
            # 이미지 프레임 읽어오기
            ret, frame = cap.read()
            if ret: 
                # 이미지 프로세싱 진행한 후 그 결과 이미지 보여주기			
                result = imageProcessing(frame, predictor, class_names, f)
                cv2.imshow("Output", result)
            else:
                break

            if cv2.waitKey(int(1000.0/120)) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:  
        print("key int")
        all_stop()
        cap.release()
        cv2.destroyAllWindows()
        return
    # 프로그램 종료 후 사용한 리소스를 해제한다.
    cap.release()

    cv2.destroyAllWindows()

    return
   
#5. 인자가 4보다 작으면 종료, 인자가 4면 카메라 추론 시작, 인자가 4보다 크면 영상파일 추론
if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
    sys.exit(0)

if len(sys.argv) == 4:
    gst_str = ("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=480,format=BGR ! appsink")
    print("camera 0")

else:
    gst_str = sys.argv[4]
    print(gst_str)

net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

#6. 영상 딥러닝 프로세싱 함수 호출
videoProcess(gst_str, net_type, model_path, label_path)
