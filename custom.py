import torch
import cv2
import time
import numpy as np
from midas.model_loader import default_models, load_model

show_size = {'x':1920, 'y':1080}

def depth_process(device, model, model_type, image, input_size, target_size, optimize, use_camera, transform):
    image = transform({"image": image/255})["image"]
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    prediction = model.forward(sample)
    prediction = (torch.nn.functional.interpolate(prediction.unsqueeze(1),size=target_size[::-1],mode="bicubic",align_corners=False,).squeeze().cpu().detach().numpy())
    return output_img_process(prediction)

def output_img_process(depth):
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3
    depth_img = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    depth_img /= 255
    return depth_img


# 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = 'dpt_swin2_large_384'
model_path = f'C:/Users/Lion/Desktop/MiDaS_custom/{model_type}.pt'
optimize = False
height = None
square = False
model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)


# 웹캠 설정
while True:
    print('웹캠 연결 중...')
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, show_size['x'])
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, show_size['y'])
    time.sleep(0.3)
    ret, _ = cap.read()
    if ret == True: break

while True:
    # 이미지 읽고 전처리
    ret, img = cap.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start = time.time()
    # 추론, depth 이미지 후처리
    depth_img = depth_process(device, model, model_type, rgb_img, (net_w, net_h), rgb_img.shape[1::-1], optimize, True, transform)
    fps = round(1 / (time.time()-start+0.000001), 2)
    print(f'FPS : {fps}')

    cv2.imshow('window', depth_img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
