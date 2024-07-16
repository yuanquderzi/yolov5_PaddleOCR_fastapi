from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import numpy as np
from paddleocr import PaddleOCR
import io
import json

app = FastAPI()

# 初始化PaddleOCR模型
ocr_model = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)

# 加载YOLOv5模型
model_path = 'best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.to(device)

def detectx(frame, model):
    frame = [frame]
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates


def recognize_plate_easyocr(img, coords):
    xmin, ymin, xmax, ymax = coords
    #nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-10:int(xmax)+10]  # 裁剪出车牌区域
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]

    ocr_result = ocr_model.ocr(nplate)

    if not ocr_result or not ocr_result[0]:
        return ""

    text = ""
    for line in ocr_result[0]:
        text += line[1][0] + " "
    return text


def plot_boxes(results, frame, classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    # 将 BGR 转为 RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("simfang.ttf", 20)  # 选择一个支持中文的字体

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.25:  # 检测阈值
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            text_d = classes[int(labels[i])]
            coords = [x1, y1, x2, y2]
            plate_num = recognize_plate_easyocr(img=np.array(pil_img), coords=coords)

            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.rectangle([x1, y1-20, x2, y1], fill="green")
            draw.text((x1, y1-20), plate_num, font=font, fill="white")

    # 将 PIL 图片转回 BGR 格式
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame


@app.post("/object-to-json")
async def detect_to_json(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    results = detectx(frame, model=model)
    
    detections = []
    labels, cord = results
    for i in range(len(labels)):
        label = model.names[int(labels[i])]
        x1, y1, x2, y2 = float(cord[i][0]), float(cord[i][1]), float(cord[i][2]), float(cord[i][3])  # YOLOv5返回的坐标顺序可能需要调整
        confidence = float(cord[i][4])
        row = cord[i]
        x1_, y1_, x2_, y2_ = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        plate_number = recognize_plate_easyocr(frame, (x1_, y1_, x2_, y2_))
        detections.append({
            "label": label,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "barcode": plate_number
        })
    
    return JSONResponse(content={"detections": detections})



@app.post("/object-to-img")
async def detect_to_img(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = detectx(frame, model=model)
    frame = plot_boxes(results, frame, classes=model.names)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(frame)
    
    # 将处理后的图像转换为字节流并返回
    img_bytes = io.BytesIO()
    result_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return StreamingResponse(img_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

