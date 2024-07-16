## **一、项目简介**
使用FastAPI构建的燃气表表号检测识别后端服务。首先利用yolov5来定位燃气表条形码区域，然后使用PaddleOCR提取条形码区域内的燃气表表号。yolov5已在燃气表图像上进行了定制化训练，检测识别准确率98%。

**识别效果：**

![条形码检测](assets/test1.jpg)

![条形码检测](assets/test2.jpg)

操作流程demo

![条形码检测](assets/动画.gif)

## **二、快速开始**
**程序运行方式：**

 - 运行api.py
 - 要访问生成的服务的 FastAPI 文档，请使用 Web 浏览器访问 http://localhost:8001/docs

## **三、联系方式**
如需要燃气表条形码检测权重best.pt，请联系

<div align=center>
<img src="assets/wechat.jpg" width="50%">
</div>