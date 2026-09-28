from ultralytics import YOLO

# model = YOLO("../weights/yolo-ppe.pt")

model = YOLO("models/yolov8n-ppe_run_1_classes_1_2.pt")
model.export(format="ncnn")
