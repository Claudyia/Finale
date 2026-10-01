from ultralytics import YOLO

# model = YOLO("../weights/yolo-ppe.pt")

model = YOLO("yolov8n_run_1.2_classes_1_2.pt")
model.export(format="ncnn")
