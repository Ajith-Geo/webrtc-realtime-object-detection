from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Export the model to ONNX format
# opset=12 is a good choice for web compatibility.
model.export(format='onnx', opset=12)

print("Model converted to ONNX format successfully.")
print("The converted model is saved as yolov8n.onnx")
