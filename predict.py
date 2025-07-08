import sys
sys.path.insert(0, r"C:\Project\YOLOv8-multi-task-main\ultralytics\__init__.py")

from ultralytics import YOLO


number = 4 #input how many tasks in your work
model = YOLO(r'C:\Project\YOLOv8-multi-task-main\ultralytics\runs\v4s.pt')  # Validate the model
model.predict(source=r'C:\Project\YOLOv8-multi-task-main\ultralytics\samples_resized', imgsz=(384,672), device='cpu',name='v4_daytime', save=True, conf=0.25, iou=0.45, show_labels=False)
