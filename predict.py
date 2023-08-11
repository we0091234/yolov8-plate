from ultralytics import YOLO
from  PIL import Image
from ultralytics.nn.tasks import  attempt_load_weights

# Load a model
model = YOLO('runs/pose/train4/weights/best.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model('h_0_008396.jpg')  # predict on an image
for r in results:
    print(r.boxes)
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    im.save('result.jpg')  # save imagel9