# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt 

# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()
# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform 

# Hook into OpenCV
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()

    # Transform input for midas 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')#Using CPU here//CUDA can also be used here if GPU is avaiable, which will definately improve the performance

    # Make a prediction
    with torch.no_grad(): 
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2], 
            mode='bicubic', 
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

        print(output)
    
  
    
    plt.title('Depth Estimation Using Monecular Camera', 
                                     fontweight ="bold")    
    plt.imshow(output)
    cv2.imshow('Raw Frames From Camera', frame)
    plt.pause(0.00001)

    if cv2.waitKey(10) & 0xFF == ord('a'): 
        cap.release()
        cv2.destroyAllWindows()

plt.show()