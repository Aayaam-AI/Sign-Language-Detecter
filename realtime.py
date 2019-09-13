import cv2
from keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def Model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3),padding = 'same', activation='relu', input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (3, 3),padding = 'same',  activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3),padding = 'same',  activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense((128),activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense((25),activation='softmax'))
    
    return model

model = Model()

model.load_weights('trained_model.h5')

def num_to_txt(n):
    if n==0:
        return 'a'
    elif n==1:
        return 'b'
    elif n==2:
        return 'c'
    elif n==3:
        return 'd'
    elif n==4:
        return 'e'
    elif n==5:
        return 'f'
    elif n==6:
        return 'g'
    elif n==7:
        return 'h'
    elif n==8:
        return 'i'
    elif n==10:
        return 'k'
    elif n==11:
        return 'l'
    elif n==12:
        return 'm'
    elif n==13:
        return 'n'
    elif n==14:
        return 'o'
    elif n==15:
        return 'p'
    elif n==16:
        return 'q'
    elif n==17:
        return 'r'
    elif n==18:
        return 's'
    elif n==19:
        return 't'
    elif n==20:
        return 'u'
    elif n==21:
        return 'v'
    elif n==22:
        return 'w'
    elif n==23:
        return 'x'
    else:
        return 'y'
    
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame,(0,200),(300,500),(255,0,0),4)
    frame1 = frame[200:500,0:300]
    cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame1 = frame1[:,:,2:]
    frame1 = cv2.resize(frame1,(28,28))
    frame1 = frame1.reshape(1,28,28,1)
    num = model.predict_classes(frame1)
    txt = num_to_txt(num)
    cv2.putText(frame,txt,org=(100,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale= 5,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()