from ultralytics import YOLO
from keras.utils import load_img
import numpy as np
import cv2
import matplotlib.pyplot as plt
# path="./backend/test/c7.jpg"
# image=cv2.imread(path)

# model1Labels={0:'single_number_plate',1:'double_number_plate'}

model2Labels={ 0: '0',1: '1',2: '2',3: '3',4: '4',5: '5',6: '6',7: '7',8: '8',9: '9',10: 'A',11: 'B',12: 'C',13: 'D',14: 'E',15: 'F',16: 'G',17: 'H'
              ,18: 'I',19: 'J',20: 'K',21: 'L',22: 'M',23: 'N',24: 'O',25: 'P',26: 'Q',27: 'R',28: 'S',29: 'T',30: 'U',31: 'V',32: 'W',33: 'X',34: 'Y',35: 'Z'}


model=YOLO("./backend/runs/detect/train2/weights/best_k1.pt")
model2=YOLO("./backend/runs/detect/train5/weights/best_k.pt")
def prediction(path,filename):
    image=cv2.imread(path)

    cv2.imwrite('./backend/static/upload/'+filename,image)
    cv2.imwrite('./alpr/public/assets/upload/'+filename,image)

    result=model.predict(source=image,conf=0.5)
   
    boxes = result[0].boxes
    height=boxes.xywh
    crd=boxes.data
    print(crd)
    
    n=len(crd)
    ht=[]
    lp_number=[]
    for i in range (0,n):
        ht=int(height[i][3])
        c=int(crd[i][5])
        
        min=(int(crd[i][0]),int(crd[i][1]))
        max=(int(crd[i][2]),int(crd[i][3]))
        xmin=min[0]
        ymin=min[1]
        xmax=max[0]
        ymax=max[1]
        img=load_img(path)
        img=np.array(img,dtype=np.uint8)
        img_lp=img[ymin:ymax,xmin:xmax]
        cv2.rectangle(img,min,max,(0,255,0),2)
        cv2.imwrite('./backend/static/predict_numberplate/'+str(i)+filename,img)
        cv2.imwrite('./backend/static/plates/'+str(i)+filename,img_lp)

        h=np.median(ht)

        #SECOND MODEL
        result2=model2.predict(source=img_lp,conf=0.3)
        boxes_ocr=result2[0].boxes
        data2=boxes_ocr.data
        print(data2)

        n2=len(data2)
        
        xaxis0=[]
        xaxis11=[]
        xaxis12=[]
        yaxis=[]
        yaxis2=[]
        label0=[]
        label11=[]
        label12=[]
        numberPlate=""
        if(c==0):
            for i in range(0,n2):
                    x=int(data2[i][2])
                    xaxis0.append(x)
                    l=int(data2[i][5])
                    label0.append(l)

            for i in range(0,n2-1):
                for j in range(i+1,n2):
                    if(xaxis0[i]>xaxis0[j]):
                        temp=xaxis0[i]
                        xaxis0[i]=xaxis0[j]
                        xaxis0[j]=temp  

                        temp=label0[i]
                        label0[i]=label0[j]
                        label0[j]=temp
            for i in range(0,len(label0)):
                numberPlate=numberPlate+(model2Labels.get(label0[i]))
            lp_number.append(numberPlate)


        elif(c==1):
            for i in range(0,n2):
                x=int(data2[i][0])
                y=int(data2[i][3])
                l=int(data2[i][5])
                if(y<(h/2)):
                    xaxis11.append(x)
                    yaxis.append(y)
                    label11.append(l)
                else:
                    xaxis12.append(x)
                    yaxis2.append(y)
                    label12.append(l)  
            for i in range(0,len(xaxis11)-1):
                for j in range(i+1,len(xaxis11)):
                    if(xaxis11[i]>xaxis11[j]):
                        temp=xaxis11[i]
                        xaxis11[i]=xaxis11[j]
                        xaxis11[j]=temp

                        temp=label11[i]
                        label11[i]=label11[j]
                        label11[j]=temp
            for i in range(0,len(xaxis12)-1):
                for j in range(i+1,len(xaxis12)):
                    if(xaxis12[i]>xaxis12[j]):
                        temp=xaxis12[i]
                        xaxis12[i]=xaxis12[j]
                        xaxis12[j]=temp

                        temp=label12[i]
                        label12[i]=label12[j]
                        label12[j]=temp
            for i in range(0,len(label11)):
                numberPlate=numberPlate+(model2Labels.get(label11[i]))
            for i in range(0,len(label12)):
                numberPlate=numberPlate+(model2Labels.get(label12[i]))
            # print(numberPlate)
            lp_number.append(numberPlate)
            
    return lp_number

# print(prediction(path,"c7.jpg"))