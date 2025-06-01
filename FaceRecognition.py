import tkinter
import PIL.Image, PIL.ImageTk
from tkinter import simpledialog
import time
from tkinter import messagebox
import os
from tkinter import *
import random
from datetime import date
from PIL import Image
import numpy as np
import cv2


class App:
    global classifier
    global labels
    global X_train
    global Y_train
    global text
    global img_canvas
    global cascPath
    global faceCascade
    global tf1, tf2
    global capture_img
    global student_details
    global recognizer
    def __init__(self, window, window_title, video_source=0):
        global cart
        global text
        cart = []
        self.window = window
        self.window.title("Face Recognition based on Local Binary Pattern Features")
        self.window.geometry("1300x1200")
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.font1 = ('times', 13, 'bold')
        self.capture_img = None

        font = ('times', 16, 'bold')
        title = Label(window, text='Face Recognition based on Local Binary Pattern Features')
        title.config(bg='darkviolet', fg='gold')  
        title.config(font=font)           
        title.config(height=3, width=120)       
        title.place(x=0,y=5)                  

        self.l1 = Label(window, text='PERSON ID: ')
        self.l1.config(font=self.font1)
        self.l1.place(x=50,y=500)

        self.tf1 = Entry(window,width=30)
        self.tf1.config(font=self.font1)
        self.tf1.place(x=250,y=500)

        self.l2 = Label(window, text='PERSON NAME')
        self.l2.config(font=self.font1)
        self.l2.place(x=50,y=550)

        self.tf2 = Entry(window,width=60)
        self.tf2.config(font=self.font1)
        self.tf2.place(x=250,y=550)

        
        self.btn_snapshot=tkinter.Button(window, text="Register Person", command=self.capturePerson)
        self.btn_snapshot.place(x=50,y=600)
        self.btn_snapshot.config(font=self.font1)
        
        self.btn_train=tkinter.Button(window, text="Train Model", command=self.trainmodel)
        self.btn_train.place(x=250,y=600)
        self.btn_train.config(font=self.font1)

        self.btn_train=tkinter.Button(window, text="Recognize", command=self.recognize)
        self.btn_train.place(x=450,y=600)
        self.btn_train.config(font=self.font1)
        
        
        self.text=Text(window,height=35,width=65)
        scroll=Scrollbar(self.text)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.place(x=1000,y=90)
        self.text.config(font=self.font1)

        self.cascPath = "model/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.recognizer = cv2.face_LBPHFaceRecognizer.create()
        self.student_details = []

        self.delay = 15
        self.update()
        self.window.config(bg='turquoise')
        self.window.mainloop()

    def getImagesAndLabels(self):
        names = []
        ids = []
        faces = []
        imagePaths = [os.path.join("person_details",f) for f in os.listdir("person_details")]
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage,'uint8')
            fname = os.path.basename(imagePath)
            arr = fname.split("_")
            Id = int(arr[1])
            names.append(arr[0])
            faces.append(imageNp)
            ids.append(Id)
        return faces,ids,names

    def getName(self, Id,names,ids):
        name = "Unable to predict name"
        for i in range(len(ids)):
          if ids[i] == Id:
            name = names[i]
            break
        return name    

    def recognize(self):
        faces,ids,names = self.getImagesAndLabels()
        self.recognizer.read("model/Trainner.yml")
        ret, frame = self.vid.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3,5)
        option = 0
        for (x, y, w, h) in faces:
            face_component = gray[y:y+h, x:x+w]
            option = 1
        if option == 1:
            Id, conf = self.recognizer.predict(face_component)
            if(conf < 50):
                name = self.getName(Id,names,ids)
                messagebox.showinfo("Person is video recognized as : "+name, "Person is video recognized as : "+name)
            else:
                messagebox.showinfo("Unable to recognized person in video", "Unable to recognized person in video")
        else:
            messagebox.showinfo("Face not detecting in video", "Face not detecting in video")        
  
    def capturePerson(self):
        global capture_img
        option = 0
        ret, frame = self.vid.get_frame()
        img = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,1.3,5)
        print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img = frame[y:y + h, x:x + w]
            img = cv2.resize(img,(200,200))
            option = 1
        if option == 1:
            self.capture_img = img
            cv2.imshow("Capture Face",img)
            cv2.waitKey(0)
        else:
            messagebox.showinfo("Face or person not detected. Please try again","Face or person not detected. Please try again")

    
             
    def getImagesAndLabels(self):
        names = []
        ids = []
        faces = []
        imagePaths = [os.path.join("person_details",f) for f in os.listdir("person_details")]
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage,'uint8')
            fname = os.path.basename(imagePath)
            arr = fname.split("_")
            Id = int(arr[1])
            print(str(imagePath)+" "+str(Id))
            faces.append(imageNp)
            ids.append(Id)
            names.append(arr[0])
        return faces,ids,names

    def trainmodel(self):
        if self.capture_img is not None:
            pid = self.tf1.get()
            name = self.tf2.get()
            cv2.imwrite("person_details/"+name+"_"+pid+"_0.png",self.capture_img)
            cv2.imwrite("person_details/"+name+"_"+pid+"_1.png",self.capture_img)
            cv2.imwrite("person_details/"+name+"_"+pid+"_2.png",self.capture_img)
            cv2.imwrite("person_details/"+name+"_"+pid+"_3.png",self.capture_img)
            faces,Id,names = self.getImagesAndLabels()
            self.recognizer.train(faces, np.array(Id))
            self.recognizer.save("model/Trainner.yml")
            messagebox.showinfo("Training task completed", "Training task completed")
        else:    
            messagebox.showinfo("Face or person not detected","Face or person not detected")
             
    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.window.after(self.delay, self.update)
            
 
class MyVideoCapture:
    def __init__(self, video_source=0):
        
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.pid = 0
 
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
App(tkinter.Tk(), "Tkinter and OpenCV")
