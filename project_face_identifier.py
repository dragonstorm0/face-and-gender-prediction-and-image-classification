#!/usr/bin/env python
# coding: utf-8

# In[1]:


#along with the code in both the ipynd and .py format i am also submitting the two images that i tested this code on.
import _tkinter
import tkinter as tk
from tkinter import messgaebox
from tkinter import ttk
import pandas as pd
from PIL import Image,ImageTk
import cv2
from tkinter import filedialog


face_clf=cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
my_label=None
def browse_image():
    global my_label
    if(my_label!=None):
        #to remove prior used image if any.
        my_label.destroy()
    path_image=filedialog.askopenfilename(initialdir="/",title="Open File",filetypes=(("Image", "*.jpg"),("Image","*.png"), ("All Files", "*.*")))
    img=cv2.imread(path_image,cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clf.detectMultiScale(gray,minSize=(100,100))
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img1=Image.fromarray(img).resize((500,500))
#to save the image file
#    img1.save("new_detected.jpg")
    my_img=ImageTk.PhotoImage(img1)
#face detection is working perfectly.
#     cv2.imshow('image',img1)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    my_label=Label(image=my_img)
    my_label.image=my_img
    my_label.configure(width=500,height=500)
    my_label.place(relx=.21,rely=.3)
def close():
    window.destroy()

window=Tk()
window.geometry("1000x800")
window.configure(bg='powder blue')
window.title('face recognition system')
window.resizable(height=False,width=False)

label_main=Label(window,text='Face Recognition System',font=('serif',40,'bold italic'),fg='black',bg='powder blue')
label_main.pack()
btn_browse=Button(window,command=lambda:browse_image(),text='Browse Image',font=('',15,'bold'),bd=5)
btn_browse.place(relx=.4,rely=.2)
btn_quit=Button(window,command=lambda:close(),text='Quit',font=('quit',15,'bold'),bd=5)
btn_quit.place(relx=.1,rely=.85)

window.mainloop()


# In[ ]:




