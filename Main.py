#import everything
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageEnhance, ImageGrab
import PIL.ImageOps
import random
import time
import math
import heapq
import cv2
import PIL
import sys
import os
from time import sleep
from imutils import contours
import imutils
from tkinter import *



# setting up tf ImageDataGenerator parameters
batch_size = 10
num_classes = 13
epochs = 20
class_names = ['0','1','2','3','4','5','6','7','8','9', 'add', 'sub', 'mult']

datasetpath = 'Dataset/_Final/'

ds_train      = image_dataset_from_directory(datasetpath, validation_split=0.3, image_size=(28, 28), label_mode='categorical', class_names=class_names, color_mode='grayscale', batch_size=batch_size, seed=134, subset='training')
ds_validation = image_dataset_from_directory(datasetpath, validation_split=0.3, image_size=(28, 28), label_mode='categorical', class_names=class_names, color_mode='grayscale', batch_size=batch_size, seed=134, subset='validation')
ds_test       = image_dataset_from_directory(datasetpath, validation_split=0.99, image_size=(28, 28), label_mode='categorical', class_names=class_names, color_mode='grayscale', batch_size=batch_size, shuffle=False, subset='validation')

def base_cnn():
    model=Sequential()
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())    

    model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
    
    model.add(Dense(13,activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #print(model.summary())
    return model


pathtomodel = 'best_model'
model = base_cnn()
model.load_weights(pathtomodel)





# Main GUI Application
from keras.preprocessing.image import img_to_array
image1 = PIL.Image.new("RGB", (1000, 300), 'black')
draw = ImageDraw.Draw(image1)
app = Tk()
app.geometry("1050x300")
app.resizable(0, 0)
app.configure(background='grey')
app.title("Multi-digit MNIST - Please draw the equation here")
lasx, lasy = 0, 0
def get_x_and_y(event):
	global lasx, lasy
	lasx, lasy = event.x, event.y
def draw_smth(event):
	global lasx, lasy, draw
	canvas.create_line((lasx, lasy, event.x, event.y), width=7, fill='white', capstyle=ROUND)
	draw.line((lasx, lasy, event.x, event.y), width=7, fill='white')
	lasx, lasy = event.x, event.y
def clear_canv():
	global canvas
	app.destroy()
	canvas.delete('all')
	canvas.create_line((700, 85,  800, 85),  width=7, fill='white', capstyle=ROUND)
	canvas.create_line((700, 120, 800, 120), width=7, fill='white', capstyle=ROUND)

def squareifyImage(roi):
	old_height, old_width, channels = roi.shape
	if(old_height > old_width): 
		new_height, new_width = old_height, old_height
	else: 
		new_height, new_width = old_width, old_width
	result = np.full((new_height,new_width, channels), (0,0,0), dtype=np.uint8)
	x_center = (new_width - old_width) // 2
	y_center = (new_height - old_height) // 2
	result[y_center:y_center+old_height, x_center:x_center+old_width] = roi
	return result

def save_image():
	image1.save(f"log.png")
	im = cv2.imread(f"log.png")
	contours, hierarchy = cv2.findContours(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
	sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
	list_of_images = []
	check_images = []
	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)
		# squareify image and resize to 28x28
		result = squareifyImage(im[y:y+h, x:x+w])
		img_pad = cv2.copyMakeBorder(cv2.resize(result, (20, 20)), 4, 4, 4, 4, cv2.BORDER_CONSTANT, (0,0,0))
		img_pad = cv2.blur(img_pad, (2,2)) * 1.1
		# if image is on the left side, add to the left side of the list
		# if image is on the right side, add to the right side of the list
		if(x < 750):
			list_of_images.append(img_to_array(img_pad[:,:,0]).reshape((28, 28, 1)))
		else:
			check_images.append(img_to_array(img_pad[:,:,0]).reshape((28, 28, 1)))
	
	resultstr = ""
	checkstr = ""
	preds2 = []
	preds = model.predict(np.array(list_of_images))
	if(len(check_images) > 0):
		preds2 = model.predict(np.array(check_images))

	for pred in preds:
		#print(pred)
		if(class_names[np.argmax(pred)] == 4 and (pred[4] != 1.0)):
			resultstr += str(heapq.nlargest(2, xrange(len(pred)), key=pred.__getitem__)[1])

		if(np.argmax(pred) == 10):
			resultstr += "+"
		elif (np.argmax(pred) == 11):
			resultstr += "-"
		elif (np.argmax(pred) == 12):
			resultstr += "*"
		else:
			resultstr += str(class_names[np.argmax(pred)])

	for pred in preds2:
		#print(pred)
		if(class_names[np.argmax(pred)] == 4 and (pred[4] != 1.0)):
			# if 4 is the largest class, and 4 is not the perfect guess, then the second largest class is the answer
			# sort the classes in descending order and get the index of the 2nd largest class
			checkstr += str(heapq.nlargest(2, xrange(len(pred)), key=pred.__getitem__)[1])

		if(np.argmax(pred) == 10):
			checkstr += "+"
		elif (np.argmax(pred) == 11):
			checkstr += "-"
		elif (np.argmax(pred) == 12):
			checkstr += "*"
		else:
			checkstr += str(class_names[np.argmax(pred)])


	for i in range(11):
		try: os.remove(f'{i}.png')
		except: pass
	
	try:
		if(len(check_images) > 0):
			label.config(text=f"{str(resultstr)+ '=' + str(checkstr)} => {eval(str(eval(resultstr))+ '==' + str(eval(checkstr)))}")
		else:
			label.config(text=f"{resultstr}={eval(resultstr)}")
	except:
		label.config(text=f"Invalid Expression ({str(resultstr)+ '==' + str(checkstr)})")

canvas = Canvas(app, bg='black', width=1050, height=200)
canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)
canvas.grid(row=0, column=0, pady=2, sticky=NSEW, columnspan=2)
canvas.create_line((700, 85, 800, 85), width=7, fill='white', capstyle=ROUND)
canvas.create_line((700, 120, 800, 120), width=7, fill='white', capstyle=ROUND)
recognize = Button(master=app, text='Solve',width=15, height=2, command=save_image).grid(row=2, column=0, sticky=NSEW, pady=1, padx=1)
clear_but = Button(master=app, text='Clear (not working yet)',width=15, height=2, command=clear_canv).grid(row=2, column=1, sticky=NSEW, pady=1, padx=1)
label=Label(app, width=10, height=1, font=("Helvetica", 30))
label.grid(row=3, sticky=NSEW, pady=1, padx=1, columnspan=2)
app.mainloop()





