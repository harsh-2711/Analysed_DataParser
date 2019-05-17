import PyPDF2 
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import docx

from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
from PIL import Image

import os
import sys

import pickle
from argparse import ArgumentParser
from platform import system
from subprocess import Popen
from sys import argv
from sys import stderr

nlp = en_core_web_sm.load()

def analyseDoc(file):

	doc = docx.Document(file)

	fullText = []

	for para in doc.paragraphs:
	    fullText.append(para.text)

	s = "".join(fullText)

	doc = nlp(s)
	print([(X.text, X.label_) for X in doc.ents])

def analysePDF(file):

	pdfFileObj = open(file, 'rb') 
	pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
	sentences = []

	for i in range(pdfReader.numPages):
		pageObj = pdfReader.getPage(i) 
		sentences.append(pageObj.extractText())

	s = "".join(l)
	s = s.split("\n")
	s = "".join(s)

	pdfFileObj.close() 

	doc = nlp(s)
	print([(X.text, X.label_) for X in doc.ents])

def detectText1(image):

	img = cv2.imread(image)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((1, 1), np.uint8)
	img = cv2.dilate(img, kernel, iterations=1)
	img = cv2.erode(img, kernel, iterations=1)

	cv2.imwrite("thres.png", img)
	result = pytesseract.image_to_string(Image.open("thres.png"))
	os.remove("thres.png")

	doc = nlp(result)
	print([(X.text, X.label_) for X in doc.ents])


def decode_predictions(scores, geometry, min_confidence):

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):

			if scoresData[x] < min_confidence:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)

def detectText2(image_path, model, min_confidence, width, height, padding):

	image = cv2.imread(image_path)
	orig = image.copy()
	(origH, origW) = image.shape[:2]

	(newW, newH) = (width, height)
	rW = origW / float(newW)
	rH = origH / float(newH)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet(model)

	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(rects, confidences) = decode_predictions(scores, geometry, min_confidence)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	results = []

	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		dX = int((endX - startX) * padding)
		dY = int((endY - startY) * padding)

		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		roi = orig[startY:endY, startX:endX]

		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)

		results.append(((startX, startY, endX, endY), text))

	results = sorted(results, key=lambda r:r[0][1])

	sentences = []

	for ((startX, startY, endX, endY), text) in results:
		sentences.append(text)

	s = "".join(sentences)
	s = s.split("\n")
	s = "".join(s)

	doc = nlp(s)
	print([(X.text, X.label_) for X in doc.ents])

def arg_parse():
	arg_p = ArgumentParser('NER Python Wrapper')
	arg_p.add_argument('-f', '--filename', type=str, default=None,
		help="path to doc file")
	arg_p.add_argument('-d', '--pdf', type=str, default=None,
		help="path to pdf file")
	arg_p.add_argument("-i", "--image", type=str, 
		help="path to input image")
	arg_p.add_argument("-east", "--east", type=str, 
		help="path to input EAST text detector")
	arg_p.add_argument("-c", "--min-confidence", type=float, default=0.5,
		help="minimum probability required to inspect a region")
	arg_p.add_argument("-w", "--width", type=int, default=320,
		help="nearest multiple of 32 for resized width")
	arg_p.add_argument("-e", "--height", type=int, default=320,
		help="nearest multiple of 32 for resized height")
	arg_p.add_argument("-p", "--padding", type=float, default=0.0,
		help="amount of padding to add to each border of ROI")
	return arg_p


def main(args):
    arg_p = vars(arg_parse().parse_args())

    if arg_p['filename'] == None and arg_p['pdf'] == None and arg_p['image'] == None:
    	print("Please provide path to a file or to an image")
    	exit(1)

    elif arg_p['pdf'] == None and arg_p['image'] == None:
    	analyseDoc(arg_p['filename'])

    elif arg_p['filename'] == None and arg_p['image'] == None:
    	analysePDF(arg_p['pdf'])

    elif arg_p['pdf'] == None and arg_p['filename'] == None:
    	inp = input("Which method do you want to use - 1 or 2? ")
    	if inp == '1':
    		detectText1(arg_p['image'])

    	elif inp == '2':
    		detectText2(arg_p['image'], arg_p['east'], arg_p['min-confidence'], 
    			arg_p['width'], arg_p['height'], arg_p['padding'])
    	else:
    		print("Please provide proper choice")
    		exit(1)

    else:
    	print("Please provide path to any one file")
    	exit(1)

if __name__ == '__main__':
    exit(main(argv))