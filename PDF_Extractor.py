import PyPDF2 
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

pdfFileObj = open('2.pdf', 'rb') 
pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 

l = []

for i in range(pdfReader.numPages):

	pageObj = pdfReader.getPage(i) 
	l.append(pageObj.extractText())

s = "".join(l)
s = s.split("\n")
s = "".join(s)
#print(s)
pdfFileObj.close() 

doc = nlp(s)
print([(X.text, X.label_) for X in doc.ents])

#print([X.text for X in doc.ents])

#print([(X, X.ent_iob_, X.ent_type_) for X in doc])
