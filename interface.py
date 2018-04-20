import codecs
from gensim import corpora, matutils
from pyvi.pyvi import ViTokenizer

import tkinter as tk
#from tkinter import ttk

from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Frame, Button, Label, Style
from tkinter import messagebox

import re
#import pickle as cPickle
import numpy as np

LARGE_FONT = ("Verdana", 12)

class FileReader(object):
	def __init__(self, filePath, encoder = None):
		self.filePath = filePath
		self.encoder = encoder if encoder != None else 'utf-16le'

	def read(self):
		try:
			with codecs.open(self.filePath, "r", "utf-8") as f:
				s = f.read()
		except Exception:
			with codecs.open(self.filePath, "r", self.encoder) as f:
				s = f.read()
		return s
	def read_stopwords(self):
		with open(self.filePath, 'r', encoding="utf-8") as f:
			stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
		return stopwords

	def load_dictionary(self):
		return corpora.Dictionary.load_from_text(self.filePath)

class NLP(object):
	def __init__(self, text = None):
		self.text = text
		self.__set_stopwords()

	def __set_stopwords(self):
		self.stopwords = FileReader('stopwords-nlp-vi.txt').read_stopwords()

	def segmentation(self):
		return ViTokenizer.tokenize(self.text)

	def split_words(self):
		text = self.segmentation()
		try:
			return [x.strip('01234”56789%@$.,“?=+-!;/()–*’…"&^:#|\n\t\'').lower() for x in text.split()]
		except TypeError:
			return []

	def get_words_feature(self):
		split_words = self.split_words()
		return [word for word in split_words if word.encode('utf-8') not in self.stopwords]

class FeatureExtraction(object):
	def __load_dictionary(self):
		self.dictionary = FileReader('dictionary/dictionary.txt').load_dictionary()

	def get_dense(self, text):
		self.__load_dictionary()
		words = NLP(text).get_words_feature()
		vec = self.dictionary.doc2bow(words)
		dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
		return dense

class Window(Frame):
	def __init__(self, parent = None):
		Frame.__init__(self, parent)
		self.parent = parent
		self.path_file = None
		self.text_predict = None
		self.init_window()
		#with open('trained_model/my_dumped_classifier.pkl', 'rb') as fid:
		#	model = cPickle.load(fid)
		#self.model = model
		self.W_best = np.load('weight/weight.npy')
		self.label = ['Văn hóa', 'Thế giới', 'Khoa học', 'Sức khỏe', 'Chính trị xã hội',
		'Vi tính', 'Kinh doanh', 'Thể thao', 'Pháp luật', 'Đời sống']

	def init_window(self):
		self.parent.title("Classfication")

		self.style = Style()
		self.style.theme_use("clam")

		self.pack(fill = BOTH, expand = 1)

		quitButton = Button(self, text = 'Quit', command = self.close_window)
		#quitButton.config(font = BUTTON_FONT)
		quitButton.place(x = 0, y = 0)

		labelBrowse = Label(self, text = 'Chọn file .txt dự đoán: ', font = LARGE_FONT)
		labelBrowse.place(x = 50, y = 50)

		browseButton = Button(self, text = 'Browse file', command = self.browse_file)
		browseButton.place(x = 240, y = 45)

		self.text = Label(self, text = self.path_file, font = LARGE_FONT)
		self.text.place(x = 340, y = 50)

		predictButton = Button(self, text = 'Predict', command = self.predict)
		predictButton.place(x = 240, y = 80)

		self.textBox = Text(self, height = 24, width = 94, font = LARGE_FONT)
		scroll = Scrollbar(self)
		#scroll.pack(side = RIGHT, fill = Y)
		#self.textBox.pack(side = LEFT, fill = Y)
		scroll.config(command = self.textBox.yview)
		#self.textBox.config(yscrollcommand = scroll.set)
		#test = "affffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		#self.textBox.insert(END, test)
		self.textBox.place(x = 80, y = 120)
		self.textBox.config(state = DISABLED)

		self.predictLabel = Label(self, text = self.text_predict, font = LARGE_FONT)
		self.predictLabel.place(x = 340, y = 87)

	def close_window(self):
		click = messagebox.askquestion("Close Window", "Are You Sure?", icon = 'warning')
		if click == 'yes':
			exit()

	def browse_file(self):
		filename = filedialog.askopenfilename()
		self.path_file = filename
		self.text.config(text = filename)

		isTxt = re.findall(r"\.\w{1,}", filename)
		if len(isTxt) == 0:
			messagebox.showinfo(title = "Error", message = "Lỗi định dạng file")
			self.textBox.config(state = NORMAL)
			self.textBox.delete('1.0', END)
			self.textBox.config(state = DISABLED)
			self.content = "";
		else:
			if isTxt[0] == ".txt":
				txt = FileReader(filename).read()
				self.textBox.config(state = NORMAL)
				self.textBox.delete('1.0', END)
				self.textBox.insert(END, txt)
				self.textBox.config(state = DISABLED)
				self.content = txt;
				#messagebox.showinfo(title = "Infomation", message = txt)
			else:
				messagebox.showinfo(title = "Error", message = "Lỗi định dạng file")
				self.textBox.config(state = NORMAL)
				self.textBox.delete('1.0', END)
				self.textBox.config(state = DISABLED)
				self.content = "";
		#messagebox.showinfo(title = 'Infomation', message = filename)

	def predict(self):
		if self.content != "":
			self.dense = FeatureExtraction().get_dense(self.content)
			#vector = np.reshape(self.dense, (-1, 1)).T
			#label_predict = self.model.predict(vector)
			#self.predictLabel.config(text = label_predict[0])
			x_predict = np.asarray(self.dense)
			x_predict = np.hstack([x_predict, 1])
			y_predict = x_predict.dot(self.W_best)
			index_predict = np.argmax(y_predict)
			self.predictLabel.config(text = self.label[index_predict])
		else:
			self.predictLabel.config(text = "")
			messagebox.showinfo(title = "Error", message = "Vui lòng chọn file")

def disable_event():
	pass

root = Tk()
root.geometry("1100x600+130+50")
root.protocol("WM_DELETE_WINDOW", disable_event)
root.resizable(0, 0)
app = Window(root)
root.mainloop()