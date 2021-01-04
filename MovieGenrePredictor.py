from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QTextEdit
import sys

#training vecrorizer
vectorizer = TfidfVectorizer("english",smooth_idf=True,use_idf=True)
features = vectorizer.fit_transform(pd.read_csv("vectorizer_input.csv")["Description"])

#loading model
svc = pickle.load(open('finalized_model.sav', 'rb'))

# supported genres as per the notebook
genre={	   'Drama': 	1,
	 	   'Action': 	2,
	 	   'Crime': 	3,
	 	   'Adventure': 4,
	 	   'Biography': 5,
	 	   'Animation': 6,
	 	   'Comedy': 	7,
	 	   'Horror': 	8	}


def pre_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

# A method that makes use of svc model and vectorizer to predict genre
# from the given text(msg).
def predict_genre(msg):
    if(len(msg)<100):
        return(" Kindly insert more than 100 characters ")
    msg=pre_process(msg)
    #using same vectorizer object to generate same number of features from given input
    ft=vectorizer.transform([msg])
    n=svc.predict(ft)[0]
    index=list(genre.values()).index(n)
    return list(genre.keys())[index]



# A simple app for Genre Prediction   

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle('Genre Predictor')
window.setGeometry(100, 100, 300, 300)
window.move(600, 15)

Msg = QLabel('<h2>Enter the movie description</h2>', parent=window)
Msg.move(10, 10)

txt=QTextEdit(parent=window)
txt.setGeometry(10,40,280,100)

result=QPushButton("",parent=window)
result.setGeometry(10,200,280,50)

button=QPushButton("GO",parent=window)
button.setGeometry(125,150,100,30)

def update():
    result.setText(predict_genre(txt.toPlainText()))
                                 
                                 
button.clicked.connect(update)
window.show()
sys.exit(app.exec_())
