import shutil
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import os
import pickle
from tracemalloc import is_tracing
import pandas as pd
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn import metrics

root = Tk()

root.title('Major Project')

#width x height
root.geometry("1280x800")

#width,height
root.minsize(1280,720)

root.maxsize(1920,1080)
scrollbar = Scrollbar(root)
scrollbar.pack(side = RIGHT, fill = Y )

#GEU LOGO
photo1 = PhotoImage(file = "/home/sparsh/pyvscode/Major_Project/GenderPredict-main/Resources/download.png")
photo1_label = Label(image=photo1)
photo1_label.pack()

#MAJOR PROJECT TITLE
Heading = Label(text = "Gender Recognition Using Voice",font=('Helvetica', 22, 'bold'))
Heading.pack(pady=45)

global is_train
is_train = 0
#RECORDING VOICE
def record(event):
    import sounddevice as sd
    from scipy.io.wavfile import write
    from scipy.io import wavfile
    import numpy as np
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    #print("Speak Now")
    for widgets in record_frame.winfo_children():
        widgets.destroy()
    Recording = Label(record_frame,text = "Recording...")
    Recording.pack()
    sd.wait()  # Wait until recording is finished
    for widgets in record_frame.winfo_children():
        widgets.destroy()
    Recorded = Label(record_frame,text = "Recording Done!")
    Recorded.pack()
    write('/home/sparsh/pyvscode/Major_Project/GenderPredict-main/output.wav', fs, myrecording)  # Save as WAV file

Record_png = PhotoImage(file = "/home/sparsh/pyvscode/Major_Project/GenderPredict-main/Resources/record.png")
Record_button = Button(root, text = 'Click Me !', image = Record_png,cursor="hand2")
Record_button.pack()
Record_button.bind('<Button-1>',record)

record_frame = Frame(root,borderwidth=6)
record_frame.pack()
record_text = Label(record_frame,text="Record\n   OR")
record_text.pack()

#UPLOAD FILE
def open_file():
    for widgets in loc_frame.winfo_children():
        widgets.destroy()
    global file
    file = askopenfile(filetypes=[('Waveform Audio File', '*wav')])
    if file is not None:
        filepath = os.path.abspath(file.name)
        location = Label(loc_frame,text='File Chosen: '+ str(filepath))
        location.pack()
        src = str(filepath)
        dst = "/home/sparsh/pyvscode/Major_Project/GenderPredict-main/output.wav"
        shutil.copyfile(src,dst)

upload_frame = Frame(root,borderwidth=2)
upload_frame.pack()
upload_button = Button(upload_frame,text = 'Choose File',command=lambda:open_file(),cursor="hand2")
upload_button.pack()
upload_label = Label(upload_frame,text = 'Upload audio in wav format')
upload_label.pack()
loc_frame = Frame(root)
loc_frame.pack()

#PREDICTION
def predict(event):
    for widgets in prediction_frame.winfo_children():
        widgets.destroy()
    model = pickle.load(open('/home/sparsh/pyvscode/Major_Project/GenderPredict-main/finalized_model.sav', 'rb'))
    os.chdir('/home/sparsh/pyvscode/Major_Project/GenderPredict-main/')
    os.system('Rscript extractor_feature.r')
    voice = pd.read_csv('my_voice.csv')
    voice = voice.drop('centroid',axis=1)
    voice = voice.drop('maxdom',axis=1)
    gender = model.predict(voice)
    if(gender[0] == 0):
        prediction = Label(prediction_frame,text="Female",font=('comicsans', 22, 'bold'))
        prediction.pack()
    else:
        prediction = Label(prediction_frame,text = "Male",font=('comicsans', 22, 'bold'))
        prediction.pack()

predict_frame = Frame(root, borderwidth=6)
predict_frame.pack()
predict_button = Button(predict_frame,text = 'Predict Gender')
predict_button.pack()
predict_button.bind('<Button-1>',predict)
prediction_frame = Label(root)
prediction_frame.pack()
#Important Label Options
#text - add the text
#bd - background
#fg - foreground
#padx - x padding
#pady - y padding
#relief - border styling - sunken , raised , groove , ridge

#IF PREDICTION IS CORRECT WE WILL SAVE THE VOICE FEATURES AND SAVE TO DATAFRAME
def save(event):
    for widgets in saved_frame.winfo_children():
        widgets.destroy()
    import csv
    from csv import writer
    model = pickle.load(open('/home/sparsh/pyvscode/Major_Project/GenderPredict-main/finalized_model.sav', 'rb'))
    voice = pd.read_csv('/home/sparsh/pyvscode/Major_Project/GenderPredict-main/my_voice.csv')
    voice = voice.drop('centroid',axis=1)
    voice = voice.drop('maxdom',axis=1)
    gender = model.predict(voice)
    with open('my_voice.csv','r') as read_obj:
        csv_reader = csv.reader(read_obj)
        list_of_csv = list(csv_reader)
    if(gender[0] == 0):
        predicted_gender = "female"
        list_of_csv[1].append("{}".format(predicted_gender))
    else:
        predicted_gender = "male"
        list_of_csv[1].append("{}".format(predicted_gender))
    
    with open ('voice.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(list_of_csv[1])
        f_object.close()
    df=pd.read_csv("/home/sparsh/pyvscode/Major_Project/GenderPredict-main/voice.csv")
    df.drop_duplicates(subset=["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx","label"],keep='last',inplace=True)
    df.to_csv("/home/sparsh/pyvscode/Major_Project/GenderPredict-main/voice.csv",index=False)
    saved = Label(saved_frame,text = "Saved Thank You!",font=('comicsans', 22, 'bold'))
    saved.pack()

correct_frame = Frame(root,borderwidth=2)
correct_frame.pack()
correct_label = Label(correct_frame,text="*Please click the save button if the prediction is correct*")
correct_label.pack()
correct_button = Button(correct_frame,text = "Save")
correct_button.pack()
correct_button.bind('<Button-1>',save)
saved_frame = Frame(root)
saved_frame.pack()

#TRAIN NEW MODEL
def train_model(event):
    for widgets in accurate_frame.winfo_children():
        widgets.destroy()
    df = pd.read_csv('/home/sparsh/pyvscode/Major_Project/GenderPredict-main/voice.csv')
    encoding_columns = [ "label"]
    Encoder = LabelEncoder()
    for column in encoding_columns :
        df[ column ] = Encoder.fit_transform(tuple(df[ column ]))
        df = df.drop('centroid',axis=1)
        df = df.drop('maxdom',axis=1)
        x = df.drop("label",axis=1)
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
        cat = CatBoostClassifier()
        randomforest =RandomForestClassifier()
        xgb = XGBClassifier()
        lgr = LogisticRegression(max_iter=7200)
        estimator = []
        estimator.append(('catboost',cat))
        estimator.append(('randomforest',randomforest))
        estimator.append(('xgb',xgb))
        estimator.append(('lgr',lgr))
        voting = VotingClassifier(estimators= estimator,voting ='soft')
        voting.fit(X_train,y_train)
        y_pred = voting.predict(X_test)
        accu = metrics.accuracy_score(y_test,y_pred)
        accu_label = Label(accurate_frame,text="accuracy = "+str(accu*100))
        accu_label.pack()
        file_name = 'finalized_model.sav'
        pickle.dump(voting, open(file_name,'wb'))
train_frame = Frame(root)
train_frame.pack()
train_label = Label(train_frame,text = "Click the button below to train on the new dataset")
train_label.pack()
train_button = Button(train_frame,text = "Train")
train_button.pack()
train_button.bind('<Button-1>',train_model)
accurate_frame = Frame(root)
accurate_frame.pack()


#BOTTOM
Footer_frame = Frame(root,borderwidth=2)
Footer_frame.pack()
Footer_text = Label(Footer_frame,text="\n             Made By-:\nSparsh Saxena (2013662)\nShivani Deoli ()")
Footer_text.pack()
def callback(url):
    webbrowser.open_new(url)
github_link = Label(root,text="Visit Github Link",cursor="hand2")
github_link.pack()
github_link.bind("<Button-1>",lambda e: callback("https://github.com/sparshsaxenain/GenderPredict"))
#Footer = Label(text = "By-:\nSparsh Saxena (2013662)\nShivani Deoli ()")
#Footer.pack()
quitting = Button(root, text = 'Quit Program', command=root.destroy)
quitting.pack()
root.mainloop()