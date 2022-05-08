import shutil
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import os
import pickle
import pandas as pd
import webbrowser

root = Tk()

root.title('Major Project')

#width x height
root.geometry("1280x720")

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

global predicted_gender
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
    write('/home/sparsh/pyvscode/Major_Project/GenderPredict-main/output1.wav', fs, myrecording)  # Save as WAV file

Record_png = PhotoImage(file = "/home/sparsh/pyvscode/Major_Project/GenderPredict-main/Resources/record.png")
Record_button = Button(root, text = 'Click Me !', image = Record_png)
Record_button.pack()
Record_button.bind('<Button-1>',record)

record_frame = Frame(root,borderwidth=6)
record_frame.pack()
record_text = Label(record_frame,text="Record\n   OR")
record_text.pack()

#UPLOAD FILE
def open_file():
    global file
    file = askopenfile(filetypes=[('Waveform Audio File', '*wav')])
    if file is not None:
        filepath = os.path.abspath(file.name)
        location = Label(upload_frame,text='File Chosen: '+ str(filepath))
        location.pack()
        src = str(filepath)
        dst = "/home/sparsh/pyvscode/Major_Project/GenderPredict-main/output.wav"
        shutil.copyfile(src,dst)

upload_frame = Frame(root,borderwidth=2)
upload_frame.pack()
upload_button = Button(upload_frame,text = 'Choose File',command=lambda:open_file())
upload_button.pack()
upload_label = Label(upload_frame,text = 'Upload audio in wav format')
upload_label.pack()

#PREDICTION
def predict(event):
    for widgets in predict_frame.winfo_children():
        widgets.destroy()
    model = pickle.load(open('/home/sparsh/pyvscode/Major_Project/GenderPredict-main/finalized_model.sav', 'rb'))
    os.chdir('/home/sparsh/pyvscode/Major_Project/GenderPredict-main/')
    os.system('Rscript extractor_feature.r')
    voice = pd.read_csv('my_voice.csv')
    voice = voice.drop('centroid',axis=1)
    voice = voice.drop('maxdom',axis=1)
    gender = model.predict(voice)
    if(gender[0] == 0):
        prediction = Label(predict_frame,text="Female",font=('comicsans', 22, 'bold'))
        prediction.pack()
    else:
        prediction = Label(predict_frame,text = "Male",font=('comicsans', 22, 'bold'))
        prediction.pack()

predict_frame = Frame(root, borderwidth=6)
predict_frame.pack()
predict_button = Button(predict_frame,text = 'Predict Gender')
predict_button.pack()
predict_button.bind('<Button-1>',predict)
#Important Label Options
#text - add the text
#bd - background
#fg - foreground
#padx - x padding
#pady - y padding
#relief - border styling - sunken , raised , groove , ridge

#IF PREDICTION IS CORRECT WE WILL SAVE THE VOICE FEATURES AND SAVE TO DATAFRAME
def save_and_train(event):
    for widgets in correct_frame.winfo_children():
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
    saved = Label(correct_frame,text = "Saved Thank You!",font=('comicsans', 22, 'bold'))
    saved.pack()
correct_frame = Frame(root,borderwidth=2)
correct_frame.pack()
correct_label = Label(correct_frame,text="*Please click the save button if the prediction is correct*")
correct_label.pack()
correct_button = Button(correct_frame,text = "Save")
correct_button.pack()
correct_button.bind('<Button-1>',save_and_train)


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