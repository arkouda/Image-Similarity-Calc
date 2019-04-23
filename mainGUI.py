from tkinter import filedialog
from tkinter import Tk, StringVar, IntVar, Label, Entry, Button, Spinbox, Radiobutton, mainloop
import sys
import datetime
import matplotlib
matplotlib.use('TkAgg')
import compute

def fun():
    print(approachVar.get())
    print("check")

def checker():

    imFilePATH = folder_path.get()

    if approachVar.get() == 1:
        approach = 'N' 
    else:
        approach = 'E'

    if multiprocessingVar.get() == 1:
        multiprocessingFlag = 'ON'
    else: 
        multiprocessingFlag = 'OFF'

    threshold = int(Spinbox_thresh.get())

    if delete_dup.get() == 1:
        fileDeleteFlag = 'YES'
    else:
        fileDeleteFlag = 'NO'

    print("Params: ", (imFilePATH, approach, threshold, multiprocessingFlag, fileDeleteFlag))
    startTime = datetime.datetime.now()
    print('START TIME: ' , startTime)
    Finaljson = compute.driverFunction(imFilePATH, approach, threshold, multiprocessingFlag, fileDeleteFlag)
    print(Finaljson)
    print('END TIME: ' , datetime.datetime.now())
    print('DURATION TIME: ' , datetime.datetime.now() - startTime)
    

    lbl_error.config(text=Finaljson)

def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)

root = Tk()
root.title("Image Similarity Calculator")
root.attributes('-alpha', True)
root.geometry("700x300+30+30")

folder_path = StringVar()
delete_dup = IntVar()
approachVar = IntVar()
multiprocessingVar = IntVar()
thresholdVar = IntVar()

lbl_dir = Label(root, text='Directory')
lbl_dir.place(x = 20, y = 30 , width=120, height=25)
lbl_path = Entry(root, textvariable=folder_path)
lbl_path.place(x = 150, y = 30 , width=120, height=25)
button_browse = Button(text="Browse", command=browse_button)
button_browse.place(x = 280, y = 30 , width=120, height=25)

lbl_thresh = Label(root, text='Threshold')
lbl_thresh.place(x = 20, y = 60 , width=120, height=25)
Spinbox_thresh = Spinbox(root, from_=0, to=100 )
Spinbox_thresh.place(x = 150, y = 60 , width=120, height=25)

lbl_approach = Label(root, text='Calc Approach')
lbl_approach.place(x = 20, y = 90, width=120, height=25)
R1 = Radiobutton(root, text="Hamming", variable=approachVar, value=1)
R1.place(x=150, y=90, width=90, height=25)
R2 = Radiobutton(root, text="SSIM Index", variable=approachVar, value=2)
R2.place(x=260, y=90, width=110, height=25)

lbl_deldup = Label(root, text='Delete Duplicate')
lbl_deldup.place(x=20, y=120, width=120, height=25)
Rb1 = Radiobutton(root, text="Yes", variable=delete_dup, value=1)
Rb1.place(x=150, y=120, width=50, height=25)
Rb2 = Radiobutton(root, text="No", variable=delete_dup, value=2)
Rb2.place(x=210, y=120, width=50, height=25)

lbl_multiprocessing = Label(root, text='Multiprocessing')
lbl_multiprocessing.place(x=20, y=150, width=120, height=25)
Rb1 = Radiobutton(root, text="ON", variable=multiprocessingVar, value=1)
Rb1.place(x=150, y=150, width=50, height=25)
Rb2 = Radiobutton(root, text="OFF", variable=multiprocessingVar, value=2)
Rb2.place(x=210, y=150, width=50, height=25)

button_browse = Button(text="Submit", command=checker)
button_browse.place(x = 150, y = 180 , width=100, height=25)
lbl_error = Label(root)
lbl_error.place(x = 100, y=210, width = 200, height=25)

approachVar.set(1)
delete_dup.set(2)
multiprocessingVar.set(1)
mainloop()