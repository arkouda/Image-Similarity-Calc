# root.configure(background='light green') 
from multiprocessing import Process, Queue, Pipe
from tkinter import filedialog
from tkinter import Tk, StringVar, IntVar, Label, Entry, Button, Spinbox, Radiobutton, mainloop

def fun():
    print(approachVar.get())
    print("check")

def checker():
    lbl_error.config(text="No Action")

def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(delete_dup)

root = Tk()
root.title("Image Similarity Calculator")
root.attributes('-notify', True)
root.geometry("700x300+30+30")

folder_path = StringVar()
delete_dup = IntVar()
approachVar = IntVar()

lbl_dir = Label(root, text='Directory')
lbl_dir.place(x = 20, y = 30 , width=120, height=25)
lbl_path = Entry(root, textvariable=folder_path)
lbl_path.place(x = 150, y = 30 , width=120, height=25)
button_browse = Button(text="Browse", command=browse_button)
button_browse.place(x = 280, y = 30 , width=120, height=25)

lbl_thresh = Label(root, text='Threshold')
lbl_thresh.place(x = 20, y = 60 , width=120, height=25)
Spinbox_thresh = Spinbox(root, from_=0, to=100)
Spinbox_thresh.place(x = 150, y = 60 , width=120, height=25)

lbl_approach = Label(root, text='Calc Approach')
lbl_approach.place(x = 20, y = 90, width=120, height=25)
R1 = Radiobutton(root, text="Normal", variable=approachVar, value=1)
R1.place(x=150, y=90, width=75, height=25)
R2 = Radiobutton(root, text="Exponential", variable=approachVar, value=2)
R2.place(x=235, y=90, width=120, height=25)
R3 = Radiobutton(root, text="Low", variable=approachVar, value=3,command=fun)
R3.place(x=350, y=90, width=75, height=25)


lbl_deldup = Label(root, text='Delete Duplicate')
lbl_deldup.place(x=20, y=120, width=120, height=25)
Rb1 = Radiobutton(root, text="Yes", variable=delete_dup, value=1)
Rb1.place(x=150, y=120, width=50, height=25)
Rb2 = Radiobutton(root, text="No", variable=delete_dup, value=2)
Rb2.place(x=210, y=120, width=50, height=25)

button_browse = Button(text="Submit", command=checker)
button_browse.place(x = 150, y = 160 , width=100, height=25)
lbl_error = Label(root)
lbl_error.place(x = 100, y=200, width = 200, height=25)


approachVar.set(1)
delete_dup.set(1)
mainloop()