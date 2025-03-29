import tkinter as tk

root = tk.Tk()
root.title("My App")

label = tk.Label(root, text="Hello World!")
label.pack()

button = tk.Button(root, text="Click Me")
button.pack()

root.mainloop()
