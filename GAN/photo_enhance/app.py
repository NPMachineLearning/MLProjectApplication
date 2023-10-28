import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import tensorflow as tf


class App(tk.Tk):
    def __init__(self, title="window", window_size=(200, 200), icon=None):
        super().__init__()
        
        self.model = tf.keras.models.load_model("esrgan-tf2")
        
        self.image_path = None
        self.input_image = None
        self.output_image = None
        self.canvas_size = (512,512)
        
        self.setup_window(title, window_size, icon=icon)
        self.layout_ui()
        
    def setup_window(self, title, size=None, icon=None):
        self.title(title)
        
        if icon is not None:
            self.iconbitmap(icon)
        
        if size is not None:
            screen_h = self.winfo_screenheight()
            screen_w = self.winfo_screenwidth()
            c_y = int(screen_h/2 - size[1]/2)
            c_x = int(screen_w/2 - size[0]/2)
            self.geometry(f"{size[0]}x{size[1]}+{c_x}+{c_y}")
            
    def layout_ui(self):
        self.s = ttk.Style()
        self.s.configure('btn.TButton', font=('Arial', 12))
        
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=20)
        self.columnconfigure(0, weight=1)
        
        self.top_frame = ttk.Frame(self)
        self.top_frame.grid(row=0, sticky=tk.N)
        self.top_frame.rowconfigure(0, weight=1)
        
        self.bottom_frame = ttk.Frame(self)
        self.bottom_frame.grid(row=1, sticky=tk.N)
        self.bottom_frame.columnconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(1, weight=1)
        
        
        self.open_image_btn = ttk.Button(self.top_frame, text="選擇圖片",
                                         style="btn.TButton",
                                         command=self.on_open_image)
        self.open_image_btn.grid(pady=(10, 10))
        
        self.convert_btn = ttk.Button(self.top_frame, text="強化",
                                      style="btn.TButton",
                                      command=self.enhance_image)
        self.convert_btn.grid(pady=(0, 10))
        
        self.convert_btn = ttk.Button(self.top_frame, text="儲存",
                                      style="btn.TButton",
                                      command=self.save_output)
        self.convert_btn.grid(pady=(0, 10))
        
        self.msg_label = ttk.Label(self.top_frame, text="")
        self.msg_label.grid(pady=(0, 10))
        
        self.input_image_canvas = tk.Canvas(self.bottom_frame, 
                                            width=self.canvas_size[0], 
                                            height=self.canvas_size[1])
        self.input_image_canvas.grid(row=0, column=0)
        
        
        
        self.output_image_canvas = tk.Canvas(self.bottom_frame, 
                                            width=self.canvas_size[0], 
                                            height=self.canvas_size[1])
        self.output_image_canvas.grid(row=0, column=1)
        
        
    
    def on_open_image(self):
        self.msg_label["text"] = ""
        
        f_types = [("image jpg", ".jpg .jpeg .png")]
        self.image_path = filedialog.askopenfilename(initialdir=".", 
                                                     filetypes=f_types)
        
        if self.image_path:
            with Image.open(self.image_path) as img:
                img = img.resize(self.canvas_size)
                self.input_image = ImageTk.PhotoImage(img)
                
            self.show_input_image()
            
        
    def show_input_image(self):
        if self.input_image is not None:
            self.input_image_canvas.create_image((0,0),
                                                 anchor=tk.NW,
                                                 image=self.input_image)
    
    def show_output_image(self):
        if self.input_image is not None:
            self.output_image_canvas.create_image((0,0), 
                                                  anchor=tk.NW,
                                                  image=self.output_image)
    def save_output(self):
        if self.output_image:
            output_img = ImageTk.getimage(self.output_image)
            output_img = output_img.convert("RGB")
            output_img.save("output_enhanced.jpg")
            self.msg_label["text"] = "儲存成功!"
            
    def enhance_image(self):
        if self.image_path is not None:
            output_image = self.tf_enhance_image(self.image_path)
            output_image = tf.clip_by_value(output_image, 0, 255)
            output_image = tf.cast(output_image, tf.uint8)
            
            image = Image.fromarray(output_image.numpy())
            image = image.resize((self.canvas_size))

            self.output_image = ImageTk.PhotoImage(image)
            
            self.show_output_image()
            
    def tf_enhance_image(self, img_path):
        input_data = self.get_input_data(img_path)
        output_data = self.model(input_data)
        output = tf.squeeze(output_data, axis=0)
        
        return output
    
    def get_input_data(self, img_path):
        input_img = tf.image.decode_image(tf.io.read_file(img_path))
        if input_img.shape[-1] == 4:
            input_img = input_img[...,:-1]
        size = (tf.convert_to_tensor(input_img.shape[:-1]))
        input_img = tf.image.crop_to_bounding_box(input_img, 0, 0, size[0], size[1])
        input_img = tf.cast(input_img, tf.float32)
        input_img = tf.expand_dims(input_img, axis=0)
        
        return input_img
 
if __name__ == "__main__":              
    app = App(title="影像強化", 
              window_size=(1024, 768),
              icon="icon.ico")
    app.mainloop()            