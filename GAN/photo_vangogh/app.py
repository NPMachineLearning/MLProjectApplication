import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import tensorflow as tf
from instance_norm import InstanceNormalization


class App(tk.Tk):
    def __init__(self, title="window", window_size=(200, 200), icon=None):
        super().__init__()
        
        custom_objects={"CycleGAN>InstanceNormalization":InstanceNormalization}
        
        self.model = tf.keras.models.load_model("gen_f.h5",
                                                custom_objects=custom_objects,
                                                compile=False)
        
        self.image_path = None
        self.input_image = None
        self.output_image = None
        self.image_size = (256, 256)
        self.canvas_size = (512, 512)
        
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
        self.open_image_btn.grid(pady=10)
        
        
        self.convert_btn = ttk.Button(self.top_frame, text="轉換成梵谷畫",
                                      style="btn.TButton",
                                      command=self.on_convert_image)
        self.convert_btn.grid(pady=10)
        
        self.input_image_canvas = tk.Canvas(self.bottom_frame, 
                                            width=self.canvas_size[0], 
                                            height=self.canvas_size[1])
        self.input_image_canvas.grid(row=0, column=0)
        
        self.output_image_canvas = tk.Canvas(self.bottom_frame, 
                                             width=self.canvas_size[0], 
                                             height=self.canvas_size[1])
        self.output_image_canvas.grid(row=0, column=1)
    
    def on_open_image(self):
        f_types = [("image jpg", ".jpg .jpeg")]
        self.image_path = filedialog.askopenfilename(initialdir=".", filetypes=f_types)
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
        if self.output_image is not None:
            self.output_image_canvas.create_image((0,0),
                                                  anchor=tk.NW,
                                                  image=self.output_image)
            
    def on_convert_image(self):
        if self.image_path is not None:
            output_img_arr = self.tf_convert_image(self.image_path)
            output_image_pil = Image.fromarray(output_img_arr)
            output_image_pil = output_image_pil.resize(self.canvas_size)
            self.output_image = ImageTk.PhotoImage(output_image_pil)
            self.show_output_image()
            
    def tf_convert_image(self, img_path):
        input_data = self.get_input_data(img_path)
        output = self.model.predict(input_data, verbose=0)
        output_img = tf.squeeze(output, axis=0)
        output_img = output_img * 127.5 + 127.5
        output_img = tf.cast(output_img, tf.uint8)
        
        return output_img.numpy()
    
    def get_input_data(self, img_path):
        input_img = tf.io.read_file(img_path)
        input_img = tf.io.decode_jpeg(input_img, 3)
        input_img = tf.cast(input_img, tf.float32)
        input_img = (input_img - 127.5) / 127.5
        input_img = tf.image.resize(input_img, self.image_size)
        input_img = tf.expand_dims(input_img, axis=0)
        
        return input_img
 
if __name__ == "__main__":              
    app = App(title="CycleGAN 圖像轉梵谷畫", 
              window_size=(1024, 768),
              icon="icon.ico")
    app.mainloop()            