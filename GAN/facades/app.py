import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import tensorflow as tf


class App(tk.Tk):
    def __init__(self, title="window", window_size=(200, 200), icon=None):
        super().__init__()
        
        self.model = tf.keras.models.load_model("p2p_gen_facades.keras", 
                                                        compile=False)
        self.image_path = None
        self.input_image = None
        self.output_image = None
        self.image_size = (256, 256)
        
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
        
        self.open_image_btn = ttk.Button(self, text="選擇圖片",
                                         style="btn.TButton",
                                         command=self.on_open_image)
        self.open_image_btn.pack(pady=10)
        
        self.input_image_canvas = tk.Canvas(self, 
                                            width=self.image_size[0], 
                                            height=self.image_size[1])
        self.input_image_canvas.pack_forget()
        
        self.convert_btn = ttk.Button(self, text="繪製建築",
                                      style="btn.TButton",
                                      command=self.on_convert_image)
        self.convert_btn.pack(pady=10)
        
        self.output_image_canvas = tk.Canvas(self, 
                                             width=self.image_size[0], 
                                             height=self.image_size[1])
        self.output_image_canvas.pack_forget()
    
    def on_open_image(self):
        f_types = [("image jpg", ".jpg .jpeg")]
        self.image_path = filedialog.askopenfilename(filetypes=f_types)
        with Image.open(self.image_path) as img:
            img = img.resize(self.image_size)
            self.input_image = ImageTk.PhotoImage(img)
            
        self.show_input_image()
        
    def show_input_image(self):
        if self.input_image is not None:
            
            self.input_image_canvas.create_image((0,0),
                                                 anchor=tk.NW,
                                                 image=self.input_image)
            self.input_image_canvas.pack(after=self.open_image_btn)
    
    def show_output_image(self):
        if self.output_image is not None:
            self.output_image_canvas.create_image((0,0),
                                                  anchor=tk.NW,
                                                  image=self.output_image)
            self.output_image_canvas.pack(after=self.convert_btn)
            
    def on_convert_image(self):
        if self.image_path is not None:
            output_img_arr = self.tf_convert_image(self.image_path)
            output_image_pil = Image.fromarray(output_img_arr)
            self.output_image = ImageTk.PhotoImage(output_image_pil)
            self.show_output_image()
            
    def tf_convert_image(self, img_path):
        input_data = self.get_input_data(img_path)
        output = self.model(input_data, training=True)
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
    app = App(title="GAN建築繪圖", 
              window_size=(600, 600),
              icon="icon.ico")
    app.mainloop()            