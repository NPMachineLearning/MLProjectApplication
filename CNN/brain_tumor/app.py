import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import cv2
import imutils


class App(tk.Tk):
    def __init__(self, title="window", window_size=(200, 200), icon=None):
        super().__init__()
        
        self.model = tf.keras.models.load_model("brain_tumor_detector.h5",
                                                compile=False)
        
        self.image_path = None
        self.input_image = None
        self.output_image = None
        self.model_image_size = (256,256)
        
        self.en_local = {"glioma": "Glioma tumor",
                         "meningioma": "Meningioma tumor",
                         "notumor": "Normal",
                         "pituitary": "Pituitary tumor"}
        
        self.ch_local = {"glioma": "膠質細胞瘤",
                         "meningioma": "腦膜瘤",
                         "notumor": "正常",
                         "pituitary": "垂體瘤"}
        
        self.setup_window(title, window_size, icon=icon)
        self.layout_ui()
        self.id_to_cls_map = self.load_id_to_class()
        
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
        

        self.desc = ttk.Label(self, text="人腦腫瘤偵測", font=("Arial", 18))
        self.desc.pack()
        
        tumor_types = list(self.en_local.keys())
        tumor_types.remove("notumor")
        tumor_names = [self.ch_local[n] for n in tumor_types]
        
        self.tumor_type_desc = ttk.Label(self, text=f'({", ".join(tumor_names)})',
                                         font=("Arial", 16))
        self.tumor_type_desc.pack()
        
        self.open_image_btn = ttk.Button(self, text="選擇 MRI 圖片",
                                         style="btn.TButton",
                                         command=self.on_open_image)
        self.open_image_btn.pack(pady=10)
        
        self.file_path_label = ttk.Label(self, text="")
        self.file_path_label.pack()
        
        self.input_image_canvas = tk.Canvas(self, 
                                            width=self.model_image_size[0], 
                                            height=self.model_image_size[1])
        self.input_image_canvas.pack_forget()
        
        self.convert_btn = ttk.Button(self, text="偵測",
                                      style="btn.TButton",
                                      command=self.detect_tumor)
        self.convert_btn.pack(pady=10)
        
        self.result = ttk.Label(self, text="", font=("Arial", 20))
        self.result.pack_forget()
        
        self.prob_result = ttk.Label(self, text="", font=("Arial", 20))
        self.prob_result.pack_forget()
        
        
    def load_id_to_class(self):
        id_to_cls = {}
        with open("id_to_class.txt", "r") as f:
            for line in f.readlines():
                print()
                target, classname = line.replace("\n", "").split("\t")
                id_to_cls[int(target)] = classname
        return id_to_cls
    
    def on_open_image(self):
        f_types = [("image jpg", ".jpg .jpeg")]
        img_path = filedialog.askopenfilename(filetypes=f_types)

        if img_path is not None and img_path != "":
            self.image_path = img_path
            with Image.open(self.image_path) as img:
                img = img.resize(self.model_image_size)
                self.input_image = ImageTk.PhotoImage(img)
                self.file_path_label["text"] = self.image_path.split("/")[-1]
                
            self.show_input_image()
            
            self.result["text"] = ""
            self.result.pack_forget()
            self.prob_result["text"] = ""
            self.prob_result.pack_forget()
        
    def show_input_image(self):
        if self.input_image is not None:
            
            self.input_image_canvas.create_image((0,0),
                                                 anchor=tk.NW,
                                                 image=self.input_image)
            self.input_image_canvas.pack(after=self.open_image_btn)
    
            
    def detect_tumor(self):
        if self.image_path is not None:
            label, prob = self.tf_detect_tumor(self.image_path)
            
            name = self.id_to_cls_map[label]
            en_name = self.en_local[name]
            ch_name = self.ch_local[name]
            self.result["text"] = f"{ch_name} ({en_name})"
            self.result.pack(after=self.convert_btn)
            
            prob = prob * 100
            self.prob_result["text"] = f"機率: {prob:.2f}%"
            
            if name != "notumor":
                self.prob_result["foreground"] = "#FF0000"
            else:
                self.prob_result["foreground"] = "#008000"
                
            self.prob_result.pack(after=self.result)
            
    def tf_detect_tumor(self, img_path):
        input_data = self.get_input_data(img_path)
        output = self.model.predict(input_data, verbose=0)
        label = tf.argmax(output, axis=1)
        label = tf.squeeze(label, axis=0)   
        
        return label.numpy(), output[0][label]
    
    def get_input_data(self, img_path):
        input_img = tf.io.read_file(img_path)
        input_img = tf.io.decode_jpeg(input_img, 3)
        input_img = self.crop_img(input_img.numpy(), 
                                   image_size=self.model_image_size)
        # input_img = tf.image.resize(input_img, self.model_image_size)
        input_img = tf.cast(input_img, tf.float32)
        input_img = tf.expand_dims(input_img, axis=0)
        
        return input_img
    
    def crop_img(self, img, image_size=(256, 256)):
        """
        Finds the extreme points on the image and crops the rectangular out of them
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
    
        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
    
        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = 0
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        
        # resize image
        new_img = cv2.resize(new_img, image_size)
        
        return new_img
 
if __name__ == "__main__":              
    app = App(title="CNN腫瘤偵測", 
              window_size=(600, 600),
              icon="icon.ico")
    app.mainloop()            