import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import cv2
from instance_norm import InstanceNormalization

class VideoCapture:
    def __init__(self, source):
        self.video_cap = cv2.VideoCapture(source)
        if not self.video_cap.isOpened():
            raise ValueError("Unable to open video source", source)
        self.width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS) 
        self.frame_counts = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = 0

    def get_frame(self):
        if self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                self.frame_num += 1
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                        self.width, self.height)
            else:
                return (ret, None, self.width, self.height)
        else:
            raise ValueError("Unable to open video to get frame",)
            
    def __del__(self):
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()

class App(tk.Tk):
    def __init__(self, title="window", window_size=(200, 200), icon=None):
        super().__init__()
        
        custom_objects={"CycleGAN>InstanceNormalization":InstanceNormalization}
        
        self.model = tf.keras.models.load_model("gen_f.h5",
                                                custom_objects=custom_objects,
                                                compile=False)
        
        self.protocol('WM_DELETE_WINDOW', self.release)
            
        self.video_path = None
        self.video_cap = None
        self.update_id = None
        self.video_out = None
        self.paint_frame = None
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.image_size = (600, 600)
        
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

        
        self.open_image_btn = ttk.Button(self.top_frame, text="選擇影片",
                                         style="btn.TButton",
                                         command=self.on_open_video)
        self.open_image_btn.grid(pady=10)
        
        self.play_btn = ttk.Button(self.top_frame, text="播放 & 轉換",
                                         style="btn.TButton",
                                         command=self.on_play_video)
        self.play_btn.grid()
        
        self.msg_label = ttk.Label(self.top_frame, text="")
        self.msg_label.grid(pady=10)
        
        self.progress = ttk.Progressbar(self.top_frame, orient='horizontal',
                                        mode='determinate',
                                        length=280)
        self.progress.grid(pady=10)
        self.progress.grid_forget()
        
        self.original_vid_canvas = tk.Canvas(self.bottom_frame, 
                                            width=self.image_size[0], 
                                            height=self.image_size[1])
        self.original_vid_canvas.grid(row=0, column=0)
        
        self.paint_vid_canvas = tk.Canvas(self.bottom_frame, 
                                            width=self.image_size[0], 
                                            height=self.image_size[1])
        self.paint_vid_canvas.grid(row=0, column=1)
        
        
    def on_open_video(self):
        f_types = [("video mp4", ".mp4")]
        self.video_path = filedialog.askopenfilename(initialdir=".", filetypes=f_types)
        if self.video_path:
            if self.update_id is not None:
                self.after_cancel(self.update_id)
                self.update_id = None
            self.video_cap = VideoCapture(self.video_path)
            self.msg_label["text"] = self.video_path.split("/")[-1]
            self.progress["value"] = 0
            self.update_frame(render_first_frame=True)
    
    def on_play_video(self):
        if self.video_cap:
            result = tk.messagebox.askyesno("儲存影片", "儲存輸出影片?")
            self.update_frame(write_output=result)
            
    def tf_convert_image(self, image):
        input_img = tf.cast(image, tf.float32)
        input_img = (input_img - 127.5) / 127.5
        input_img = tf.image.resize(input_img, (256, 256))
        input_img = tf.expand_dims(input_img, axis=0)
        
        output_img = self.model(input_img, training=True)
        output_img = tf.squeeze(output_img, axis=0)
        output_img = output_img * 127.5 + 127.5
        output_img = tf.cast(output_img, tf.uint8)
        
        return output_img.numpy()
            
    def update_frame(self, render_first_frame=False, write_output=False):
        if self.video_cap is None:
            raise ValueError("No video capture")
            
        if write_output:
            if self.video_out is None:
                self.video_out = cv2.VideoWriter("output.avi", 
                                                 fourcc=self.fourcc,
                                                 fps=self.video_cap.fps,
                                                 frameSize=(256, 256))
            if self.paint_frame is not None:
                s_frame = cv2.cvtColor(self.paint_frame, cv2.COLOR_RGB2BGR)
                self.video_out.write(s_frame)
                
            
        ret, frame, width, height = self.video_cap.get_frame()
        
        if ret and frame is not None:
            p = (self.video_cap.frame_num / self.video_cap.frame_counts) * 100
            self.progress["value"] = p
            self.progress.grid()
            
            scale = height / width
            w = int(self.image_size[0]*scale)
            h = int(self.image_size[1]*scale)
            
            frame_resized = cv2.resize(frame, (w, h))
            img = Image.fromarray(frame_resized)
            self.origin_image = ImageTk.PhotoImage(image=img)
            self.original_vid_canvas.configure(width=w, height=h)
            self.original_vid_canvas.create_image(0,0,
                                                  anchor=tk.NW,
                                                  image=self.origin_image)
            
            
            self.paint_frame = self.tf_convert_image(frame)
            frame_resized = cv2.resize(self.paint_frame, (w, h))
            img = Image.fromarray(frame_resized)
            self.paint_image = ImageTk.PhotoImage(image=img)
            self.paint_vid_canvas.configure(width=w, height=h)
            self.paint_vid_canvas.create_image(0,0,
                                                anchor=tk.NW,
                                                image=self.paint_image)
            
            
            if not render_first_frame:
                self.update_id = self.after(33, 
                                            lambda : self.update_frame(render_first_frame, 
                                                                       write_output))
        else:
            self.video_out.release()
            self.video_out = None
            self.msg_label["text"] = "影片結束!"
    
    def release(self):
        if self.video_out:
            self.video_out.release()
            self.video_out = None
        self.destroy()
            

if __name__ == "__main__":              
    app = App(title="CycleGAN 影片轉梵谷", 
              window_size=(1024, 800),
              icon="icon.ico")
    app.mainloop()            