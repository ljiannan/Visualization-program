# _*_ coding: utf-8 _*_
"""
Time:     2025/1/13 14:54
Author:   ZhaoQi Cao(czq)
Version:  V 0.1
File:     gui_show_annotation.py
Describe: Write during the python at zgxmt, Github link: https://github.com/caozhaoqi
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
import os
import re
import time
import numpy as np
import threading
import queue

class KeypointViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Keypoint Annotation Viewer")
        self.geometry("800x600")

        self.image_dir = ""
        self.json_dir = "" # json directory
        self.images = []
        self.current_image_index = 0
        self.image_labels = {} #image labels
        self.annotation_status = tk.StringVar(value="")
        self.zoom_level = 1.0  # initial zoom is 100% for output
        self.preview_zoom_level = 1.0 #initial zoom for preview is 100%
        self.output_video = None
        self.is_playing = False  # Flag to control slideshow
        self.play_timer = None  # Store the timer id
        self.keypoint_labels = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
                       "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
                       "right_eye", "left_eye", "right_ear", "left_ear"] #define the keypoint labels

        # Define connections between keypoints
        self.skeleton = [
          [1, 2], [1, 15], [1, 16], [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12],
          [12, 13], [13, 14], [15, 17], [16, 18]
        ]

         #Image queue
        self.image_queue = queue.Queue()
        self.processed_frame = None #Processed frame from queue

        # Frame for image display
        self.image_frame = ttk.Frame(self)
        self.image_frame.pack(pady=10, expand=True, fill=tk.BOTH) # Make the image frame expand
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(anchor=tk.CENTER,expand=True) # Use anchor to center image

        # Frame for buttons
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(pady=5)
        self.load_button = ttk.Button(self.button_frame, text="Load Data", command=self.load_data)
        self.load_button.pack(side=tk.LEFT, padx=5)
        self.prev_button = ttk.Button(self.button_frame, text="Previous", command=self.show_prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(self.button_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.play_button = ttk.Button(self.button_frame, text="Start Processing", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)


        # Label for displaying annotation status
        self.status_label = ttk.Label(self, textvariable=self.annotation_status)
        self.status_label.pack(pady=5)
        self.after(10, self.process_queue) #start queue processor
        self.original_image = None  # Store the original image
        self.tk_image = None #Store displayed image


    def load_data(self):
        """Loads the images and json files from a directory."""
        self.image_dir = filedialog.askdirectory(title="Select Images Directory")
        if not self.image_dir:
           return #return if cancel

        self.json_dir = filedialog.askdirectory(title="Select JSON Directory")
        if not self.json_dir:
            return  # return if cancel

        self.images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.images.sort()
        json_files = [f for f in os.listdir(self.json_dir) if f.lower().endswith('.json')]
        self.image_labels = {}

        for image_file in self.images:
           image_base_name = os.path.splitext(image_file)[0]
           for json_file in json_files:
             match = re.match(r"^(.*)_response\.json$", os.path.basename(json_file)) #remove "_response" tag
             if match:
                 json_base_name = match.group(1)
                 if image_base_name == json_base_name:
                     json_path = os.path.join(self.json_dir, json_file)
                     try:
                       with open(json_path, 'r', encoding='utf-8') as infile:
                           self.image_labels[image_file] = json.load(infile)
                           print(f"JSON for {image_file} loaded correctly") #DEBUG
                     except Exception as e:
                        print(f"Error loading json {json_path}: {e}")
                        self.annotation_status.set(f"Error loading json {json_path}")
                     break
        #add debug print
        print("self.image_labels: ", self.image_labels)
        self.current_image_index = 0
        self.annotation_status.set("Images and JSON Loaded.")
        self.toggle_play() # Directly start processing,  but still showing preview

    def show_image(self):
        """Displays the current image and keypoints."""
        if not self.images:
            return
        current_image_file = self.images[self.current_image_index]
        image_path = os.path.join(self.image_dir, current_image_file)
        try:
           image = Image.open(image_path).convert("RGB")
           max_width = 800
           max_height = 600

           #Calculate the scaling ratio
           ratio = min(max_width / image.width, max_height / image.height)
           width = int(image.width * ratio * self.preview_zoom_level)
           height = int(image.height * ratio*self.preview_zoom_level)

           if width > 0 and height > 0:
              image = image.resize((width, height))  #scale image for the UI
              self.tk_image = ImageTk.PhotoImage(image)
              self.image_label.config(image=self.tk_image)
           else:
              self.image_label.config(image=None)
           self.annotation_status.set(f"Showing image: {current_image_file}")
           self.update_image_display(current_image_file)

        except Exception as e:
           print(f"Error loading images {image_path} : {e}")
           self.annotation_status.set(f"Error loading images {image_path}")


    def update_image_display(self, current_image_file):
         """Update the image display to incorporate zoom and panning"""
         if not self.images or not current_image_file:
            return
         image_path = os.path.join(self.image_dir, current_image_file)
         try:
           original_image = Image.open(image_path).convert("RGB") #open original each time.
           width = int(original_image.width* self.zoom_level)  #get original size and apply zoom
           height = int(original_image.height* self.zoom_level)
           image = original_image.resize((width,height)) if width > 0 and height > 0 else original_image #scale original image
           draw = ImageDraw.Draw(image)


           if current_image_file in self.image_labels:
            api_response = self.image_labels[current_image_file]
            if api_response and 'Data' in api_response and 'Outputs' in api_response['Data'] and api_response['Data']['Outputs']:
                for output_item in api_response['Data']['Outputs']:
                    if "Results" in output_item and output_item["Results"]:
                       for result_item in output_item["Results"]:
                            if 'Bodies' in result_item and result_item['Bodies']:
                                keypoint_dict = {}
                                for idx, body_part in enumerate(result_item['Bodies']):
                                        label = body_part.get("Label")
                                        if 'Positions' in body_part and body_part['Positions'] and label in self.keypoint_labels:
                                           for position in body_part['Positions']:
                                                points = position.get("Points")
                                                if points and len(points) == 2:
                                                  x = points[0] * original_image.width
                                                  y = points[1] * original_image.height
                                                  keypoint_dict[label] = (x, y)
                                                  draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='red')
                                                  draw.text((x + 5, y - 10), f"{idx+1}.{label}", fill="black",font = ImageFont.truetype("arial.ttf", size=10))

                                #draw connections
                                for connection in self.skeleton:
                                    start_index = connection[0] - 1
                                    end_index = connection[1] - 1
                                    if start_index < len(self.keypoint_labels) and end_index < len(self.keypoint_labels):
                                      start_label = self.keypoint_labels[start_index]
                                      end_label = self.keypoint_labels[end_index]
                                      if start_label in keypoint_dict and end_label in keypoint_dict:
                                          x1, y1 = keypoint_dict[start_label]
                                          x2, y2 = keypoint_dict[end_label]
                                          draw.line((x1,y1,x2,y2), fill="blue", width=2)
                self.annotation_status.set(f"Showing image: {current_image_file}, annotation loaded.")
           frame = np.array(image)
           frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
           self.image_queue.put(frame) #put frame to queue.
         except Exception as e:
             print(f"Error updating display {image_path} : {e}")
             self.annotation_status.set(f"Error updating display {image_path}")

    def process_queue(self):
        """Process the image queue"""
        try:
           frame = self.image_queue.get(block=False) #get without blocking
           self.processed_frame = frame
           if self.tk_image:
              image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #Convert back to pil
              self.tk_image = ImageTk.PhotoImage(image)
              self.image_label.config(image=self.tk_image)
        except queue.Empty:
           pass
        self.after(10, self.process_queue) # loop again to process the queue.



    def toggle_play(self):
       """Toggle the play mode on and off"""
       self.is_playing = not self.is_playing
       if self.is_playing:
          self.play_button.config(text="Stop Processing")
          threading.Thread(target=self.export_video).start() # Run video export in background
       else:
           self.play_button.config(text="Start Processing")


    def export_video(self):
        """Exports the video of images and annotations"""
        if not self.images:
             messagebox.showwarning("Warning","Load images and annotations first.")
             return

        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", title="Save video")
        if not output_path:
            return #return if cancel

        # Get the first image properties for video settings
        first_image_path = os.path.join(self.image_dir, self.images[0])
        first_image = Image.open(first_image_path).convert("RGB")
        output_width = first_image.width
        output_height = first_image.height
        if output_width <= 0: #check for 0 sizes and default to 1
           output_width = 1
        if output_height <= 0:
            output_height = 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(output_path, fourcc, 2, (output_width,output_height)) #fps =2
        self.current_image_index = 0
        while self.is_playing and self.current_image_index < len(self.images):
             current_image_file = self.images[self.current_image_index]
             self.update_image_display(current_image_file)
             if self.processed_frame is not None and self.output_video:
                self.output_video.write(self.processed_frame) #write frame
             self.current_image_index += 1
             time.sleep(0.02) # Delay between frames
        if self.output_video:
          self.output_video.release()
          self.output_video = None
        self.is_playing = False
        self.play_button.config(text="Export Video")
        messagebox.showinfo("Info", "Video export completed!")

    def show_next_image(self):
       """Displays the next image in the list."""
       if self.images:
           self.current_image_index = (self.current_image_index + 1) % len(self.images)
           self.show_image()
    def show_prev_image(self):
        """Displays the previous image in the list."""
        if self.images:
            self.current_image_index = (self.current_image_index - 1) % len(self.images)
            self.show_image()

if __name__ == "__main__":
    app = KeypointViewer()
    app.mainloop()