import numpy as np
import cv2
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class ISPPipeline:
    def __init__(self):
        self.raw_image = None
        self.processed_image = None
        
    def load_raw_image(self, path):
        # Read 12-bit RAW image
        with open(path, 'rb') as f:
            # Read raw bytes
            raw_data = np.fromfile(f, dtype=np.uint16)
            
            # Assuming 1920x1280 resolution as mentioned in the assignment
            width, height = 1920, 1280
            
            # Reshape the data into 2D array
            raw_data = raw_data.reshape((height, width))
            
            # Convert to proper 12-bit format
            raw_data = raw_data & 0x0FFF  # Mask to 12 bits
            self.raw_image = raw_data
            
            return self.raw_image

    def demosaic(self, bayer_image):
        # Implementing 5x5 edge-based demosaicing
        blue = np.zeros_like(bayer_image, dtype=np.float32)
        red = np.zeros_like(bayer_image, dtype=np.float32)
        green = np.zeros_like(bayer_image, dtype=np.float32)
        
        # Assuming GRBG Bayer pattern
        # Green pixels
        green[0::2, 0::2] = bayer_image[0::2, 0::2]  # Green in red row
        green[1::2, 1::2] = bayer_image[1::2, 1::2]  # Green in blue row
        
        # Red pixels
        red[0::2, 1::2] = bayer_image[0::2, 1::2]
        
        # Blue pixels
        blue[1::2, 0::2] = bayer_image[1::2, 0::2]
        
        # Interpolate missing values using 5x5 kernel
        kernel = np.array([
            [1, 2, 4, 2, 1],
            [2, 4, 8, 4, 2],
            [4, 8, 16, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1]
        ], dtype=np.float32) / 64.0
        
        red = cv2.filter2D(red, -1, kernel)
        green = cv2.filter2D(green, -1, kernel)
        blue = cv2.filter2D(blue, -1, kernel)
        
        return cv2.merge([blue, green, red])
    
    def white_balance(self, image, gray_world=True):
        if gray_world:
            b, g, r = cv2.split(image)
            b_avg = np.mean(b)
            g_avg = np.mean(g)
            r_avg = np.mean(r)
            
            gray = (b_avg + g_avg + r_avg) / 3
            kb = gray / b_avg if b_avg != 0 else 1
            kg = gray / g_avg if g_avg != 0 else 1
            kr = gray / r_avg if r_avg != 0 else 1
            
            b = cv2.multiply(b, kb)
            g = cv2.multiply(g, kg)
            r = cv2.multiply(r, kr)
            
            return cv2.merge([b, g, r])
        return image
    
    def denoise(self, image, kernel_size=5, sigma=1.0):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def gamma_correction(self, image, gamma=2.2):
        normalized = image / 4095.0  # Normalize 12-bit to [0,1]
        gamma_corrected = np.power(normalized, 1/gamma)
        return (gamma_corrected * 255).astype(np.uint8)
    
    def sharpen(self, image, amount=1.0):
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(image, -1, kernel * amount)
        return np.clip(sharpened, 0, 255)

    def process_automatic_combinations(self, raw_image_path, save_directory):
        """
        Automatically process the image with all required combinations
        """
        # Load the RAW image
        raw_image = self.load_raw_image(raw_image_path)
        results = {}
        
        # Combination 1: Demosaic + Gamma
        image1 = self.demosaic(raw_image.copy())
        image1 = self.gamma_correction(image1)
        results['Demosaic_Gamma'] = image1
        
        # Combination 2: Demosaic + White balance + Gamma
        image2 = self.demosaic(raw_image.copy())
        image2 = self.white_balance(image2)
        image2 = self.gamma_correction(image2)
        results['Demosaic_WB_Gamma'] = image2
        
        # Combination 3: Demosaic + White Balance + Denoise + Gamma
        image3 = self.demosaic(raw_image.copy())
        image3 = self.white_balance(image3)
        image3 = self.denoise(image3)
        image3 = self.gamma_correction(image3)
        results['Demosaic_WB_Denoise_Gamma'] = image3
        
        # Combination 4: Demosaic + White Balance + Denoise + Gamma + Sharpen
        image4 = self.demosaic(raw_image.copy())
        image4 = self.white_balance(image4)
        image4 = self.denoise(image4)
        image4 = self.gamma_correction(image4)
        image4 = self.sharpen(image4)
        results['Demosaic_WB_Denoise_Gamma_Sharpen'] = image4
        
        # Save all results in the specified directory
        saved_paths = []
        for name, image in results.items():
            save_path = os.path.join(save_directory, f'{name}.png')
            cv2.imwrite(save_path, image)
            saved_paths.append(save_path)
        
        return results, saved_paths

class ISPTuningTool:
    def __init__(self):
        self.isp = ISPPipeline()
        self.raw_image = None
        self.processed_image = None
        self.raw_image_path = None
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("ISP Tuning Tool")
        
        # Controls frame
        controls_frame = ttk.Frame(self.root)
        controls_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Load image button
        ttk.Button(controls_frame, text="Load RAW Image", 
                  command=self.load_image).pack(pady=5)
        
        # Automatic processing button
        ttk.Button(controls_frame, text="Run Automatic Processing", 
                  command=self.run_automatic_processing).pack(pady=5)
        
        # Pipeline steps checkboxes
        self.steps_vars = {
            'demosaic': tk.BooleanVar(value=True),
            'wb': tk.BooleanVar(value=True),
            'denoise': tk.BooleanVar(value=True),
            'gamma': tk.BooleanVar(value=True),
            'sharpen': tk.BooleanVar(value=True)
        }
        
        ttk.Label(controls_frame, text="Manual Pipeline Steps:").pack(anchor=tk.W, pady=5)
        for step, var in self.steps_vars.items():
            ttk.Checkbutton(controls_frame, text=step, variable=var, 
                           command=self.process_image).pack(anchor=tk.W)
        
        # Parameters
        ttk.Label(controls_frame, text="Parameters:").pack(anchor=tk.W, pady=5)
        self.params = {
            'Denoise Sigma': tk.DoubleVar(value=1.0),
            'Gamma': tk.DoubleVar(value=2.2),
            'Sharpen Amount': tk.DoubleVar(value=1.0)
        }
        
        for param, var in self.params.items():
            ttk.Label(controls_frame, text=param).pack(anchor=tk.W)
            ttk.Scale(controls_frame, from_=0.1, to=5.0, variable=var,
                     command=self.process_image).pack(anchor=tk.W)
        
        # Process button
        ttk.Button(controls_frame, text="Process Image", 
                  command=self.process_image).pack(pady=10)
        
        # Save button
        ttk.Button(controls_frame, text="Save Processed Image", 
                  command=self.save_image).pack(pady=5)
        
        # Image display
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)

    def run_automatic_processing(self):
        if self.raw_image_path is None:
            messagebox.showerror("Error", "Please load a RAW image first")
            return

        # Ask user for save directory
        save_directory = filedialog.askdirectory(title="Select Directory to Save Processed Images")
        if not save_directory:  # User cancelled
            return

        try:
            results, saved_paths = self.isp.process_automatic_combinations(self.raw_image_path, save_directory)
            
            # Create success message with full paths
            success_message = "Automatic processing complete!\nFiles saved:\n" + "\n".join(saved_paths)
            messagebox.showinfo("Success", success_message)
            
            # Display the final result
            self.processed_image = results['Demosaic_WB_Denoise_Gamma_Sharpen']
            self.display_image(self.processed_image)
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("RAW files", "*.raw"), ("All files", "*.*")])
        if file_path:
            self.raw_image_path = file_path
            self.raw_image = self.isp.load_raw_image(file_path)
            self.process_image()
    
    def process_image(self, *args):
        if self.raw_image is None:
            return
            
        image = self.raw_image.copy()
        steps = []
        
        if self.steps_vars['demosaic'].get():
            image = self.isp.demosaic(image)
            steps.append('demosaic')
            
        if self.steps_vars['wb'].get():
            image = self.isp.white_balance(image)
            steps.append('wb')
            
        if self.steps_vars['denoise'].get():
            image = self.isp.denoise(image, sigma=self.params['Denoise Sigma'].get())
            steps.append('denoise')
            
        if self.steps_vars['gamma'].get():
            image = self.isp.gamma_correction(image, self.params['Gamma'].get())
            steps.append('gamma')
            
        if self.steps_vars['sharpen'].get():
            image = self.isp.sharpen(image, self.params['Sharpen Amount'].get())
            steps.append('sharpen')
        
        self.processed_image = image
        self.display_image(image)
    
    def display_image(self, image):
        # Convert to 8-bit if needed
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)
        
        # Resize for display
        height, width = image.shape[:2]
        max_size = 800
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert to PIL format
        if len(image.shape) == 2:  # If grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        
        # Update canvas
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
    
    def save_image(self):
        if self.processed_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        
        if file_path:
            if self.processed_image.dtype != np.uint8:
                save_image = (self.processed_image / self.processed_image.max() * 255).astype(np.uint8)
            else:
                save_image = self.processed_image
                
            cv2.imwrite(file_path, save_image)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    tool = ISPTuningTool()
    tool.run()