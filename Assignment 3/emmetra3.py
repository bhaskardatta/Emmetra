import cv2
import numpy as np
from pathlib import Path

class HDRProcessor:
    def __init__(self):
        self.gamma = 1.0
        self.saturation = 1.2
        self.contrast = 1.1
        
    def load_exposure_images(self, image_paths):
        images = []
        exposure_times = [1/50, 1, 1/5]
        
        first_img = cv2.imread(str(image_paths[0]))
        if first_img is None:
            raise Exception(f"Could not load image: {image_paths[0]}")
            
        max_dimension = 1500
        height, width = first_img.shape[:2]
        scale = min(1.0, max_dimension / max(height, width))
        target_width = int(width * scale)
        target_height = int(height * scale)
        
        if scale < 1.0:
            first_img = cv2.resize(first_img, (target_width, target_height))
        images.append(first_img)
        
        for path in image_paths[1:]:
            img = cv2.imread(str(path))
            if img is None:
                raise Exception(f"Could not load image: {path}")
                
            if img.shape[:2] != (target_height, target_width):
                img = cv2.resize(img, (target_width, target_height))
            images.append(img)
        
        images_array = np.array(images, dtype=np.uint8)
        times_array = np.array(exposure_times, dtype=np.float32)
        
        return images_array, times_array
    
    def align_images(self, images):
        aligned_images = []
        reference = images[1]
        
        for img in images:
            try:
                ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001)
                
                cc, warp_matrix = cv2.findTransformECC(
                    ref_gray, img_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
                
                aligned = cv2.warpAffine(img, warp_matrix, 
                                       (img.shape[1], img.shape[0]),
                                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_images.append(aligned)
            except:
                print(f"Warning: Alignment failed for an image, using original")
                aligned_images.append(img)
        
        return np.array(aligned_images, dtype=np.uint8)
    
    def create_hdr_image(self, images, exposure_times):
        aligned_images = self.align_images(images)
        merge_debevec = cv2.createMergeDebevec()
        hdr = merge_debevec.process(aligned_images, exposure_times)
        return hdr
    
    def enhance_image(self, image):
        image = cv2.convertScaleAbs(image, alpha=self.contrast, beta=0)
        
        if self.saturation != 1.0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,1] = hsv[:,:,1] * self.saturation
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image
    
    def tone_map_image(self, hdr_image):
        tonemap = cv2.createTonemapDrago(gamma=self.gamma)
        tone_mapped = tonemap.process(hdr_image)
        tone_mapped_8bit = np.clip(tone_mapped * 255, 0, 255).astype(np.uint8)
        enhanced_image = self.enhance_image(tone_mapped_8bit)
        return enhanced_image
    
    def process_hdr(self, image_paths, output_path):
        try:
            images, exp_times = self.load_exposure_images(image_paths)
            hdr_image = self.create_hdr_image(images, exp_times)
            final_image = self.tone_map_image(hdr_image)
            
            cv2.imwrite(str(output_path), final_image)
            return final_image
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

def main():
    image_paths = [
        r"C:\Users\bhask\Downloads\DSC_2665.JPG",
        r"C:\Users\bhask\Downloads\DSC_2666.JPG",
        r"C:\Users\bhask\Downloads\DSC_2667.JPG"
    ]
    output_path = r"C:\Users\bhask\Downloads\dslr_hdr_indoor_f.jpg"
    
    processor = HDRProcessor()
    processor.process_hdr(image_paths, output_path)

if __name__ == "__main__":
    main()