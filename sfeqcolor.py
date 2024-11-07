import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class SpatialFrequencyEqualizer:
    def __init__(self):
        # Initialize GUI application
        self.root = tk.Tk()
        self.root.title('Spatial Frequency Equalizer')

        # Load image using file dialog
        self.load_image()

        # Perform Fourier Transform on the loaded image
        self.original_fft = [
            np.fft.fftshift(np.fft.fft2(self.image[:, :, channel])) for channel in range(3)
        ]
        self.modified_fft = [np.copy(channel_fft) for channel_fft in self.original_fft]
        self.magnitude_spectrum = [np.abs(channel_fft) for channel_fft in self.original_fft]

        # Set up slider ranges and GUI components
        self.num_sliders = 5
        self.slider_values = [1.0] * self.num_sliders
        self.frequency_bands = self.calculate_frequency_bands()
        self.create_gui()

        self.update_image()
        self.root.mainloop()

    def load_image(self):
        # Ask user to select an image file
        file_path = filedialog.askopenfilename()
        if file_path:
            # Load and convert to color
            self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.image = cv2.resize(self.image, (512, 512))  # Resize for practicality
        else:
            raise FileNotFoundError("No image file selected.")

    def calculate_frequency_bands(self):
        # Divide the frequency range into bands, from low to high frequencies
        height, width = self.image.shape[:2]
        max_freq = np.sqrt((height // 2) ** 2 + (width // 2) ** 2)
        step = max_freq / self.num_sliders
        bands = [(i * step, (i + 1) * step) for i in range(self.num_sliders)]
        return bands

    def create_gui(self):
        # Convert image for display in tkinter
        self.display_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)))
        self.image_label = tk.Label(self.root, image=self.display_image)
        self.image_label.pack()

        # Create sliders to adjust frequency bands
        self.sliders = []
        for i in range(self.num_sliders):
            slider = tk.Scale(self.root, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label=f'Band {i + 1}',
                             command=lambda val, idx=i: self.update_slider(idx, val))
            slider.set(1.0)
            slider.pack(fill=tk.X)
            self.sliders.append(slider)

    def update_slider(self, band_idx, value):
        # Update slider value and modify FFT accordingly
        self.slider_values[band_idx] = float(value)
        self.apply_frequency_modifications()
        self.update_image()

    def apply_frequency_modifications(self):
        # Modify the magnitude of the FFT based on the slider values
        height, width = self.image.shape[:2]
        center_y, center_x = height // 2, width // 2
        self.modified_fft = [np.copy(channel_fft) for channel_fft in self.original_fft]

        for idx, (low, high) in enumerate(self.frequency_bands):
            gain = self.slider_values[idx]
            for channel in range(3):
                for y in range(height):
                    for x in range(width):
                        # Calculate the distance from the center of the frequency domain
                        distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                        if low <= distance < high:
                            self.modified_fft[channel][y, x] *= gain

    def update_image(self):
        # Inverse FFT to get the modified image back in the spatial domain
        modified_image = np.zeros_like(self.image)
        for channel in range(3):
            channel_image = np.fft.ifft2(np.fft.ifftshift(self.modified_fft[channel])).real
            modified_image[:, :, channel] = np.clip(channel_image, 0, 255).astype(np.uint8)

        # Update the displayed image
        self.display_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)))
        self.image_label.config(image=self.display_image)
        self.image_label.image = self.display_image

if __name__ == '__main__':
    SpatialFrequencyEqualizer()
