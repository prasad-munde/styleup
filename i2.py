import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QFrame, QSlider)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from sklearn.cluster import KMeans

class ClothingColorDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.roi_size = 0.4  # 40% of image height
        self.current_image = None

    def initUI(self):
        self.setWindowTitle("Clothing Color Detector")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                min-width: 100px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QLabel {
                font-size: 14px;
            }
        """)
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setAlignment(Qt.AlignCenter)
        
        # Create upload button
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setFixedWidth(200)
        self.layout.addWidget(self.upload_btn, alignment=Qt.AlignCenter)
        
        # Create image display label
        self.image_label = QLabel()
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                background-color: white;
            }
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        
        # Add slider for adjusting ROI
        self.slider_label = QLabel("Adjust detection area:")
        self.layout.addWidget(self.slider_label, alignment=Qt.AlignCenter)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(20)
        self.slider.setMaximum(80)
        self.slider.setValue(40)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.update_roi)
        self.layout.addWidget(self.slider)
        
        # Create frame for color display
        self.color_frame = QFrame()
        self.color_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        color_layout = QVBoxLayout(self.color_frame)
        
        # Create color information label
        self.color_label = QLabel("Detected Clothing Colors will appear here")
        self.color_label.setStyleSheet("padding: 10px;")
        self.color_label.setAlignment(Qt.AlignCenter)
        color_layout.addWidget(self.color_label)
        
        self.layout.addWidget(self.color_frame)
        
        # Set window size
        self.setFixedSize(600, 800)

    def update_roi(self, value):
        self.roi_size = value / 100
        if self.current_image is not None:
            self.process_image(self.current_image)

    def get_clothing_region(self, image):
        """Extract the clothing region from the center of the image"""
        height, width = image.shape[:2]
        
        # Calculate ROI dimensions
        roi_width = int(width * 0.6)
        roi_height = int(height * self.roi_size)
        
        # Calculate ROI position
        roi_x = (width - roi_width) // 2
        roi_y = (height - roi_height) // 2
        
        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = 255
        
        # Draw rectangle on display image
        display_image = image.copy()
        cv2.rectangle(display_image, 
                     (roi_x, roi_y), 
                     (roi_x+roi_width, roi_y+roi_height), 
                     (0, 255, 0), 2)
        
        return mask, display_image

    def upload_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image",
                "",
                "Image Files (*.jpg *.jpeg *.png *.bmp)"
            )
            
            if file_name:
                # Read and store image array
                image = cv2.imread(file_name)
                if image is None:
                    raise ValueError("Could not load image")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_image = image
                self.process_image(image)
                
        except Exception as e:
            self.color_label.setText(f"Error: {str(e)}")

    def process_image(self, image):
        # Get clothing region mask and display image
        mask, display_image = self.get_clothing_region(image)
        
        # Resize display image
        display_image = self.resize_image(display_image, (400, 400))
        
        # Convert to QImage for display
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Display image
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
        
        # Extract ROI pixels using mask
        roi_pixels = image[mask == 255]
        dominant_colors = self.detect_colors(roi_pixels)
        self.display_color_info(dominant_colors)

    def resize_image(self, image, target_size):
        h, w = image.shape[:2]
        aspect = w/h
        
        if w > h:
            new_w = target_size[0]
            new_h = int(new_w/aspect)
        else:
            new_h = target_size[1]
            new_w = int(new_h*aspect)
            
        return cv2.resize(image, (new_w, new_h))

    def detect_colors(self, roi_pixels, n_colors=3):
        if len(roi_pixels) == 0:
            return []
            
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(roi_pixels)
        
        # Get the colors and percentages
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        percentages = [(sum(labels == i) / len(labels)) * 100 for i in range(n_colors)]
        
        # Sort by percentage
        color_info = list(zip(colors, percentages))
        color_info.sort(key=lambda x: x[1], reverse=True)
        
        return color_info

    def rgb_to_color_name(self, rgb):
        colors = {
            'Red': (255, 0, 0),
            'Green': (0, 255, 0),
            'Blue': (0, 0, 255),
            'White': (255, 255, 255),
            'Black': (0, 0, 0),
            'Yellow': (255, 255, 0),
            'Purple': (128, 0, 128),
            'Orange': (255, 165, 0),
            'Pink': (255, 192, 203),
            'Gray': (128, 128, 128),
            'Brown': (165, 42, 42),
            'Navy': (0, 0, 128),
            'Teal': (0, 128, 128),
            'Maroon': (128, 0, 0),
            'Light Blue': (173, 216, 230),
            'Dark Green': (0, 100, 0),
            'Beige': (245, 245, 220),
            'Burgundy': (128, 0, 32),
            'Khaki': (240, 230, 140),
            'Olive': (128, 128, 0)
        }
        
        min_distance = float('inf')
        closest_color = "Unknown"
        
        for color_name, color_rgb in colors.items():
            distance = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
                
        return closest_color

    def display_color_info(self, color_info):
        if not color_info:
            self.color_label.setText("No clothing colors detected!")
            return
            
        info_text = "<h3>Detected Clothing Colors:</h3><br>"
        
        for color, percentage in color_info:
            color_name = self.rgb_to_color_name(color)
            rgb_str = f"rgb({color[0]}, {color[1]}, {color[2]})"
            
            # Create color box using HTML/CSS
            color_box = f"""
                <div style='
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    background-color: {rgb_str};
                    border: 1px solid #000;
                    margin-right: 10px;
                    vertical-align: middle;
                '></div>
            """
            
            info_text += f"{color_box}<b>{color_name}</b>: {percentage:.1f}%<br>"
            info_text += f"RGB: {rgb_str}<br><br>"
        
        self.color_label.setText(info_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClothingColorDetector()
    ex.show()
    sys.exit(app.exec_())