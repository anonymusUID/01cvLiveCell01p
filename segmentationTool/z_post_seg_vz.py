from tkinter import (Tk, Label, Button, Scale, filedialog, Canvas, Frame, LEFT, Scrollbar, HORIZONTAL,
                     TOP, X, W, BOTH, RIGHT, VERTICAL, StringVar, OptionMenu, IntVar, Checkbutton, Toplevel,
                     Entry, messagebox,Menu)
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.measure import label, regionprops
import os
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance as dist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ContourEditor:
    def __init__(self, image_path):
        self.root = Tk()
        self.root.title("Contour Editor")
        self.root.geometry("800x800")  # Increased height for new sliders

        # Load the image
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError("Image not found or invalid path.")
        self.image = self.original_image.copy()
        
        # Initialize parameters for contour refinement
        self.force_constant = 5.0
        self.decay_exponent = 2.0
        self.gradient_threshold = 30
        self.epsilon = 0.0001
        
        self.contours_list = self.detect_contours(spacing=8, use_adaptive=False)  # Use fixed threshold on load
        self.selected_contour = None
        self.selected_point = None
        self.kernel_size = 3  # Default kernel size
        self.zoom_scale = 1.0  # Default zoom scale

        # Parameters for thresholdArea_Elimination
        self.threshold_area = 35
        self.area_mode = "fill"  # "fill" or "mark"

        # Parameters for AspectRatio_Elimination
        self.min_aspect_ratio = 0.9
        self.max_aspect_ratio = 1.2
        self.aspect_ratio_mode = "fill"  # "fill" or "mark"

        # Parameters for isoperimetric_algo
        self.min_area = 5
        self.max_area = 250
        self.min_circularity = 0.31
        self.max_circularity = 2.0
        self.isoperimetric_mode = "remove"  # "detect" or "remove"

        # Parameters for blur
        self.blur_kernel_size = 5
        self.blur_type = "gaussian"  # "gaussian", "median", "average", "bilateral"

        # Parameters for eccentricity filtering
        self.min_eccentricity = 0.2
        self.max_eccentricity = 0.8
        self.eccentricity_mode = "fill"  # "fill" or "mark"

        # For undo/redo functionality
        self.undo_stack = []  # Stores previous image states
        self.redo_stack = []  # Stores states that were undone
        self.max_history = 6  # Maximum number of undo steps to store

        # Initialize tracked contours list
        self.tracked_contours = []

        self.operations_history = []  # List to store operation history


        # Initialize UI
        self.init_ui()

    def load_new_image(self):
        """Load a new image while keeping the application running."""
        # Ask user to select an image file
        image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        
        if image_path:  # Ensure the user selects a file
            try:
                # Clear undo/redo stacks
                self.undo_stack = []
                self.redo_stack = []
                
                # Clear tracked contours
                self.tracked_contours = []
                
                # Only update tracking tab if it exists
                if hasattr(self, 'contour_tree'):
                    self.update_tracking_tab()
                
                # Load the new image
                self.image_path = image_path
                self.original_image = cv2.imread(image_path)
                
                if self.original_image is None:
                    messagebox.showerror("Error", "Failed to load image. Invalid file.")
                    return
                    
                self.image = self.original_image.copy()
                
                # Reset zoom
                self.zoom_scale = 1.0
                
                # Update contours and display
                self.contours_list = self.detect_contours(spacing=8)
                self.update_display()
                
                # Update window title with new image path
                self.root.title(f"Contour Editor - {os.path.basename(image_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                print(f"Error in load_new_image: {str(e)}")


    def save_state(self):
        """Save current image state to the undo stack."""
        # Create a deep copy of the current image
        current_state = self.image.copy()
        
        # Add to undo stack, maintaining maximum size
        self.undo_stack.append(current_state)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)  # Remove oldest state if exceeding limit
        
        # Clear redo stack when a new action is performed
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        current_state = self.image.copy()
        self.redo_stack.append(current_state)
        self.image = self.undo_stack.pop()
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        if self.operations_history:
            self.operations_history.pop()  # Remove last operation
        self.update_operations_history()

    def redo(self):
        if not self.redo_stack:
            return
        current_state = self.image.copy()
        self.undo_stack.append(current_state)
        redo_state = self.redo_stack.pop()
        self.image = redo_state
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        # Note: Redo doesn’t re-add to history; it’s assumed to be restored from undo

    def open_aspect_dialog(self):
        """Open the aspect ratio dialog."""
        dialog = self.open_dialog("Aspect Ratio Filter", self.setup_aspect_tab)
        return dialog

    def open_morphology_dialog(self, operation):
        """Open the morphology dialog."""
        def setup_func(dialog):
            control_frame = Frame(dialog)
            control_frame.pack(fill="x", padx=10, pady=10)
            
            # Kernel size slider
            self.kernel_label = Label(control_frame, text=f"Kernel Size: {self.kernel_size}")
            self.kernel_label.pack(anchor="w")
            
            kernel_slider = Scale(control_frame, from_=1, to=21, orient=HORIZONTAL, 
                                command=self.update_kernel_size)
            kernel_slider.set(self.kernel_size)
            kernel_slider.pack(fill="x")
            
            # Apply button
            Button(control_frame, text="Apply", 
                command=lambda: self.apply_morphology(operation)).pack(fill="x", pady=10)
        
        dialog = self.open_dialog(f"{operation.capitalize()} Operation", setup_func)
        return dialog

    def open_blur_dialog(self):
        """Open the blur dialog."""
        dialog = self.open_dialog("Blur", self.setup_blur_tab)
        return dialog

    def open_eccentricity_dialog(self):
        """Open the eccentricity dialog."""
        dialog = self.open_dialog("Eccentricity Filter", self.setup_eccentricity_tab)
        return dialog

    def open_iso_dialog(self):
        """Open the circularity dialog."""
        dialog = self.open_dialog("Circularity Filter", self.setup_iso_tab)
        return dialog


    def open_fill_dialog(self):
        """Open the fill options dialog."""
        dialog = self.open_dialog("Fill Options", self.setup_fill_tab)
        return dialog


    def open_range_dialog(self, slider, label_widget, value_formatter=None, value_updater=None):
        """Open a dialog to set custom range for a slider."""
        # Release grab and topmost from parent dialogs
        for attr_name in dir(self):
            if attr_name.endswith('_dialog'):
                dialog = getattr(self, attr_name)
                if hasattr(dialog, 'winfo_exists') and dialog.winfo_exists():
                    if dialog.grab_status():
                        dialog.grab_release()
                    dialog.attributes("-topmost", False)
        
        dialog = Toplevel(self.root)
        dialog.title("Set Custom Range")
        dialog.geometry("300x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        # Get current slider values
        current_min = slider.cget("from")
        current_max = slider.cget("to")
        current_value = slider.get()

        # Input frames
        min_frame = Frame(dialog)
        min_frame.pack(fill="x", padx=10, pady=5)
        Label(min_frame, text="Minimum value:").pack(side="left")
        min_var = StringVar(value=str(current_min))
        min_entry = Entry(min_frame, textvariable=min_var, width=10)
        min_entry.pack(side="right")

        max_frame = Frame(dialog)
        max_frame.pack(fill="x", padx=10, pady=5)
        Label(max_frame, text="Maximum value:").pack(side="left")
        max_var = StringVar(value=str(current_max))
        max_entry = Entry(max_frame, textvariable=max_var, width=10)
        max_entry.pack(side="right")

        # Apply changes
        def apply_range():
            try:
                new_min = float(min_var.get())
                new_max = float(max_var.get())
                if new_min >= new_max:
                    messagebox.showerror("Invalid Range", "Minimum must be less than maximum")
                    return
                slider.config(from_=new_min, to=new_max)
                if current_value < new_min:
                    slider.set(new_min)
                elif current_value > new_max:
                    slider.set(new_max)
                if value_formatter and label_widget:
                    label_widget.config(text=value_formatter(slider.get()))
                if value_updater:
                    value_updater(slider.get())
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers")

        # Close dialog and restore parent
        def on_dialog_close():
            dialog.destroy()
            for attr_name in dir(self):
                if attr_name.endswith('_dialog'):
                    parent_dialog = getattr(self, attr_name)
                    if hasattr(parent_dialog, 'winfo_exists') and parent_dialog.winfo_exists():
                        parent_dialog.attributes("-topmost", True)

        # Buttons (single frame)
        button_frame = Frame(dialog)
        button_frame.pack(fill="x", padx=10, pady=10)
        Button(button_frame, text="Cancel", command=on_dialog_close).pack(side="right")
        Button(button_frame, text="Apply", command=apply_range).pack(side="right", padx=5)

        min_entry.focus_set()


    def thresholdArea_Elimination(self):
        """Eliminate contours below a certain area threshold."""
        self.save_state()
        self.update_display()
        cell_image = self.image.copy()
        if len(cell_image.shape) != 2:
            gray_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = cell_image.copy()
        
        # Find contours with hierarchy information
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.area_mode == "mark":
            # For mark mode: Start with the original image
            result = cell_image.copy()
            
            # Draw red outlines only on contours above threshold
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.threshold_area:
                    cv2.drawContours(result, [contour], -1, (0, 0, 255), thickness=3)  # Red contours
            
            self.image = result
        else:  # Fill mode
            # Create a blank mask
            mask = np.zeros_like(gray_image, dtype=np.uint8)
            
            # Process contours based on their area and hierarchy
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area >= self.threshold_area:
                    # Fill this contour
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                    
                    # Check if this contour has child contours (holes)
                    if hierarchy is not None and len(hierarchy) > 0:
                        # Get the first child of this contour
                        child = hierarchy[0][i][2]
                        while child != -1:  # -1 indicates no more children
                            child_area = cv2.contourArea(contours[child])
                            if child_area >= self.threshold_area:
                                # This is a large enough hole, cut it out
                                cv2.drawContours(mask, [contours[child]], -1, 0, thickness=cv2.FILLED)
                            
                            # Move to the next child
                            child = hierarchy[0][child][0]
            
            # Apply the mask to create the final image
            if len(self.image.shape) == 3:
                self.image = cv2.bitwise_and(cell_image, cell_image, mask=mask)
            else:
                self.image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        # Calculate metrics if ground truth is available
        metrics = self.calculate_metrics_for_history()

        # Create operation entry
        operation_entry = {
            "name": "Area Threshold Elimination",
            "params": {
                "Threshold Area": self.threshold_area,
                "Mode": self.area_mode
            }
        }

        # Add metrics if available
        if metrics:
            operation_entry["metrics"] = metrics

        # Append to operations history
        self.operations_history.append(operation_entry)

        # Update the operations history display
        self.update_operations_history()



    def AspectRatio_Elimination(self):
        self.save_state()
        """Eliminate contours based on aspect ratio."""
        if len(self.image.shape) == 2:
            _image = self.image.copy()
            cell_image = self.image.copy()
        else:
            _image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            cell_image = _image.copy()
        
        _, binary = cv2.threshold(cell_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.aspect_ratio_mode == "mark":
            cell_image = cv2.cvtColor(cell_image, cv2.COLOR_GRAY2BGR)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:  # Avoid division by zero
                continue
            aspect_ratio = float(w) / h
            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                if self.aspect_ratio_mode == "fill":
                    cv2.drawContours(cell_image, [contour], -1, 0, thickness=cv2.FILLED)  # Fill black
                elif self.aspect_ratio_mode == "mark":
                    cv2.drawContours(cell_image, [contour], -1, (0, 0, 255), thickness=3)  # Red contours
        
        if self.aspect_ratio_mode == "fill":
            if len(cell_image.shape) == 2:
                self.image = cv2.cvtColor(cell_image, cv2.COLOR_GRAY2BGR)
            else:
                self.image = cell_image
        else:
            self.image = cell_image
        
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        # Calculate metrics if ground truth is available
        metrics = self.calculate_metrics_for_history()
        
        # Create operation entry
        operation_entry = {
            "name": "Aspect Ratio Elimination",
            "params": {
                "Min Aspect Ratio": f"{self.min_aspect_ratio:.1f}",
                "Max Aspect Ratio": f"{self.max_aspect_ratio:.1f}",
                "Mode": self.aspect_ratio_mode
            }
        }
        
        # Add metrics if available
        if metrics:
            operation_entry["metrics"] = metrics
        
        # Append to operations history (only once)
        self.operations_history.append(operation_entry)
        self.update_operations_history()
        


    def eccentricity_elimination(self, use_adaptive=False):
        self.save_state()
        """Eliminate contours within specified eccentricity and area range, optimized for somatic cells."""
        if len(self.image.shape) == 2:
            _image = self.image.copy()
            cell_image = self.image.copy()
        else:
            _image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            cell_image = _image.copy()

        # Preprocessing adjustments
        cell_image = cv2.GaussianBlur(cell_image, (3, 3), 0)
        if use_adaptive:
            binary = cv2.adaptiveThreshold(
                cell_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 7, 1
            )
        else:
            _, binary = cv2.threshold(cell_image, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Debug: Save binary mask to inspect preprocessing
        cv2.imwrite("binary_mask.png", binary)

        # Convert binary image to labeled regions using scikit-image
        labeled_image = label(binary // 255)  # Convert to 0/1 binary for labeling
        regions = regionprops(labeled_image)

        # Create mask for filtering
        if self.eccentricity_mode == "mark":
            mask = cv2.cvtColor(_image, cv2.COLOR_GRAY2BGR)
        else:
            mask = np.ones_like(_image, dtype=np.uint8) * 255

        filtered_count = 0
        for region in regions:
            area = region.area
            if area < self.min_area or area > self.max_area:
                continue

            # Get eccentricity from regionprops
            eccentricity = region.eccentricity  # This is robust and handles irregular shapes
            if np.isnan(eccentricity) or eccentricity < 0 or eccentricity > 1:
                print(f"Invalid eccentricity for region (area: {area}): {eccentricity}")
                continue

            # Debug: Print eccentricity values
            print(f"Region area: {area}, Eccentricity: {eccentricity:.3f}")

            if self.min_eccentricity <= eccentricity <= self.max_eccentricity:
                filtered_count += 1
                # Get the coordinates of the region to draw on the mask
                coords = region.coords  # Coordinates of the region pixels
                if self.eccentricity_mode == "fill":
                    mask[coords[:, 0], coords[:, 1]] = 0  # Black out the region
                elif self.eccentricity_mode == "mark":
                    # Convert region to contour for marking
                    contours, _ = cv2.findContours(
                        (labeled_image == region.label).astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        cv2.drawContours(mask, contours, -1, (0, 0, 255), thickness=3)

        print(f"Filtered {filtered_count} regions based on eccentricity.")

        if self.eccentricity_mode == "fill":
            cv2.imwrite("eccentricity_mask.png", mask)

        if self.eccentricity_mode == "fill":
            if len(self.image.shape) == 3:
                self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
            else:
                self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
        else:
            self.image = mask

        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        metrics = self.calculate_metrics_for_history()

        operation_entry = {
            "name": "Eccentricity Elimination",
            "params": {
                "Min Area": self.min_area,
                "Max Area": self.max_area,
                "Min Eccentricity": f"{self.min_eccentricity:.2f}",
                "Max Eccentricity": f"{self.max_eccentricity:.2f}",
                "Mode": self.eccentricity_mode,
                "Use Adaptive Thresholding": use_adaptive
            }
        }

        if metrics:
            operation_entry["metrics"] = metrics

        self.operations_history.append(operation_entry)
        self.update_operations_history()


    def isoperimetric_algo(self, recursive=False):
        self.save_state()
        
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
        
        # Use adaptive thresholding for better contour detection
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for contours that match the filter criteria
        mask = np.zeros_like(gray)
        
        # Track which contours have been processed
        processed_contours = set()
        
        for i, contour in enumerate(contours):
            # Skip if hierarchy is invalid
            # Calculate properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Avoid division by zero
            if perimeter <= 0:
                continue
                
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # Apply filters
            if self.min_area <= area <= self.max_area and self.min_circularity <= circularity <= self.max_circularity:
                if self.isoperimetric_mode == 'remove':
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Apply mask correctly
        if self.isoperimetric_mode == 'remove':
            # Use bitwise_not(mask) to keep everything except the matched contours
            self.image = cv2.bitwise_and(self.image, self.image, mask=cv2.bitwise_not(mask))
        else:  # 'detect' mode
            # Highlight the detected contours
            if len(self.image.shape) == 3:
                highlight = self.image.copy()
                for i, contour in enumerate(contours):
                    if cv2.countNonZero(cv2.bitwise_and(mask, cv2.drawContours(np.zeros_like(mask), [contour], -1, 255, 1))) > 0:
                        cv2.drawContours(highlight, [contour], -1, (0, 0, 255), 2)
                self.image = highlight
        
        # Update contours and display
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        metrics = self.calculate_metrics_for_history()

        operation_entry = {
            "name": "Circularity Filter",
            "params": {
                "Min Area": self.min_area,
                "Max Area": self.max_area,
                "Min Circularity": f"{self.min_circularity:.2f}",
                "Max Circularity": f"{self.max_circularity:.2f}",
                "Mode": self.isoperimetric_mode
            }
        }

        if metrics:
            operation_entry["metrics"] = metrics

        self.operations_history.append(operation_entry)
        self.update_operations_history()



    def apply_blur(self):
        """Apply blur to the image."""
        self.save_state()
        kernel_size = self.blur_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        if self.blur_type == "gaussian":
            blurred_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        elif self.blur_type == "median":
            blurred_image = cv2.medianBlur(self.image, kernel_size)
        elif self.blur_type == "average":
            blurred_image = cv2.blur(self.image, (kernel_size, kernel_size))
        elif self.blur_type == "bilateral":
            blurred_image = cv2.bilateralFilter(self.image, kernel_size, 75, 75)
        else:
            return
        
        self.image = blurred_image
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        metrics = self.calculate_metrics_for_history()

        operation_entry = {
            "name": "Blur",
            "params": {
                "Blur Kernel Size": self.blur_kernel_size,
                "Blur Type": self.blur_type
            }
        }

        if metrics:
            operation_entry["metrics"] = metrics

        self.operations_history.append(operation_entry)
        self.update_operations_history()


    def update_threshold_area(self, value):
        """Update threshold area parameter."""
        self.threshold_area = int(value)
        self.threshold_area_label.config(text=f"Threshold Area: {self.threshold_area}")

    def set_area_mode(self, mode):
        """Set the area mode."""
        self.area_mode = mode

    def update_min_aspect_ratio(self, value):
        """Update minimum aspect ratio parameter."""
        self.min_aspect_ratio = float(value) / 10.0
        self.min_aspect_ratio_label.config(text=f"Min Aspect Ratio: {self.min_aspect_ratio:.1f}")

    def update_max_aspect_ratio(self, value):
        """Update maximum aspect ratio parameter."""
        self.max_aspect_ratio = float(value) / 10.0
        self.max_aspect_ratio_label.config(text=f"Max Aspect Ratio: {self.max_aspect_ratio:.1f}")

    def set_aspect_ratio_mode(self, mode):
        """Set the aspect ratio mode."""
        self.aspect_ratio_mode = mode

    def update_min_area(self, value):
        """Update minimum area parameter for isoperimetric algorithm."""
        self.min_area = int(value)
        self.min_area_label.config(text=f"Min Area: {self.min_area}")

    def update_max_area(self, value):
        """Update maximum area parameter for isoperimetric algorithm."""
        self.max_area = int(value)
        self.max_area_label.config(text=f"Max Area: {self.max_area}")

    def update_min_circularity(self, value):
        """Update minimum circularity parameter."""
        self.min_circularity = float(value) / 100.0
        self.min_circularity_label.config(text=f"Min Circularity: {self.min_circularity:.2f}")

    def update_max_circularity(self, value):
        """Update maximum circularity parameter."""
        self.max_circularity = float(value) / 100.0
        self.max_circularity_label.config(text=f"Max Circularity: {self.max_circularity:.2f}")

    def set_isoperimetric_mode(self, mode):
        """Set the isoperimetric mode."""
        self.isoperimetric_mode = mode

    def update_blur_kernel_size(self, value):
        """Update blur kernel size parameter."""
        self.blur_kernel_size = int(value)
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1  # Ensure odd kernel size
        self.blur_kernel_label.config(text=f"Blur Kernel Size: {self.blur_kernel_size}")

    def set_blur_type(self, blur_type):
        """Set the blur type."""
        self.blur_type = blur_type

    def update_min_eccentricity(self, value):
        """Update minimum eccentricity parameter."""
        self.min_eccentricity = float(value) / 100.0
        self.min_eccentricity_label.config(text=f"Min Eccentricity: {self.min_eccentricity:.2f}")

    def update_max_eccentricity(self, value):
        """Update maximum eccentricity parameter."""
        self.max_eccentricity = float(value) / 100.0
        self.max_eccentricity_label.config(text=f"Max Eccentricity: {self.max_eccentricity:.2f}")

    def set_eccentricity_mode(self, mode):
        """Set the eccentricity mode."""
        self.eccentricity_mode = mode


    def detect_contours(self, spacing=8, use_adaptive=False):
        """Detect contours and return a list of sampled boundary points.
        Args:
            spacing (int): Minimum distance between sampled points.
            use_adaptive (bool): If True, use adaptive thresholding; otherwise, use fixed threshold.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if use_adaptive:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours detected.")
            return []

        sampled_contours = []
        for contour in contours:
            reduced_contour = [contour[0][0]]
            for i in range(1, len(contour)):
                if np.linalg.norm(contour[i][0] - reduced_contour[-1]) >= spacing:
                    reduced_contour.append(contour[i][0])
            sampled_contours.append(np.array(reduced_contour))
        return sampled_contours

    def draw_contours(self):
        """Draw contours with points."""
        temp = self.image.copy()
        if not self.contours_list:
            return temp

        for contour in self.contours_list:
            cv2.polylines(temp, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
        return temp

    def update_display(self, use_highlighted=False):
        """Update the displayed image with proper scrolling."""
        if use_highlighted and hasattr(self, 'preview_image'):
            display_img = self.preview_image
        elif use_highlighted and hasattr(self, 'image_with_highlight'):
            display_img = self.image_with_highlight
        else:
            display_img = self.draw_contours()
        
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_img)
        
        # Calculate new dimensions based on zoom scale
        new_width = int(img.width * self.zoom_scale)
        new_height = int(img.height * self.zoom_scale)
        
        # Resize the image using PIL
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage for Tkinter
        self.tk_image = ImageTk.PhotoImage(img_resized)
        
        # Update the canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        
        # Update the scroll region to match the resized image
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))



    def apply_morphology(self, operation):
        """Apply erosion or dilation on contours."""
        self.save_state()
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        if operation == "dilate":
            modified = cv2.dilate(gray, kernel, iterations=1)
        elif operation == "erode":
            modified = cv2.erode(gray, kernel, iterations=1)
        else:
            return

        _, binary = cv2.threshold(modified, 127, 255, cv2.THRESH_BINARY)
        self.image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        # Create operation entry
        metrics = self.calculate_metrics_for_history()
        operation_entry = {
            "name": f"Morphological {operation.capitalize()}",
            "params": {
                "Operation": operation,
                "Kernel Size": self.kernel_size
            }
        }
        
        # Add metrics if available
        if metrics:
            operation_entry["metrics"] = metrics
        
        self.operations_history.append(operation_entry)
        self.update_operations_history()

    def reset_image(self):
        self.image = self.original_image.copy()
        self.contours_list = self.detect_contours(spacing=8)
        self.zoom_scale = 1.0
        self.operations_history.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_display()
        self.update_operations_history()

    def update_coordinates(self, event):
        """Update coordinate display when mouse moves over the image."""
        x, y = self.canvas_to_image_coords(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.coord_display.config(text=f"X: {x}, Y: {y}")


    def save_image(self):


        file_path= image_path
    
        if file_path:
            cv2.imwrite(file_path, self.image)
            print(f"Image saved as {file_path}")
            

    def update_kernel_size(self, value):
        """Update the kernel size for erosion/dilation ensuring odd values."""
        self.kernel_size = int(value)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1  # Ensure odd kernel size
        self.kernel_label.config(text=f"Kernel Size: {self.kernel_size}")

    def zoom_in(self):
        """Zoom in the image."""
        self.zoom_scale *= 1.2  # Increase zoom scale by 20%
        self.update_display()

    def zoom_out(self):
        """Zoom out the image."""
        self.zoom_scale /= 1.2  # Decrease zoom scale by 20%
        self.update_display()

    def canvas_to_image_coords(self, x, y):
        """Convert canvas coordinates to image coordinates."""
        return int(x / self.zoom_scale), int(y / self.zoom_scale)

    def on_mouse_click(self, event):
        """Handle mouse clicks for selecting contour points and getting contour details."""
        # Convert canvas coordinates to image coordinates
        x, y = self.canvas_to_image_coords(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if hasattr(self, 'color_analysis_mode') and self.color_analysis_mode:
            # Find all contours that contain the clicked point
            matching_contours = []
            for i, contour in enumerate(self.contours_list):
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    area = cv2.contourArea(contour)
                    if area >= 30:  # Ignore small contours
                        matching_contours.append((i, contour, area))
            
            if matching_contours:
                # Sort by area (smallest first - innermost contours)
                matching_contours.sort(key=lambda x: x[2])
                # Analyze the innermost contour (smallest area)
                i, contour, _ = matching_contours[0]
                self.analyze_contour_color(i, contour)
            return
        # Check if we're in contour info mode
        if hasattr(self, 'contour_info_mode') and self.contour_info_mode:
            # Find all contours that contain the clicked point
            matching_contours = []
            for i, contour in enumerate(self.contours_list):
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    area = cv2.contourArea(contour)
                    if area >= 30:  # Ignore small contours
                        matching_contours.append((i, contour, area))
            
            if matching_contours:
                # Sort by area (smallest first - innermost contours)
                matching_contours.sort(key=lambda x: x[2])
                
                # Select the innermost contour (smallest area)
                i, contour, _ = matching_contours[0]
                self.extract_contour_properties(i, contour)
            return
        
        # Check if we're in selective fill mode
        elif hasattr(self, 'selective_fill_mode') and self.selective_fill_mode:
            # Find all contours that contain the clicked point
            matching_contours = []
            for i, contour in enumerate(self.contours_list):
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    area = cv2.contourArea(contour)
                    if area >= 30:  # Ignore small contours
                        matching_contours.append((i, contour, area))
            
            if matching_contours:
                # Sort by area (smallest first - innermost contours)
                matching_contours.sort(key=lambda x: x[2])
                
                # Fill the innermost contour (smallest area)
                i, contour, _ = matching_contours[0]
                self.fill_selected_contour(i, contour)
            return
    

    def delete_selected_contour(self):
        """Delete the currently selected contour."""
        if hasattr(self, 'selected_contour_index') and self.selected_contour_index is not None:
            # Save current state for undo
            self.save_state()
            
            # Create a mask with all contours except the selected one
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            for i, contour in enumerate(self.contours_list):
                if i != self.selected_contour_index:
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Apply the mask to the image
            if len(self.image.shape) == 3:
                self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
            else:
                self.image = cv2.bitwise_and(self.image, mask)
            
            # Update contours and display
            self.contours_list = self.detect_contours(spacing=8)
            self.update_display()
            
            # Calculate metrics if ground truth is available
            metrics = self.calculate_metrics_for_history()
            
            # Create operation entry
            operation_entry = {
                "name": "Delete Contour",
                "params": {
                    "Contour Index": self.selected_contour_index
                }
            }
            
            # Add metrics if available
            if metrics:
                operation_entry["metrics"] = metrics
            
            self.operations_history.append(operation_entry)
            self.update_operations_history()
            self.selected_contour_index = None


    def highlight_selected_contour(self, contour):
        """Highlight the selected contour in the image."""
        temp_image = self.image.copy()
        cv2.drawContours(temp_image, [contour], -1, (0, 0, 255), 2)  # Red color
    
        # Display the image with highlighted contour
        self.image_with_highlight = temp_image
        self.update_display(use_highlighted=True)

    def setup_fill_tab(self, parent):
        """Setup the redesigned fill tab with multiple sub-tabs."""
        # Use ttk.Notebook for tabbed interface
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Existing Fill Options tab
        existing_fill_frame = Frame(notebook)
        notebook.add(existing_fill_frame, text="Selective Fill")
        self.setup_selective_fill_tab(existing_fill_frame)

        # New All Fill tab
        all_fill_frame = Frame(notebook)
        notebook.add(all_fill_frame, text="All Fill")
        self.setup_all_fill_tab(all_fill_frame)

    def setup_selective_fill_tab(self, parent):
        """Setup the original selective fill tab (renamed from setup_fill_tab)."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Fill mode selection
        mode_frame = Frame(control_frame)
        mode_frame.pack(fill="x", pady=5)
        Label(mode_frame, text="Fill Mode:").pack(side="left")
        self.fill_mode_var = StringVar(value="white")
        fill_modes = [("Fill White Regions", "white"), ("Fill Black Regions", "black")]
        for text, mode in fill_modes:
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.fill_mode_var, value=mode)
            rb.pack(side="left")
        
        # Separator
        separator = Frame(control_frame, height=2, bd=1, relief="sunken")
        separator.pack(fill="x", padx=5, pady=10)
        
        # Filter parameters
        filter_frame = Frame(control_frame, bd=1, relief="groove")
        filter_frame.pack(fill="x", pady=5)
        Label(filter_frame, text="Filter Parameters", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Enable/disable checkboxes for each filter
        self.use_area_filter = IntVar(value=1)
        Checkbutton(filter_frame, text="Enable Area Filter", variable=self.use_area_filter).pack(anchor="w")
        
        # Area sliders
        area_frame = Frame(filter_frame)
        area_frame.pack(fill="x", pady=5)
        
        # Min Area
        min_area_frame = Frame(area_frame)
        min_area_frame.pack(fill="x")
        self.min_fill_area_label = Label(min_area_frame, text=f"Min Area: {self.min_area}")
        self.min_fill_area_label.pack(side="left")
        Button(min_area_frame, text="Set Range",
            command=lambda: self.open_range_dialog(
                self.min_fill_area_slider, self.min_fill_area_label,
                lambda val: f"Min Area: {int(val)}",
                lambda val: self.update_min_fill_area(val)
            )).pack(side="right")
        self.min_fill_area_slider = Scale(area_frame, from_=1, to=500, orient="horizontal",
                                        command=lambda val: self.update_min_fill_area(val))
        self.min_fill_area_slider.set(self.min_area)
        self.min_fill_area_slider.pack(fill="x")
        
        # Max Area
        max_area_frame = Frame(area_frame)
        max_area_frame.pack(fill="x")
        self.max_fill_area_label = Label(max_area_frame, text=f"Max Area: {self.max_area}")
        self.max_fill_area_label.pack(side="left")
        Button(max_area_frame, text="Set Range",
            command=lambda: self.open_range_dialog(
                self.max_fill_area_slider, self.max_fill_area_label,
                lambda val: f"Max Area: {int(val)}",
                lambda val: self.update_max_fill_area(val)
            )).pack(side="right")
        self.max_fill_area_slider = Scale(area_frame, from_=1, to=500, orient="horizontal",
                                        command=lambda val: self.update_max_fill_area(val))
        self.max_fill_area_slider.set(self.max_area)
        self.max_fill_area_slider.pack(fill="x")
        
        # Enable circularity filter checkbox
        self.use_circ_filter = IntVar(value=1)
        Checkbutton(filter_frame, text="Enable Circularity Filter", variable=self.use_circ_filter).pack(anchor="w")
        
        # Circularity sliders
        circ_frame = Frame(filter_frame)
        circ_frame.pack(fill="x", pady=5)
        
        # Min Circularity
        min_circ_frame = Frame(circ_frame)
        min_circ_frame.pack(fill="x")
        self.min_fill_circ_label = Label(min_circ_frame, text=f"Min Circularity: {self.min_circularity:.2f}")
        self.min_fill_circ_label.pack(side="left")
        Button(min_circ_frame, text="Set Range",
            command=lambda: self.open_range_dialog(
                self.min_fill_circ_slider, self.min_fill_circ_label,
                lambda val: f"Min Circularity: {float(val)/100:.2f}",
                lambda val: self.update_min_fill_circularity(val)
            )).pack(side="right")
        self.min_fill_circ_slider = Scale(circ_frame, from_=0, to=100, orient="horizontal",
                                        command=lambda val: self.update_min_fill_circularity(val))
        self.min_fill_circ_slider.set(int(self.min_circularity * 100))
        self.min_fill_circ_slider.pack(fill="x")
        
        # Max Circularity
        max_circ_frame = Frame(circ_frame)
        max_circ_frame.pack(fill="x")
        self.max_fill_circ_label = Label(max_circ_frame, text=f"Max Circularity: {self.max_circularity:.2f}")
        self.max_fill_circ_label.pack(side="left")
        Button(max_circ_frame, text="Set Range",
            command=lambda: self.open_range_dialog(
                self.max_fill_circ_slider, self.max_fill_circ_label,
                lambda val: f"Max Circularity: {float(val)/100:.2f}",
                lambda val: self.update_max_fill_circularity(val)
            )).pack(side="right")
        self.max_fill_circ_slider = Scale(circ_frame, from_=0, to=100, orient="horizontal",
                                        command=lambda val: self.update_max_fill_circularity(val))
        self.max_fill_circ_slider.set(int(self.max_circularity * 100))
        self.max_fill_circ_slider.pack(fill="x")
        
        # Enable eccentricity filter checkbox
        self.use_ecc_filter = IntVar(value=1)
        Checkbutton(filter_frame, text="Enable Eccentricity Filter", variable=self.use_ecc_filter).pack(anchor="w")
        
        # Eccentricity sliders
        ecc_frame = Frame(filter_frame)
        ecc_frame.pack(fill="x", pady=5)
        
        # Min Eccentricity
        min_ecc_frame = Frame(ecc_frame)
        min_ecc_frame.pack(fill="x")
        self.min_fill_ecc_label = Label(min_ecc_frame, text=f"Min Eccentricity: {self.min_eccentricity:.2f}")
        self.min_fill_ecc_label.pack(side="left")
        Button(min_ecc_frame, text="Set Range",
            command=lambda: self.open_range_dialog(
                self.min_fill_ecc_slider, self.min_fill_ecc_label,
                lambda val: f"Min Eccentricity: {float(val)/100:.2f}",
                lambda val: self.update_min_fill_eccentricity(val)
            )).pack(side="right")
        self.min_fill_ecc_slider = Scale(ecc_frame, from_=0, to=95, orient="horizontal",
                                        command=lambda val: self.update_min_fill_eccentricity(val))
        self.min_fill_ecc_slider.set(int(self.min_eccentricity * 100))
        self.min_fill_ecc_slider.pack(fill="x")
        
        # Max Eccentricity
        max_ecc_frame = Frame(ecc_frame)
        max_ecc_frame.pack(fill="x")
        self.max_fill_ecc_label = Label(max_ecc_frame, text=f"Max Eccentricity: {self.max_eccentricity:.2f}")
        self.max_fill_ecc_label.pack(side="left")
        Button(max_ecc_frame, text="Set Range",
            command=lambda: self.open_range_dialog(
                self.max_fill_ecc_slider, self.max_fill_ecc_label,
                lambda val: f"Max Eccentricity: {float(val)/100:.2f}",
                lambda val: self.update_max_fill_eccentricity(val)
            )).pack(side="right")
        self.max_fill_ecc_slider = Scale(ecc_frame, from_=0, to=95, orient="horizontal",
                                        command=lambda val: self.update_max_fill_eccentricity(val))
        self.max_fill_ecc_slider.set(int(self.max_eccentricity * 100))
        self.max_fill_ecc_slider.pack(fill="x")
        
        # Separator
        separator = Frame(control_frame, height=2, bd=1, relief="sunken")
        separator.pack(fill="x", padx=5, pady=10)
        
        # Preview and Fill buttons
        action_frame = Frame(control_frame)
        action_frame.pack(fill="x", pady=5)
        Button(action_frame, text="Preview", command=self.update_fill_preview).pack(side="left", padx=5)
        Button(action_frame, text="Fill", command=self.apply_filtered_fill).pack(side="left", padx=5)
        Button(action_frame, text="Clear Preview", command=self.clear_fill_preview).pack(side="left", padx=5)
        Button(action_frame, text="Cancel", command=self.cancel_fill_operation).pack(side="left", padx=5)
        
        # Selective fill controls
        Label(control_frame, text="Selective Fill", font=("Arial", 10, "bold")).pack(anchor="w")
        self.selective_fill_mode = False
        self.selective_fill_button = Button(
            control_frame, text="Enable Selective Fill", command=self.toggle_selective_fill_mode
        )
        self.selective_fill_button.pack(fill="x", pady=5)
        Label(control_frame, text="Click on a contour to fill it with white color.", wraplength=300).pack(pady=5)


    def update_min_fill_area(self, value):
        """Update minimum area parameter for fill filtering."""
        self.min_area = int(value)
        self.min_fill_area_label.config(text=f"Min Area: {self.min_area}")

    def update_max_fill_area(self, value):
        """Update maximum area parameter for fill filtering."""
        self.max_area = int(value)
        self.max_fill_area_label.config(text=f"Max Area: {self.max_area}")

    def update_min_fill_circularity(self, value):
        """Update minimum circularity parameter for fill filtering."""
        self.min_circularity = float(value) / 100.0
        self.min_fill_circ_label.config(text=f"Min Circularity: {self.min_circularity:.2f}")

    def update_max_fill_circularity(self, value):
        """Update maximum circularity parameter for fill filtering."""
        self.max_circularity = float(value) / 100.0
        self.max_fill_circ_label.config(text=f"Max Circularity: {self.max_circularity:.2f}")

    def update_min_fill_eccentricity(self, value):
        """Update minimum eccentricity parameter for fill filtering."""
        self.min_eccentricity = float(value) / 100.0
        self.min_fill_ecc_label.config(text=f"Min Eccentricity: {self.min_eccentricity:.2f}")

    def update_max_fill_eccentricity(self, value):
        """Update maximum eccentricity parameter for fill filtering."""
        self.max_eccentricity = float(value) / 100.0
        self.max_fill_ecc_label.config(text=f"Max Eccentricity: {self.max_eccentricity:.2f}")

    def update_fill_preview(self):
        """Generate a preview with selective filtering based on enabled checkboxes."""
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
        
        # Use adaptive thresholding for better detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Invert binary for black regions if needed
        if self.fill_mode_var.get() == "black":
            binary = cv2.bitwise_not(binary)
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create preview image
        self.preview_image = self.image.copy()
        if len(self.preview_image.shape) == 2:
            self.preview_image = cv2.cvtColor(self.preview_image, cv2.COLOR_GRAY2BGR)
        
        # Get filter parameters based on checkboxes
        use_area = self.use_area_filter.get() == 1
        use_circ = self.use_circ_filter.get() == 1
        use_ecc = self.use_ecc_filter.get() == 1
        
        min_area = self.min_fill_area_slider.get() if use_area else 0
        max_area = self.max_fill_area_slider.get() if use_area else float('inf')
        
        min_circ = float(self.min_fill_circ_slider.get()) / 100.0 if use_circ else 0
        max_circ = float(self.max_fill_circ_slider.get()) / 100.0 if use_circ else float('inf')
        
        min_ecc = float(self.min_fill_ecc_slider.get()) / 100.0 if use_ecc else 0
        max_ecc = float(self.max_fill_ecc_slider.get()) / 100.0 if use_ecc else float('inf')
        
        self.filtered_contours = []
        
        for i, contour in enumerate(contours):
            # Create mask for current contour
            mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Calculate properties using regionprops
            props = regionprops(mask.astype(np.uint8))
            if not props:
                continue
                
            props = props[0]
            area = props.area
            
            # Skip filters that aren't enabled
            passes_filters = True
            
            if use_area and not (min_area <= area <= max_area):
                passes_filters = False
                
            if use_circ and passes_filters:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    if not (min_circ <= circularity <= max_circ):
                        passes_filters = False
                else:
                    passes_filters = False
                    
            if use_ecc and passes_filters:
                try:
                    eccentricity = props.eccentricity
                    if not (min_ecc <= eccentricity <= max_ecc):
                        passes_filters = False
                except:
                    passes_filters = False
            
        # If contour passes all enabled filters, add to filtered list
        if passes_filters:
            self.filtered_contours.append((i, contour))
            cv2.drawContours(self.preview_image, [contour], -1, (0, 255, 255), thickness=cv2.FILLED)
    
        # Update display with preview
        self.update_display(use_highlighted=True)


    def clear_fill_preview(self):
        """Clear the preview and revert to the original image display."""
        if hasattr(self, 'preview_image'):
            del self.preview_image
        self.update_display()


    def apply_filtered_fill(self):
        """Apply fill to filtered contours with proper error handling."""
        if not hasattr(self, 'filtered_contours') or not self.filtered_contours:
            messagebox.showinfo("No Contours", "No contours match the current filter criteria. Try adjusting the filters or running Preview first.")
            return
        
        self.save_state()
        
        # Create mask for filling
        if len(self.image.shape) == 3:
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros_like(self.image, dtype=np.uint8)
        
        # Fill the filtered contours
        for i, contour in self.filtered_contours:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Apply mask to create final image
        if len(self.image.shape) == 3:
            # For color images
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.image = np.where(mask_3ch > 0, 255, self.image)
        else:
            # For grayscale images
            self.image = np.where(mask > 0, 255, self.image)
        
        # Clear preview and update display
        if hasattr(self, 'preview_image'):
            del self.preview_image
        
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        
        # Calculate metrics if ground truth is available
        metrics = self.calculate_metrics_for_history()
        
        # Create parameter dictionary
        params = {"Fill Mode": self.fill_mode_var.get()}
        if self.use_area_filter.get():
            params["Min Area"] = self.min_area
            params["Max Area"] = self.max_area
        if self.use_circ_filter.get():
            params["Min Circularity"] = f"{self.min_circularity:.2f}"
            params["Max Circularity"] = f"{self.max_circularity:.2f}"
        if self.use_ecc_filter.get():
            params["Min Eccentricity"] = f"{self.min_eccentricity:.2f}"
            params["Max Eccentricity"] = f"{self.max_eccentricity:.2f}"
        
        # Create operation entry
        operation_entry = {
            "name": "Filtered Selective Fill",
            "params": params
        }
        
        # Add metrics if available
        if metrics:
            operation_entry["metrics"] = metrics
        
        self.operations_history.append(operation_entry)
        self.update_operations_history()


    def setup_main_tab(self, parent):
        """Setup the main tab with controls only."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)

        """Setup the main tab with controls only."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Add undo/redo buttons
        Button(control_frame, text="Undo", command=self.undo).pack(side="left")
        Button(control_frame, text="Redo", command=self.redo).pack(side="left")

        # Morphological operation buttons
        Button(control_frame, text="Erode", command=lambda: self.apply_morphology("erode")).pack(side="left")
        Button(control_frame, text="Dilate", command=lambda: self.apply_morphology("dilate")).pack(side="left")
        Button(control_frame, text="Reset", command=self.reset_image).pack(side="left")
        Button(control_frame, text="Save Image", command=self.save_image).pack(side="left")

        # Zoom buttons
        Button(control_frame, text="Zoom In (+)", command=self.zoom_in).pack(side="left")
        Button(control_frame, text="Zoom Out (-)", command=self.zoom_out).pack(side="left")

        # Add load new image button
        Button(control_frame, text="Load New Image", command=self.load_new_image).pack(side="left")

        # Kernel size slider
        self.kernel_label = Label(control_frame, text=f"Kernel Size: {self.kernel_size}")
        self.kernel_label.pack(side="left")
        kernel_slider = Scale(control_frame, from_=1, to=10, orient="horizontal", command=self.update_kernel_size)
        kernel_slider.set(self.kernel_size)
        kernel_slider.pack(side="left")

        # Separator
        separator = Frame(parent, height=2, bd=1, relief="sunken")
        separator.pack(fill="x", padx=5, pady=5)

        # Refinement parameters
        refinement_frame = Frame(parent)
        refinement_frame.pack(fill="x", padx=10, pady=5)

        Button(refinement_frame, text="Refine Contours", command=self.refine_contours).pack(side="top", pady=5)

        self.force_constant_label = Label(refinement_frame, text=f"Force Constant: {self.force_constant:.1f}")
        self.force_constant_label.pack()
        force_constant_slider = Scale(refinement_frame, from_=1, to=100, orient="horizontal", command=self.update_force_constant)
        force_constant_slider.set(int(self.force_constant * 10))
        force_constant_slider.pack(fill="x")

        self.decay_exponent_label = Label(refinement_frame, text=f"Decay Exponent: {self.decay_exponent:.1f}")
        self.decay_exponent_label.pack()
        decay_exponent_slider = Scale(refinement_frame, from_=1, to=50, orient="horizontal", command=self.update_decay_exponent)
        decay_exponent_slider.set(int(self.decay_exponent * 10))
        decay_exponent_slider.pack(fill="x")

        self.gradient_threshold_label = Label(refinement_frame, text=f"Gradient Threshold: {self.gradient_threshold}")
        self.gradient_threshold_label.pack()
        gradient_threshold_slider = Scale(refinement_frame, from_=1, to=100, orient="horizontal", command=self.update_gradient_threshold)
        gradient_threshold_slider.set(self.gradient_threshold)
        gradient_threshold_slider.pack(fill="x")

        self.epsilon_label = Label(refinement_frame, text=f"Epsilon: {self.epsilon:.4f}")
        self.epsilon_label.pack()
        epsilon_slider = Scale(refinement_frame, from_=1, to=100, orient="horizontal", command=self.update_epsilon)
        epsilon_slider.set(int(self.epsilon * 10000))
        epsilon_slider.pack(fill="x")

    def toggle_contour_info_mode(self):
        """Toggle between contour info mode and point selection mode."""
        self.contour_info_mode = not self.contour_info_mode
        
        if self.contour_info_mode:
            self.toggle_info_button.config(text="Disable Contour Selection")
            # Since we no longer have tabs, just store the current state
            self.status_bar.config(text="Contour selection mode enabled")
        else:
            self.toggle_info_button.config(text="Enable Contour Selection")
            self.status_bar.config(text="Contour selection mode disabled")


    def update_tracking_tab(self):
        """Update the contour tracking tab with current tracked contours."""
        # Clear existing items
        for item in self.contour_tree.get_children():
            self.contour_tree.delete(item)
        
        # Add tracked contours to the treeview
        for props in self.tracked_contours:
            self.contour_tree.insert(
                "", "end",
                values=(
                    props["ID"],
                    props["Area"],
                    props["Perimeter"],
                    props["Circularity"],
                    props["Aspect Ratio"],
                    props["Centroid"],
                    props["Bounding Box"],
                    props["Eccentricity"]
                )
            )

    def clear_tracked_contours(self):
        """Clear the list of tracked contours."""
        self.tracked_contours = []
        self.update_tracking_tab()


    def on_mouse_drag(self, event):
        """Disabled - previously handled mouse dragging for moving contour points."""
        pass


    def on_right_click(self, event):
        """Disabled - previously handled right mouse click to fill contours."""
        pass


    def refine_contours(self):
        self.save_state()
        """Refine contours using Coulomb-like forces, distance decay, and gradient stopping."""
        # Convert original image to grayscale for gradient calculation
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient in x and y directions using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient = cv2.magnitude(grad_x, grad_y)
        
        # Create edge mask where gradient is above threshold
        edge_mask = (gradient > self.gradient_threshold).astype(np.uint8) * 255
        
        # Invert the edge mask for distance transform
        inverted_edges = cv2.bitwise_not(edge_mask)
        
        # Calculate distance transform (distance to nearest edge)
        distance = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
        
        # Initialize list for refined contours
        new_contours = []
        
        # Process each contour
        for contour in self.contours_list:
            # Convert contour to float32 for precise calculations
            contour = contour.astype(np.float32)
            
            # Calculate centroid of the contour
            centroid = np.mean(contour, axis=0)
            
            # Calculate distances from each point to centroid
            dists = np.linalg.norm(contour - centroid, axis=1)
            
            # Get maximum distance (avoid division by zero)
            d_max = np.max(dists) if np.max(dists) != 0 else 1
            
            # Initialize list for refined points
            refined_contour = []
            
            # Process each point in the contour
            for pt in contour:
                # Convert point coordinates to integers for indexing
                x, y = int(pt[0]), int(pt[1])
                
                # Check bounds to avoid index errors
                if y >= gradient.shape[0] or x >= gradient.shape[1] or y < 0 or x < 0:
                    refined_contour.append(pt)
                    continue
                
                # If point is on a strong edge, keep it as is
                if gradient[y, x] > self.gradient_threshold:
                    refined_contour.append(pt)
                    continue
                    
                # Get distance to nearest edge
                r = distance[y, x]
                
                # Calculate step size using inverse power law with decay
                step = self.force_constant / ((r + self.epsilon) ** self.decay_exponent)
                
                # Get gradient components at this point
                gx = grad_x[y, x]
                gy = grad_y[y, x]
                
                # Calculate gradient magnitude
                mag = np.sqrt(gx * gx + gy * gy)
                
                # If gradient is zero, keep point as is
                if mag == 0:
                    refined_contour.append(pt)
                    continue
                    
                # Normalize gradient to get direction
                nx = gx / mag
                ny = gy / mag
                
                # Calculate distance from point to centroid
                d = np.linalg.norm(pt - centroid)
                
                # Apply dielectric-like effect based on distance from centroid
                dielectric = 1.0 + (d / d_max)
                
                # Adjust step size based on dielectric effect
                step_adjusted = step / dielectric
                
                # Calculate new point position by moving against gradient direction
                new_x = pt[0] - nx * step_adjusted
                new_y = pt[1] - ny * step_adjusted
                
                # Add the new point to refined contour
                refined_contour.append([new_x, new_y])
                
            # Convert refined points to numpy array
            refined_contour = np.array(refined_contour, dtype=np.float32)
            
            # Create a copy for smoothing
            smoothed_contour = refined_contour.copy()
            
            # Get number of points in contour
            num_points = len(refined_contour)
            
            # Apply smoothing by averaging each point with its neighbors
            for i in range(num_points):
                # Get previous, current, and next points (with wraparound)
                prev_pt = refined_contour[i - 1]
                curr_pt = refined_contour[i]
                next_pt = refined_contour[(i + 1) % num_points]
                
                # Average the three points
                smoothed_contour[i] = (prev_pt + curr_pt + next_pt) / 3.0
                
            # Add the smoothed contour to new contours list
            new_contours.append(smoothed_contour.astype(np.int32))
            
            # Check if contour is convex, if not adjust it
            if not cv2.isContourConvex(smoothed_contour):
                try:
                    # Compute the convex hull
                    hull = cv2.convexHull(smoothed_contour, returnPoints=True).squeeze()
                    
                    # Ensure hull points are in correct shape
                    hull_points = hull if len(hull.shape) == 2 else hull.reshape(-1, 2)
                    
                    # Create a copy of the smoothed contour for adjustment
                    adjusted_contour = smoothed_contour.copy()
                    
                    # Check each point and adjust if needed
                    for i, pt in enumerate(smoothed_contour):
                        # If point is not inside the hull, adjust it
                        if cv2.pointPolygonTest(hull_points, (pt[0], pt[1]), False) < 0:
                            # Find closest point on hull
                            distances = np.linalg.norm(hull_points - pt, axis=1)
                            closest_idx = np.argmin(distances)
                            
                            # Replace with closest hull point
                            adjusted_contour[i] = hull_points[closest_idx]
                            
                    # Use the adjusted contour
                    smoothed_contour = adjusted_contour
                except Exception as e:
                    print(f"Error adjusting contour: {e}")
        
        # Update the contours list with refined contours
        self.contours_list = new_contours
        
        # Create a mask for the final contours
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        # Fill the contours in the mask
        for contour in self.contours_list:
            cv2.fillPoly(mask, [contour], 255)
            
        # Convert mask to BGR image
        self.image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Update the display to show refined contours
        self.update_display()

    def update_force_constant(self, value):
        """Update force constant based on slider value."""
        self.force_constant = float(value) / 10.0
        self.force_constant_label.config(text=f"Force Constant: {self.force_constant:.1f}")

    def update_decay_exponent(self, value):
        """Update decay exponent based on slider value."""
        self.decay_exponent = float(value) / 10.0
        self.decay_exponent_label.config(text=f"Decay Exponent: {self.decay_exponent:.1f}")

    def update_gradient_threshold(self, value):
        """Update gradient threshold based on slider value."""
        self.gradient_threshold = int(value)
        self.gradient_threshold_label.config(text=f"Gradient Threshold: {self.gradient_threshold}")

    def update_epsilon(self, value):
        """Update epsilon based on slider value."""
        self.epsilon = float(value) / 10000.0
        self.epsilon_label.config(text=f"Epsilon: {self.epsilon:.4f}")

    def setup_aspect_tab(self, parent):
        """Setup the aspect ratio tab."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Min Aspect Ratio slider with range button
        min_aspect_frame = Frame(control_frame)
        min_aspect_frame.pack(fill="x")
        self.min_aspect_ratio_label = Label(min_aspect_frame, text=f"Min Aspect Ratio: {self.min_aspect_ratio:.1f}")
        self.min_aspect_ratio_label.pack(side="left")
        range_button = Button(min_aspect_frame, text="Set Range",
                     command=lambda: self.open_range_dialog(
                         min_aspect_slider,
                         self.min_aspect_ratio_label,
                         lambda val: f"Min Aspect Ratio: {float(val)/10:.1f}",
                         lambda val: self.update_min_aspect_ratio(val)
                     ))
        range_button.pack(side="right")
        
        min_aspect_slider = Scale(control_frame, from_=0, to=50, orient="horizontal",
                                command=lambda val: self.update_min_aspect_ratio(val))
        min_aspect_slider.set(int(self.min_aspect_ratio * 10))
        min_aspect_slider.pack(fill="x")
        
        # Max Aspect Ratio slider with range button
        max_aspect_frame = Frame(control_frame)
        max_aspect_frame.pack(fill="x")
        self.max_aspect_ratio_label = Label(max_aspect_frame, text=f"Max Aspect Ratio: {self.max_aspect_ratio:.1f}")
        self.max_aspect_ratio_label.pack(side="left")
        Button(max_aspect_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                max_aspect_slider, 
                self.max_aspect_ratio_label,
                lambda val: f"Max Aspect Ratio: {float(val)/10:.1f}",
                lambda val: self.update_max_aspect_ratio(val)
            )).pack(side="right")
        
        max_aspect_slider = Scale(control_frame, from_=0, to=50, orient="horizontal",
                                command=lambda val: self.update_max_aspect_ratio(val))
        max_aspect_slider.set(int(self.max_aspect_ratio * 10))
        max_aspect_slider.pack(fill="x")
        
        # Mode selection with radio buttons
        mode_frame = Frame(control_frame)
        mode_frame.pack(fill="x", pady=5)
        Label(mode_frame, text="Mode:").pack(side="left")
        self.aspect_ratio_mode_var = StringVar(value=self.aspect_ratio_mode)
        for mode in ["fill", "mark"]:
            rb = ttk.Radiobutton(mode_frame, text=mode, variable=self.aspect_ratio_mode_var, value=mode,
                            command=lambda: self.set_aspect_ratio_mode(self.aspect_ratio_mode_var.get()))
            rb.pack(side="left")
        
        # Apply button
        Button(control_frame, text="Apply Aspect Ratio Elimination",
            command=self.AspectRatio_Elimination).pack(fill="x", pady=10)

    def setup_eccentricity_tab(self, parent):
        """Setup the eccentricity tab for removing contours within specified ranges."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Min Area slider with range button
        min_area_frame = Frame(control_frame)
        min_area_frame.pack(fill="x")
        self.min_area_label = Label(min_area_frame, text=f"Min Area: {self.min_area}")
        self.min_area_label.pack(side="left")
        range_button = Button(min_area_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                min_area_slider, 
                self.min_area_label,
                lambda val: f"Min Area: {int(val)}",
                lambda val: self.update_min_area(val)
            ))
        range_button.pack(side="right")
        
        min_area_slider = Scale(control_frame, from_=1, to=700, orient="horizontal",
                            command=lambda val: self.update_min_area(val))
        min_area_slider.set(self.min_area)
        min_area_slider.pack(fill="x")
        
        # Max Area slider with range button
        max_area_frame = Frame(control_frame)
        max_area_frame.pack(fill="x")
        self.max_area_label = Label(max_area_frame, text=f"Max Area: {self.max_area}")
        self.max_area_label.pack(side="left")
        Button(max_area_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                max_area_slider, 
                self.max_area_label,
                lambda val: f"Max Area: {int(val)}",
                lambda val: self.update_max_area(val)
            )).pack(side="right")
        
        max_area_slider = Scale(control_frame, from_=1, to=1500, orient="horizontal",
                            command=lambda val: self.update_max_area(val))
        max_area_slider.set(self.max_area)
        max_area_slider.pack(fill="x")
        
        # Min Eccentricity slider with range button
        min_ecc_frame = Frame(control_frame)
        min_ecc_frame.pack(fill="x")
        self.min_eccentricity_label = Label(min_ecc_frame, text=f"Min Eccentricity: {self.min_eccentricity:.2f}")
        self.min_eccentricity_label.pack(side="left")
        Button(min_ecc_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                min_eccentricity_slider, 
                self.min_eccentricity_label,
                lambda val: f"Min Eccentricity: {float(val)/100:.2f}",
                lambda val: self.update_min_eccentricity(val)
            )).pack(side="right")
        
        min_eccentricity_slider = Scale(control_frame, from_=0, to=95, orient="horizontal",
                                    command=lambda val: self.update_min_eccentricity(val))
        min_eccentricity_slider.set(int(self.min_eccentricity * 100))
        min_eccentricity_slider.pack(fill="x")
        
        # Max Eccentricity slider with range button
        max_ecc_frame = Frame(control_frame)
        max_ecc_frame.pack(fill="x")
        self.max_eccentricity_label = Label(max_ecc_frame, text=f"Max Eccentricity: {self.max_eccentricity:.2f}")
        self.max_eccentricity_label.pack(side="left")
        Button(max_ecc_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                max_eccentricity_slider, 
                self.max_eccentricity_label,
                lambda val: f"Max Eccentricity: {float(val)/100:.2f}",
                lambda val: self.update_max_eccentricity(val)
            )).pack(side="right")
        
        max_eccentricity_slider = Scale(control_frame, from_=0, to=95, orient="horizontal",
                                    command=lambda val: self.update_max_eccentricity(val))
        max_eccentricity_slider.set(int(self.max_eccentricity * 100))
        max_eccentricity_slider.pack(fill="x")
        
        # Rest of the method remains the same...

        
        # Mode selection with radio buttons
        mode_frame = Frame(control_frame)
        mode_frame.pack(fill="x", pady=5)
        Label(mode_frame, text="Mode:").pack(side="left")
        self.eccentricity_mode_var = StringVar(value=self.eccentricity_mode)
        for mode in ["fill", "mark"]:
            rb = ttk.Radiobutton(mode_frame, text=mode, variable=self.eccentricity_mode_var, value=mode,
                                 command=lambda: self.set_eccentricity_mode(self.eccentricity_mode_var.get()))
            rb.pack(side="left")
        
        # Thresholding option
        threshold_frame = Frame(control_frame)
        threshold_frame.pack(fill="x", pady=5)
        self.use_adaptive_ecc = IntVar(value=0)
        Checkbutton(threshold_frame, text="Use Adaptive Thresholding", variable=self.use_adaptive_ecc).pack(anchor="w")
        
        # Apply button
        Button(control_frame, text="Remove Contours by Eccentricity",
              command=lambda: self.eccentricity_elimination(use_adaptive=self.use_adaptive_ecc.get())).pack(fill="x", pady=10)


    def setup_iso_tab(self, parent):
        """Setup the circularity tab."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Min Area slider with range button
        min_area_frame = Frame(control_frame)
        min_area_frame.pack(fill="x")
        self.min_area_label = Label(min_area_frame, text=f"Min Area: {self.min_area}")
        self.min_area_label.pack(side="left")
        range_button = Button(min_area_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                min_area_slider, 
                self.min_area_label,
                lambda val: f"Min Area: {int(val)}",
                lambda val: self.update_min_area(val)
            ))
        range_button.pack(side="right")
        
        min_area_slider = Scale(control_frame, from_=120, to=500, orient="horizontal",
                            command=lambda val: self.update_min_area(val))
        min_area_slider.set(self.min_area)
        min_area_slider.pack(fill="x")
        
        # Max Area slider with range button
        max_area_frame = Frame(control_frame)
        max_area_frame.pack(fill="x")
        self.max_area_label = Label(max_area_frame, text=f"Max Area: {self.max_area}")
        self.max_area_label.pack(side="left")
        Button(max_area_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                max_area_slider, 
                self.max_area_label,
                lambda val: f"Max Area: {int(val)}",
                lambda val: self.update_max_area(val)
            )).pack(side="right")
        
        max_area_slider = Scale(control_frame, from_=200, to=1800, orient="horizontal",
                            command=lambda val: self.update_max_area(val))
        max_area_slider.set(self.max_area)
        max_area_slider.pack(fill="x")
        
        # Min Circularity slider with range button
        min_circ_frame = Frame(control_frame)
        min_circ_frame.pack(fill="x")
        self.min_circularity_label = Label(min_circ_frame, text=f"Min Circularity: {self.min_circularity:.2f}")
        self.min_circularity_label.pack(side="left")
        Button(min_circ_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                min_circularity_slider, 
                self.min_circularity_label,
                lambda val: f"Min Circularity: {float(val)/100:.2f}",
                lambda val: self.update_min_circularity(val)
            )).pack(side="right")
        
        min_circularity_slider = Scale(control_frame, from_=0, to=100, orient="horizontal",
                                    command=lambda val: self.update_min_circularity(val))
        min_circularity_slider.set(int(self.min_circularity * 100))
        min_circularity_slider.pack(fill="x")
        
        # Max Circularity slider with range button
        max_circ_frame = Frame(control_frame)
        max_circ_frame.pack(fill="x")
        self.max_circularity_label = Label(max_circ_frame, text=f"Max Circularity: {self.max_circularity:.2f}")
        self.max_circularity_label.pack(side="left")
        Button(max_circ_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                max_circularity_slider, 
                self.max_circularity_label,
                lambda val: f"Max Circularity: {float(val)/100:.2f}",
                lambda val: self.update_max_circularity(val)
            )).pack(side="right")
        
        max_circularity_slider = Scale(control_frame, from_=0, to=100, orient="horizontal",
                                    command=lambda val: self.update_max_circularity(val))
        max_circularity_slider.set(int(self.max_circularity * 100))
        max_circularity_slider.pack(fill="x")
        
        # Mode selection with radio buttons
        mode_frame = Frame(control_frame)
        mode_frame.pack(fill="x", pady=5)
        Label(mode_frame, text="Mode:").pack(side="left")
        self.isoperimetric_mode_var = StringVar(value=self.isoperimetric_mode)
        for mode in ["detect", "remove"]:
            rb = ttk.Radiobutton(mode_frame, text=mode, variable=self.isoperimetric_mode_var, value=mode,
                            command=lambda: self.set_isoperimetric_mode(self.isoperimetric_mode_var.get()))
            rb.pack(side="left")
        
        # Apply button
        Button(control_frame, text="Apply Circularity Algorithm",
            command=self.isoperimetric_algo).pack(fill="x", pady=10)


    def setup_blur_tab(self, parent):
        """Setup the blur tab."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)

        # Blur Kernel Size slider
        self.blur_kernel_label = Label(control_frame, text=f"Blur Kernel Size: {self.blur_kernel_size}")
        self.blur_kernel_label.pack(anchor="w")
        blur_kernel_slider = Scale(control_frame, from_=1, to=21, orient="horizontal",
                                   command=lambda val: self.update_blur_kernel_size(val))
        blur_kernel_slider.set(self.blur_kernel_size)
        blur_kernel_slider.pack(fill="x")

        # Blur Type selection with radio buttons
        type_frame = Frame(control_frame)
        type_frame.pack(fill="x", pady=5)
        Label(type_frame, text="Blur Type:").pack(side="left")
        self.blur_type_var = StringVar(value=self.blur_type)
        blur_types = ["gaussian", "median", "average", "bilateral"]
        for btype in blur_types:
            rb = ttk.Radiobutton(type_frame, text=btype, variable=self.blur_type_var, value=btype,
                                 command=lambda: self.set_blur_type(self.blur_type_var.get()))
            rb.pack(side="left")

        # Apply button
        Button(control_frame, text="Apply Blur",
               command=self.apply_blur).pack(fill="x", pady=10)

    def setup_area_tab(self, parent):
        """Setup the area threshold tab."""
        # Frame for controls
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Threshold Area slider with range button
        slider_frame = Frame(control_frame)
        slider_frame.pack(fill="x")
        
        self.threshold_area_label = Label(slider_frame, text=f"Threshold Area: {self.threshold_area}")
        self.threshold_area_label.pack(side="left")
    
        
        threshold_area_slider = Scale(control_frame, from_=1, to=1500, orient="horizontal",
                                    command=self.update_threshold_area)
        threshold_area_slider.set(self.threshold_area)
        threshold_area_slider.pack(fill="x")
        
        # Mode selection
        mode_frame = Frame(control_frame)
        mode_frame.pack(fill="x", pady=5)
        Label(mode_frame, text="Mode:").pack(side="left")
        self.area_mode_var = IntVar()
        self.area_mode_var.set(1)  # Default to "fill"
        Checkbutton(mode_frame, text="Fill", variable=self.area_mode_var, onvalue=1, offvalue=0,
                command=lambda: self.set_area_mode("fill" if self.area_mode_var.get() == 1 else "mark")).pack(side="left")
        
        # Apply button
        Button(control_frame, text="Apply Area Threshold",
            command=lambda: self.thresholdArea_Elimination()).pack(fill="x", pady=10)


    def setup_combined_tab(self, parent):
        """Setup the combined filtering tab."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Create frames for each filter type
        area_frame = Frame(control_frame, bd=1, relief="groove")
        area_frame.pack(fill="x", pady=5)
        Label(area_frame, text="Area Filtering", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Area range
        area_range_frame = Frame(area_frame)
        area_range_frame.pack(fill="x")
        
        # Enable area filtering checkbox
        self.use_area_filter = IntVar(value=0)
        Checkbutton(area_range_frame, text="Enable", variable=self.use_area_filter).pack(side="left")
        
        # Min Area with range button
        min_area_frame = Frame(area_frame)
        min_area_frame.pack(fill="x")
        Label(min_area_frame, text="Min:").pack(side="left")
        min_area_value_frame = Frame(min_area_frame)
        min_area_value_frame.pack(side="left", fill="x", expand=True)
        
        self.combined_min_area_label = Label(min_area_value_frame, text=f"Min Area: {self.min_area}")
        self.combined_min_area_label.pack(side="left")
        range_button = Button(min_area_value_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                self.combined_min_area, 
                self.combined_min_area_label,
                lambda val: f"Min Area: {int(val)}",
                None
            ))
        range_button.pack(side="right")
        
        self.combined_min_area = Scale(area_frame, from_=1, to=500, orient="horizontal")
        self.combined_min_area.set(self.min_area)
        self.combined_min_area.pack(fill="x")
        
        # Max Area with range button
        max_area_frame = Frame(area_frame)
        max_area_frame.pack(fill="x")
        Label(max_area_frame, text="Max:").pack(side="left")
        max_area_value_frame = Frame(max_area_frame)
        max_area_value_frame.pack(side="left", fill="x", expand=True)
        
        self.combined_max_area_label = Label(max_area_value_frame, text=f"Max Area: {self.max_area}")
        self.combined_max_area_label.pack(side="left")
        Button(max_area_value_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                self.combined_max_area, 
                self.combined_max_area_label,
                lambda val: f"Max Area: {int(val)}",
                None
            )).pack(side="right")
        
        self.combined_max_area = Scale(area_frame, from_=1, to=500, orient="horizontal")
        self.combined_max_area.set(self.max_area)
        self.combined_max_area.pack(fill="x")
        
        # Aspect Ratio frame
        aspect_frame = Frame(control_frame, bd=1, relief="groove")
        aspect_frame.pack(fill="x", pady=5)
        Label(aspect_frame, text="Aspect Ratio Filtering", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Aspect ratio range
        aspect_range_frame = Frame(aspect_frame)
        aspect_range_frame.pack(fill="x")
        
        # Enable aspect ratio filtering checkbox
        self.use_aspect_filter = IntVar(value=0)
        Checkbutton(aspect_range_frame, text="Enable", variable=self.use_aspect_filter).pack(side="left")
        
        # Min Aspect Ratio with range button
        min_aspect_frame = Frame(aspect_frame)
        min_aspect_frame.pack(fill="x")
        Label(min_aspect_frame, text="Min:").pack(side="left")
        min_aspect_value_frame = Frame(min_aspect_frame)
        min_aspect_value_frame.pack(side="left", fill="x", expand=True)
        
        self.combined_min_aspect_label = Label(min_aspect_value_frame, text=f"Min Aspect: {self.min_aspect_ratio:.1f}")
        self.combined_min_aspect_label.pack(side="left")
        Button(min_aspect_value_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                self.combined_min_aspect, 
                self.combined_min_aspect_label,
                lambda val: f"Min Aspect: {float(val)/10:.1f}",
                None
            )).pack(side="right")
        
        self.combined_min_aspect = Scale(aspect_frame, from_=0, to=50, orient="horizontal")
        self.combined_min_aspect.set(int(self.min_aspect_ratio * 10))
        self.combined_min_aspect.pack(fill="x")
        
        # Max Aspect Ratio with range button
        max_aspect_frame = Frame(aspect_frame)
        max_aspect_frame.pack(fill="x")
        Label(max_aspect_frame, text="Max:").pack(side="left")
        max_aspect_value_frame = Frame(max_aspect_frame)
        max_aspect_value_frame.pack(side="left", fill="x", expand=True)
        
        self.combined_max_aspect_label = Label(max_aspect_value_frame, text=f"Max Aspect: {self.max_aspect_ratio:.1f}")
        self.combined_max_aspect_label.pack(side="left")
        Button(max_aspect_value_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                self.combined_max_aspect, 
                self.combined_max_aspect_label,
                lambda val: f"Max Aspect: {float(val)/10:.1f}",
                None
            )).pack(side="right")
        
        self.combined_max_aspect = Scale(aspect_frame, from_=0, to=50, orient="horizontal")
        self.combined_max_aspect.set(int(self.max_aspect_ratio * 10))
        self.combined_max_aspect.pack(fill="x")
        
        # Circularity frame
        circ_frame = Frame(control_frame, bd=1, relief="groove")
        circ_frame.pack(fill="x", pady=5)
        Label(circ_frame, text="Circularity Filtering", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Circularity range
        circ_range_frame = Frame(circ_frame)
        circ_range_frame.pack(fill="x")
        
        # Enable circularity filtering checkbox
        self.use_circ_filter = IntVar(value=0)
        Checkbutton(circ_range_frame, text="Enable", variable=self.use_circ_filter).pack(side="left")
        
        # Min Circularity with range button
        min_circ_frame = Frame(circ_frame)
        min_circ_frame.pack(fill="x")
        Label(min_circ_frame, text="Min:").pack(side="left")
        min_circ_value_frame = Frame(min_circ_frame)
        min_circ_value_frame.pack(side="left", fill="x", expand=True)
        
        self.combined_min_circ_label = Label(min_circ_value_frame, text=f"Min Circularity: {self.min_circularity:.2f}")
        self.combined_min_circ_label.pack(side="left")
        Button(min_circ_value_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                self.combined_min_circ, 
                self.combined_min_circ_label,
                lambda val: f"Min Circularity: {float(val)/100:.2f}",
                None
            )).pack(side="right")
        
        self.combined_min_circ = Scale(circ_frame, from_=0, to=100, orient="horizontal")
        self.combined_min_circ.set(int(self.min_circularity * 100))
        self.combined_min_circ.pack(fill="x")
        
        # Max Circularity with range button
        max_circ_frame = Frame(circ_frame)
        max_circ_frame.pack(fill="x")
        Label(max_circ_frame, text="Max:").pack(side="left")
        max_circ_value_frame = Frame(max_circ_frame)
        max_circ_value_frame.pack(side="left", fill="x", expand=True)
        
        self.combined_max_circ_label = Label(max_circ_value_frame, text=f"Max Circularity: {self.max_circularity:.2f}")
        self.combined_max_circ_label.pack(side="left")
        Button(max_circ_value_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                self.combined_max_circ, 
                self.combined_max_circ_label,
                lambda val: f"Max Circularity: {float(val)/100:.2f}",
                None
            )).pack(side="right")
        
        self.combined_max_circ = Scale(circ_frame, from_=0, to=100, orient="horizontal")
        self.combined_max_circ.set(int(self.max_circularity * 100))
        self.combined_max_circ.pack(fill="x")
        
        # Eccentricity frame
        ecc_frame = Frame(control_frame, bd=1, relief="groove")
        ecc_frame.pack(fill="x", pady=5)
        Label(ecc_frame, text="Eccentricity Filtering", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Eccentricity range
        ecc_range_frame = Frame(ecc_frame)
        ecc_range_frame.pack(fill="x")
        
        # Enable eccentricity filtering checkbox
        self.use_ecc_filter = IntVar(value=0)
        Checkbutton(ecc_range_frame, text="Enable", variable=self.use_ecc_filter).pack(side="left")
        
        # Min Eccentricity with range button
        min_ecc_frame = Frame(ecc_frame)
        min_ecc_frame.pack(fill="x")
        Label(min_ecc_frame, text="Min:").pack(side="left")
        min_ecc_value_frame = Frame(min_ecc_frame)
        min_ecc_value_frame.pack(side="left", fill="x", expand=True)
        
        self.combined_min_ecc_label = Label(min_ecc_value_frame, text=f"Min Eccentricity: {self.min_eccentricity:.2f}")
        self.combined_min_ecc_label.pack(side="left")
        Button(min_ecc_value_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                self.combined_min_ecc, 
                self.combined_min_ecc_label,
                lambda val: f"Min Eccentricity: {float(val)/100:.2f}",
                None
            )).pack(side="right")
        
        self.combined_min_ecc = Scale(ecc_frame, from_=0, to=100, orient="horizontal")
        self.combined_min_ecc.set(int(self.min_eccentricity * 100))
        self.combined_min_ecc.pack(fill="x")
        
        # Max Eccentricity with range button
        max_ecc_frame = Frame(ecc_frame)
        max_ecc_frame.pack(fill="x")
        Label(max_ecc_frame, text="Max:").pack(side="left")
        max_ecc_value_frame = Frame(max_ecc_frame)
        max_ecc_value_frame.pack(side="left", fill="x", expand=True)
        
        self.combined_max_ecc_label = Label(max_ecc_value_frame, text=f"Max Eccentricity: {self.max_eccentricity:.2f}")
        self.combined_max_ecc_label.pack(side="left")
        Button(max_ecc_value_frame, text="Set Range", 
            command=lambda: self.open_range_dialog(
                self.combined_max_ecc, 
                self.combined_max_ecc_label,
                lambda val: f"Max Eccentricity: {float(val)/100:.2f}",
                None
            )).pack(side="right")
        
        self.combined_max_ecc = Scale(ecc_frame, from_=0, to=100, orient="horizontal")
        self.combined_max_ecc.set(int(self.max_eccentricity * 100))
        self.combined_max_ecc.pack(fill="x")
        
        # Action frame
        action_frame = Frame(control_frame)
        action_frame.pack(fill="x", pady=10)
        
        # Apply combined filtering button
        Button(action_frame, text="Apply Combined Filtering", command=self.apply_combined_filtering).pack(fill="x")

    def toggle_selective_fill_mode(self):
        """Toggle between selective fill mode and normal mode."""
        self.selective_fill_mode = not self.selective_fill_mode
        
        if self.selective_fill_mode:
            self.selective_fill_button.config(text="Disable Selective Fill")
            self.status_bar.config(text="Selective fill mode enabled")
        else:
            self.selective_fill_button.config(text="Enable Selective Fill")
            self.status_bar.config(text="Selective fill mode disabled")

    def fill_selected_contour(self, contour_index, contour):
        """Fill the selected contour with white."""
        self.save_state()
        
        # Create a mask for the selected contour
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Apply the mask to fill the contour with white
        if len(self.image.shape) == 3:
            # For color images
            white_fill = np.ones_like(self.image) * 255
            self.image = np.where(mask[:, :, np.newaxis] > 0, white_fill, self.image)
        else:
            # For grayscale images
            self.image = np.where(mask > 0, 255, self.image)
        
        # Update contours and display
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        # Calculate metrics if ground truth is available
        metrics = self.calculate_metrics_for_history()

        operation_entry = {
            "name": "Selective Fill",
            "params": {
                "Contour Index": contour_index
            }
        }

        # Add metrics if available
        if metrics:
            operation_entry["metrics"] = metrics

        self.operations_history.append(operation_entry)
        self.update_operations_history()



    def open_mask_evaluation_dialog(self):
        """Open dialog to evaluate the current mask against a ground truth mask."""
        # Check if the dialog already exists
        if hasattr(self, 'eval_dialog') and self.eval_dialog.winfo_exists():
            self.eval_dialog.lift()
            return

        self.eval_dialog = Toplevel(self.root)
        self.eval_dialog.title("Mask Performance Evaluator")
        self.eval_dialog.geometry("800x450")  # Increased width for more columns
        self.eval_dialog.transient(self.root)

        # Frame for ground truth selection
        selection_frame = Frame(self.eval_dialog, pady=10, padx=10)
        selection_frame.pack(fill="x")
        Label(selection_frame, text="Ground Truth Mask:", font=("Arial", 10, "bold")).pack(anchor="w", pady=5)

        # File selection section
        file_frame = Frame(selection_frame)
        file_frame.pack(fill="x", pady=5)
        self.ground_truth_entry = Entry(file_frame, width=40)
        self.ground_truth_entry.pack(side="left", padx=5, expand=True, fill="x")
        Button(file_frame, text="Browse",
            command=self.select_ground_truth_mask).pack(side="left", padx=5)

        # Separator
        ttk.Separator(self.eval_dialog, orient='horizontal').pack(fill='x', pady=10)

        # Results section title
        Label(self.eval_dialog, text="Evaluation Results", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)

        # Table for results - now with 8 columns
        result_frame = Frame(self.eval_dialog, pady=10, padx=10)
        result_frame.pack(fill="both", expand=True)

        # Create treeview widget for results with new columns
        columns = ("Index", "Accuracy", "Precision", "Recall", "F1 Score", "Dice", "Jaccard", "Hausdorff")
        self.metrics_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=10)

        # Set column headings
        self.metrics_tree.heading("Index", text="#")
        self.metrics_tree.heading("Accuracy", text="Accuracy")
        self.metrics_tree.heading("Precision", text="Precision")
        self.metrics_tree.heading("Recall", text="Recall")
        self.metrics_tree.heading("F1 Score", text="F1 Score")
        self.metrics_tree.heading("Dice", text="Dice")
        self.metrics_tree.heading("Jaccard", text="Jaccard")
        self.metrics_tree.heading("Hausdorff", text="Hausdorff")

        # Configure column widths
        self.metrics_tree.column("Index", width=40, anchor="center")
        self.metrics_tree.column("Accuracy", width=80, anchor="center")
        self.metrics_tree.column("Precision", width=80, anchor="center")
        self.metrics_tree.column("Recall", width=80, anchor="center")
        self.metrics_tree.column("F1 Score", width=80, anchor="center")
        self.metrics_tree.column("Dice", width=80, anchor="center")
        self.metrics_tree.column("Jaccard", width=80, anchor="center")
        self.metrics_tree.column("Hausdorff", width=100, anchor="center")

        # Add scrollbar
        scrollbar = Scrollbar(result_frame, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)

        # Pack tree and scrollbar
        scrollbar.pack(side="right", fill="y")
        self.metrics_tree.pack(side="left", fill="both", expand=True)

        # Status section
        self.eval_status_var = StringVar()
        self.eval_status_var.set("Ready to evaluate")
        status_label = Label(self.eval_dialog, textvariable=self.eval_status_var, relief="sunken", anchor="w")
        status_label.pack(side="bottom", fill="x")

        # Evaluation button
        Button(self.eval_dialog, text="Evaluate Current Mask",
            command=self.perform_mask_evaluation,
            font=("Arial", 10, "bold"),
            bg="#4CAF50", fg="white",
            padx=10, pady=5).pack(pady=15)

        # Initialize counter for row indices
        if not hasattr(self, 'eval_index_counter'):
            self.eval_index_counter = 1

        
    def select_ground_truth_mask(self):
        """Select ground truth mask file and update entry."""
        file_path = filedialog.askopenfilename(
            title="Select Ground Truth Mask",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            self.ground_truth_entry.delete(0, "end")
            self.ground_truth_entry.insert(0, file_path)

    def perform_mask_evaluation(self):
        """Evaluate current mask against ground truth and append results to the table."""
        ground_truth_path = self.ground_truth_entry.get()
        if not ground_truth_path:
            messagebox.showerror("Error", "Please select a ground truth mask.")
            return

        if not hasattr(self, 'image') or self.image is None:
            messagebox.showerror("Error", "No current mask to evaluate.")
            return

        try:
            self.eval_status_var.set("Evaluating mask...")

            # Load ground truth mask
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth is None:
                messagebox.showerror("Error", "Failed to load ground truth mask.")
                return

            # Convert current image to grayscale if it's not already
            if len(self.image.shape) == 3:
                current_mask = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                current_mask = self.image.copy()

            # Ensure both masks have the same dimensions
            if current_mask.shape != ground_truth.shape:
                messagebox.showerror("Error", "Current mask and ground truth mask have different dimensions.")
                return

            # Binarize both masks
            _, current_mask_bin = cv2.threshold(current_mask, 127, 255, cv2.THRESH_BINARY)
            _, ground_truth_bin = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)

            # Normalize to 0 and 1
            current_mask_norm = current_mask_bin / 255.0
            ground_truth_norm = ground_truth_bin / 255.0

            # Calculate metrics
            accuracy = self.pixel_accuracy(ground_truth_norm, current_mask_norm)
            prec = self.precision(ground_truth_norm, current_mask_norm)
            rec = self.recall(ground_truth_norm, current_mask_norm)
            f1 = self.f1_score(ground_truth_norm, current_mask_norm)
            
            # Calculate new metrics
            dice = self.dice_coefficient(ground_truth_norm, current_mask_norm)
            jaccard = self.jaccard_index(ground_truth_norm, current_mask_norm)
            
            # Calculate Hausdorff distance (may be slow for large masks)
            try:
                hausdorff = self.hausdorff_distance_metric(ground_truth_norm, current_mask_norm)
                hausdorff_str = f"{hausdorff:.2f}"
            except Exception as e:
                print(f"Hausdorff calculation error: {e}")
                hausdorff_str = "Error"

            # Add new results as a new row (don't clear existing items)
            self.metrics_tree.insert("", "end", values=(
                self.eval_index_counter, 
                f"{accuracy:.4f}", 
                f"{prec:.4f}", 
                f"{rec:.4f}", 
                f"{f1:.4f}",
                f"{dice:.4f}",
                f"{jaccard:.4f}",
                hausdorff_str
            ))

            # Increment index counter for next evaluation
            self.eval_index_counter += 1
            self.eval_status_var.set("Evaluation completed successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during evaluation: {str(e)}")
            self.eval_status_var.set("Evaluation failed.")


    def pixel_accuracy(self, y_true, y_pred):
        """Calculate pixel accuracy between two binary masks using sklearn."""
        # Flatten the arrays for sklearn functions
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return accuracy_score(y_true_flat, y_pred_flat)

    def precision(self, y_true, y_pred):
        """Calculate precision between two binary masks using sklearn."""
        # Flatten the arrays for sklearn functions
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return precision_score(y_true_flat, y_pred_flat, zero_division=0)

    def recall(self, y_true, y_pred):
        """Calculate recall between two binary masks using sklearn."""
        # Flatten the arrays for sklearn functions
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return recall_score(y_true_flat, y_pred_flat, zero_division=0)

    def f1_score(self, y_true, y_pred):
        """Calculate F1 score between two binary masks using sklearn."""
        # Flatten the arrays for sklearn functions
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return f1_score(y_true_flat, y_pred_flat, zero_division=0)


    def dice_coefficient(self, y_true, y_pred):
        """Calculate Dice coefficient between two binary masks."""
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

    def jaccard_index(self, y_true, y_pred):
        """Calculate Jaccard index (IoU) between two binary masks."""
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / (union + 1e-7)

    def hausdorff_distance_metric(self, y_true, y_pred):
        """Calculate Hausdorff distance between two binary masks."""
        # Convert to boolean arrays
        y_true_bool = y_true.astype(bool)
        y_pred_bool = y_pred.astype(bool)
        
        # Get points where mask is True
        y_true_points = np.argwhere(y_true_bool)
        y_pred_points = np.argwhere(y_pred_bool)
        
        # Handle empty masks
        if len(y_true_points) == 0 or len(y_pred_points) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        forward = directed_hausdorff(y_true_points, y_pred_points)[0]
        backward = directed_hausdorff(y_pred_points, y_true_points)[0]
        
        # Return the maximum of the two directed distances
        return max(forward, backward)



    def apply_fill(self):
        """Fill contours based on the selected mode."""
        self.save_state()
        
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
        
        # Create binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a blank mask
        mask = np.zeros_like(gray)
        
        if self.fill_mode_var.get() == "total":
            # Total fill mode - fill based on outermost contours only
            for i, contour in enumerate(contours):
                # Check if this is an outer contour (no parent)
                if hierarchy is not None and hierarchy[0][i][3] == -1:
                    # Check if the contour area is mostly black
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = binary[y:y+h, x:x+w]
                    if roi.size > 0:
                        # Calculate percentage of white pixels
                        white_percentage = np.sum(roi == 255) / roi.size
                        
                        # Only fill if the contour isn't mostly black (removed area)
                        if white_percentage > 0.1:
                            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        else:  # Hierarchical fill
            # Process each contour based on hierarchy
            for i, contour in enumerate(contours):
                # Check if the contour area is mostly black
                x, y, w, h = cv2.boundingRect(contour)
                if w > 0 and h > 0:
                    roi = binary[y:y+h, x:x+w]
                    if roi.size > 0:
                        # Calculate percentage of white pixels
                        white_percentage = np.sum(roi == 255) / roi.size
                        
                        # Only fill if the contour isn't mostly black (removed area)
                        if white_percentage > 0.1:
                            # Fill this contour
                            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                            
                            # If this contour has children (holes), cut them out
                            if hierarchy is not None and len(hierarchy) > 0:
                                child = hierarchy[0][i][2]
                                while child != -1:  # -1 indicates no more children
                                    # Check if child contour is mostly white
                                    child_x, child_y, child_w, child_h = cv2.boundingRect(contours[child])
                                    if child_w > 0 and child_h > 0:
                                        child_roi = binary[child_y:child_y+child_h, child_x:child_x+child_w]
                                        if child_roi.size > 0:
                                            child_white_percentage = np.sum(child_roi == 255) / child_roi.size
                                            
                                            # Only cut out if the child contour isn't mostly black
                                            if child_white_percentage > 0.1:
                                                cv2.drawContours(mask, [contours[child]], -1, 0, thickness=cv2.FILLED)
                                    
                                    # Move to the next child
                                    child = hierarchy[0][child][0]
        
        # Apply the mask to create the final image
        if len(self.image.shape) == 3:
            # Create a 3-channel mask
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # Where mask is white, set the result to white, otherwise keep original
            self.image = np.where(mask_3ch > 0, 255, self.image)
        else:
            self.image = np.where(mask > 0, 255, self.image)
        
        # Update contours and display
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()

    def open_color_analysis_dialog(self):
        """Open the color analysis dialog."""
        dialog = self.open_dialog("Contour Color Analysis", self.setup_color_analysis_tab)
        return dialog

    def setup_color_analysis_tab(self, parent):
        """Setup the contour color analysis tab for black/white detection."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Instructions
        Label(control_frame, text="Click on a contour to check if it's black or white", 
            font=("Arial", 10, "bold")).pack(anchor="w", pady=5)
        
        # Enable color analysis mode button
        self.color_analysis_mode = False
        self.color_analysis_button = Button(
            control_frame, 
            text="Enable Color Analysis Mode", 
            command=self.toggle_color_analysis_mode
        )
        self.color_analysis_button.pack(fill="x", pady=5)
        
        # Color display frame
        color_frame = Frame(control_frame, bd=2, relief="groove")
        color_frame.pack(fill="x", pady=10)
        
        Label(color_frame, text="Contour Analysis", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Create a frame for the color sample
        self.color_sample_frame = Frame(color_frame, width=100, height=50, bg="#FFFFFF")
        self.color_sample_frame.pack(pady=10)
        self.color_sample_frame.pack_propagate(False)
        
        # Simplified color information
        self.color_info_frame = Frame(color_frame)
        self.color_info_frame.pack(fill="x", pady=5)
        
        self.rgb_label = Label(self.color_info_frame, text="Result: ---")
        self.rgb_label.pack(anchor="w")
        
        # Keep area label, remove HSV and Hex
        self.hsv_label = Label(self.color_info_frame, text="")
        self.hsv_label.pack(anchor="w")
        self.hex_label = Label(self.color_info_frame, text="")
        self.hex_label.pack(anchor="w")
        self.area_label = Label(self.color_info_frame, text="Area: --- pixels")
        self.area_label.pack(anchor="w")

    def toggle_color_analysis_mode(self):
        """Toggle between color analysis mode and normal mode."""
        self.color_analysis_mode = not self.color_analysis_mode
        
        if self.color_analysis_mode:
            self.color_analysis_button.config(text="Disable Color Analysis Mode")
            self.status_bar.config(text="Color analysis mode enabled. Click on a contour to analyze.")
        else:
            self.color_analysis_button.config(text="Enable Color Analysis Mode")
            self.status_bar.config(text="Color analysis mode disabled")

    def save_operations_history(self):
        """Save the operations history to a text file."""
        if not self.operations_history:
            messagebox.showinfo("No History", "No operations to save.")
            return

        import time
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        timestamp = int(time.time() % 10000) # 4-digit time
        filename = f"op_{image_name}_{timestamp}.txt"

        with open(filename, "w") as f:
            for i, op in enumerate(self.operations_history):
                f.write(f"{i+1}. {op['name']}\n")
                # Save parameters
                for key, value in op['params'].items():
                    f.write(f"   {key}: {value}\n")
                
                # Save metrics if available
                if 'metrics' in op:
                    f.write("   Performance Metrics:\n")
                    for metric_name, metric_value in op['metrics'].items():
                        f.write(f"      {metric_name}: {metric_value}\n")
                
                f.write("\n")

        messagebox.showinfo("Saved", f"Operations history saved to {filename}")


    def open_operations_history(self):
        """Open or raise the operations history window."""
        if not hasattr(self, 'history_window') or not self.history_window.winfo_exists():
            self.history_window = Toplevel(self.root)
            self.history_window.title("Operations History")
            self.history_window.geometry("400x500")
            self.history_window.protocol("WM_DELETE_WINDOW", self.close_history_window)
            
            # Main container frame
            main_frame = Frame(self.history_window)
            main_frame.pack(fill="both", expand=True)
            
            # Button frame at bottom - pack this FIRST
            button_frame = Frame(self.history_window)
            button_frame.pack(side="bottom", fill="x")
            
            # Save button - now packed to the right of the bottom frame
            #Button(button_frame, text="Save History", command=self.save_operations_history).pack(side="right", pady=5, padx=5)
            Button(button_frame, text="Save Profile", command=self.save_profile).pack(side="right", pady=5, padx=5)

            
            
            # Scrollable frame - pack this AFTER the button frame
            canvas = Canvas(main_frame)
            scrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            self.history_frame = Frame(canvas)
            self.history_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=self.history_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            self.update_operations_history()
        
        self.history_window.lift()


    def update_operations_history(self):
        """Update the operations history display."""
        if not hasattr(self, 'history_window') or not self.history_window.winfo_exists():
            return
            
        # Clear existing content
        for widget in self.history_frame.winfo_children():
            widget.destroy()
            
        # Add operations
        for i, op in enumerate(self.operations_history):
            op_text = f"{i+1}. {op['name']}\n"
            
            # Add parameters
            for key, value in op['params'].items():
                op_text += f" {key}: {value}\n"
            
            # Add metrics if available
            if 'metrics' in op:
                op_text += " Performance Metrics:\n"
                for metric_name, metric_value in op['metrics'].items():
                    op_text += f"  {metric_name}: {metric_value}\n"
                    
            Label(self.history_frame, text=op_text, anchor="w", justify="left").pack(fill="x")


    def close_history_window(self):
        """Close the history window."""
        if hasattr(self, 'history_window'):
            self.history_window.destroy()
            del self.history_window

    def upload_ground_truth_mask(self):
        """Upload a ground truth mask for continuous evaluation."""
        file_path = filedialog.askopenfilename(
            title="Select Ground Truth Mask",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Load the mask to verify it's valid
                ground_truth = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if ground_truth is None:
                    messagebox.showerror("Error", "Failed to load ground truth mask.")
                    return
                    
                # Check if dimensions match the current image
                if self.image.shape[:2] != ground_truth.shape:
                    messagebox.showerror("Error", 
                        "Ground truth mask dimensions don't match the current image.\n"
                        f"Image: {self.image.shape[:2]}, Mask: {ground_truth.shape}")
                    return
                    
                # Store the ground truth mask path
                self.ground_truth_mask_path = file_path
                messagebox.showinfo("Success", "Ground truth mask uploaded successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")


    def calculate_metrics_for_history(self):
        """Calculate metrics against ground truth for operations history."""
        if not hasattr(self, 'ground_truth_mask_path'):
            return None
            
        try:
            # Load ground truth mask
            ground_truth = cv2.imread(self.ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth is None:
                return None
                
            # Convert current image to grayscale if it's not already
            if len(self.image.shape) == 3:
                current_mask = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                current_mask = self.image.copy()
                
            # Binarize both masks
            _, current_mask_bin = cv2.threshold(current_mask, 127, 255, cv2.THRESH_BINARY)
            _, ground_truth_bin = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
            
            # Normalize to 0 and 1
            current_mask_norm = current_mask_bin / 255.0
            ground_truth_norm = ground_truth_bin / 255.0
            
            # Calculate metrics
            accuracy = self.pixel_accuracy(ground_truth_norm, current_mask_norm)
            prec = self.precision(ground_truth_norm, current_mask_norm)
            rec = self.recall(ground_truth_norm, current_mask_norm)
            f1 = self.f1_score(ground_truth_norm, current_mask_norm)
            
            # Return metrics rounded to 2 decimal places
            return {
                "Accuracy": f"{accuracy:.2f}",
                "Precision": f"{prec:.2f}",
                "Recall": f"{rec:.2f}",
                "F1 Score": f"{f1:.2f}"
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return None


    def create_menu_bar(self):
        """Create a single consolidated menu bar with essential functions."""
        menubar = Menu(self.root)
        
        # Create a single Tools menu
        contour_tools_menu = Menu(menubar, tearoff=0)
        
        # Add File menu items
        """
        contour_tools_menu.add_command(label="Open Image", command=self.load_new_image)
        contour_tools_menu.add_command(label="Upload Ground Truth Mask", command=self.upload_ground_truth_mask)
        """
        
        
        # Add Edit menu items
        contour_tools_menu.add_command(label="Undo", command=self.undo)
        contour_tools_menu.add_command(label="Redo", command=self.redo)
        contour_tools_menu.add_command(label="Reset Image", command=self.reset_image)
        contour_tools_menu.add_separator()

               
        # Add Morphology operations
        contour_tools_menu.add_command(label="Erode", command=lambda: self.open_morphology_dialog("erode"))
        contour_tools_menu.add_command(label="Dilate", command=lambda: self.open_morphology_dialog("dilate"))
        contour_tools_menu.add_separator()
        
        
        # Add the new Fill Below Threshold option
        contour_tools_menu.add_command(label="Fill Below Threshold", command=self.open_fill_below_threshold_dialog)
        contour_tools_menu.add_separator()

        # Add Circularity
        contour_tools_menu.add_command(label="Circularity", command=self.open_iso_dialog)
        # Add Median Blur (only)
        contour_tools_menu.add_command(label="Median Blur", command=self.open_median_blur_dialog)
        # Add Aspect Ratio
        contour_tools_menu.add_command(label="Aspect Ratio Filter", command=self.open_aspect_dialog)
        contour_tools_menu.add_separator()


        """
        # Add Area Threshold
        contour_tools_menu.add_command(label="Area Threshold", command=self.open_area_dialog)

        # Add Eccentricity
        contour_tools_menu.add_command(label="Eccentricity", command=self.open_eccentricity_dialog)        
        # Add Fill Options
        contour_tools_menu.add_command(label="Fill Options", command=self.open_fill_dialog)
        contour_tools_menu.add_separator()
        """
        
        #Add evaluation tools
        contour_tools_menu.add_command(label="Mask Performance Evaluator", command=self.open_mask_evaluation_dialog)
        contour_tools_menu.add_separator()
       
        
        # Add Save Profile
        contour_tools_menu.add_command(label="Operations History", command=self.open_operations_history)
        #contour_tools_menu.add_command(label="Save Profile", command=self.save_profile)
        contour_tools_menu.add_command(label="Save Image", command=self.save_image)      
        contour_tools_menu.add_command(label="Exit", command=self.root.quit)
        
        # Add the menu to the menubar
        menubar.add_cascade(label="Tools", menu=contour_tools_menu)
        
        self.root.config(menu=menubar)
        
        
    

     
        
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    def open_fill_below_threshold_dialog(self):
        dialog = Toplevel(self.root)
        dialog.title("Fill Contours Below Threshold")
        dialog.geometry("400x200")
        
        # Threshold slider
        threshold_frame = Frame(dialog)
        threshold_frame.pack(pady=10)
        
        self.threshold_label = Label(threshold_frame, text="Threshold Area: 100")
        self.threshold_label.pack()
        
        self.threshold_slider = Scale(dialog, from_=1, to=1000, orient=HORIZONTAL,
                                    command=self.update_fill_threshold)
        self.threshold_slider.set(100)
        self.threshold_slider.pack(fill=X, padx=10, pady=5)
        
        # Preview and Apply buttons
        button_frame = Frame(dialog)
        button_frame.pack(pady=10)
        
        Button(button_frame, text="Preview", command=self.preview_fill_below_threshold).pack(side=LEFT, padx=5)
        Button(button_frame, text="Apply", command=self.apply_fill_below_threshold).pack(side=LEFT, padx=5)
        Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=LEFT, padx=5)
        
    
    def update_fill_threshold(self, value):
        self.fill_threshold = int(value)
        self.threshold_label.config(text=f"Threshold Area: {self.fill_threshold}")
        
    
    def preview_fill_below_threshold(self):
        self.save_state()
        
        # Create a copy of the image for preview
        preview_image = self.image.copy()
        if len(preview_image.shape) == 2:
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2BGR)
        
        # Get contours
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours below threshold in red
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.fill_threshold:
                cv2.drawContours(preview_image, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)
        
        # Show preview
        self.preview_image = preview_image
        self.update_display(use_highlighted=True)
    
    def apply_fill_below_threshold(self):
        
        self.save_state()
        
        # Get contours
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for filling
        mask = np.zeros_like(gray)
        
        # Fill contours below threshold
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.fill_threshold:
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Apply mask to image
        if len(self.image.shape) == 3:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.image = np.where(mask_3ch > 0, 255, self.image)
        else:
            self.image = np.where(mask > 0, 255, self.image)
        
        # Update display
        self.contours_list = self.detect_contours(spacing=8)
        self.update_display()
        
        # Add to operations history
        operation_entry = {
            "name": "Fill Below Threshold",
            "params": {
                "Threshold Area": self.fill_threshold
            }
        }
        
        metrics = self.calculate_metrics_for_history()
        if metrics:
            operation_entry["metrics"] = metrics
            
        self.operations_history.append(operation_entry)
        self.update_operations_history()        
            
            















    
    
    
    
    
        
        
        

    def open_dialog(self, title, setup_func):
        """Create a persistent dialog window with scrollbars if needed."""
        dialog_name = f"{title.lower().replace(' ', '_')}_dialog"
        
        # Check if dialog already exists
        if hasattr(self, dialog_name) and getattr(self, dialog_name).winfo_exists():
            # Bring existing window to front
            dialog = getattr(self, dialog_name)
            dialog.lift()
            dialog.focus_set()
            return dialog
        
        # Create new dialog
        dialog = Toplevel(self.root)
        
        # Set title to include function name
        function_name = setup_func.__name__.replace("setup_", "").replace("_tab", "").replace("_", " ").title()
        dialog.title(f"{title} - {function_name}")
        
        # Make dialog stay on top but don't grab focus
        dialog.attributes("-topmost", True)
        
        dialog.protocol("WM_DELETE_WINDOW", lambda: self.close_dialog(dialog, dialog_name))
        
        # Store reference to prevent garbage collection
        setattr(self, dialog_name, dialog)
        
        # Create a canvas with scrollbars for potentially long content
        canvas_frame = Frame(dialog)
        canvas_frame.pack(fill="both", expand=True)
        
        # Create canvas and scrollbar
        vscrollbar = Scrollbar(canvas_frame, orient="vertical")
        vscrollbar.pack(side="right", fill="y")
        
        canvas = Canvas(canvas_frame, yscrollcommand=vscrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        
        vscrollbar.config(command=canvas.yview)
        
        # Create a frame inside the canvas for content
        content_frame = Frame(canvas)
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        
        # Set up the dialog content in the content frame
        setup_func(content_frame)
        
        # Update the scroll region after the content is added
        def configure_scroll_region(event):
            # Update the scrollregion to encompass the inner frame
            canvas.configure(scrollregion=canvas.bbox("all"))
            
            # Get screen height and dialog content height
            screen_height = dialog.winfo_screenheight()
            content_height = content_frame.winfo_reqheight()
            
            # If content is taller than 80% of screen, limit dialog height
            if content_height > screen_height * 0.8:
                dialog_height = int(screen_height * 0.8)
                dialog.geometry(f"{content_frame.winfo_reqwidth() + vscrollbar.winfo_reqwidth() + 20}x{dialog_height}")
            else:
                # Otherwise, size to content plus some padding
                dialog.geometry(f"{content_frame.winfo_reqwidth() + vscrollbar.winfo_reqwidth() + 20}x{content_height + 20}")
        
        content_frame.bind("<Configure>", configure_scroll_region)
        
        return dialog


    def open_median_blur_dialog(self):
        """Open a dialog specifically for median blur."""
        def setup_median_blur_tab(parent):
            control_frame = Frame(parent)
            control_frame.pack(fill="x", padx=10, pady=10)
            
            # Blur Kernel Size slider
            self.blur_kernel_label = Label(control_frame, text=f"Blur Kernel Size: {self.blur_kernel_size}")
            self.blur_kernel_label.pack(anchor="w")
            
            blur_kernel_slider = Scale(control_frame, from_=1, to=21, orient=HORIZONTAL,
                                    command=lambda val: self.update_blur_kernel_size(val))
            blur_kernel_slider.set(self.blur_kernel_size)
            blur_kernel_slider.pack(fill="x")
            
            # Set blur type to median
            self.blur_type = "median"
            
            # Apply button
            Button(control_frame, text="Apply Median Blur",
                command=self.apply_blur).pack(fill="x", pady=10)
        
        dialog = self.open_dialog("Median Blur", setup_median_blur_tab)
        return dialog

    def setup_scale_tab(self, parent):
        """Setup the contour scaling tab."""
        control_frame = Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Scale percentage slider
        scale_frame = Frame(control_frame)
        scale_frame.pack(fill="x", pady=5)
        
        self.scale_percentage_label = Label(scale_frame, text="Scale Contours By: 10%")
        self.scale_percentage_label.pack(anchor="w")
        
        self.scale_percentage = Scale(control_frame, from_=1, to=200, orient="horizontal",
                                    command=self.update_scale_percentage)
        self.scale_percentage.set(10)
        self.scale_percentage.pack(fill="x")
        
        # Preview and Apply buttons
        button_frame = Frame(control_frame)
        button_frame.pack(fill="x", pady=10)
        
        Button(button_frame, text="Preview", command=self.preview_scaled_contours).pack(side="left", padx=5)
        Button(button_frame, text="Apply", command=self.apply_contour_scaling).pack(side="left", padx=5)
        Button(button_frame, text="Cancel", command=self.cancel_scaling).pack(side="left", padx=5)



    def close_dialog(self, dialog, dialog_name):
        """Close the dialog and remove the reference."""
        dialog.destroy()
        if hasattr(self, dialog_name):
            delattr(self, dialog_name)

    def open_area_dialog(self):
        """Open the area threshold dialog."""
        dialog = self.open_dialog("Area Threshold", self.setup_area_dialog)
        return dialog

    def setup_area_dialog(self, dialog):
        """Set up the area threshold dialog content."""
        control_frame = Frame(dialog)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Threshold Area slider with range button
        slider_frame = Frame(control_frame)
        slider_frame.pack(fill="x")
        
        self.threshold_area_label = Label(slider_frame, text=f"Threshold Area: {self.threshold_area}")
        self.threshold_area_label.pack(side="left")
        
        # Add range button
        range_button = Button(slider_frame, text="Set Range",
                            command=lambda: self.open_range_dialog(
                                threshold_area_slider,
                                self.threshold_area_label,
                                lambda val: f"Threshold Area: {int(val)}",
                                self.update_threshold_area
                            ))
        range_button.pack(side="right")
        
        threshold_area_slider = Scale(control_frame, from_=1, to=500, orient=HORIZONTAL,
                                    command=self.update_threshold_area)
        threshold_area_slider.set(self.threshold_area)
        threshold_area_slider.pack(fill="x")
        
        # Mode selection
        mode_frame = Frame(control_frame)
        mode_frame.pack(fill="x", pady=5)
        
        Label(mode_frame, text="Mode:").pack(side="left")
        
        self.area_mode_var = StringVar(value=self.area_mode)
        for mode in ["fill", "mark"]:
            rb = ttk.Radiobutton(mode_frame, text=mode, variable=self.area_mode_var, value=mode,
                                command=lambda: self.set_area_mode(self.area_mode_var.get()))
            rb.pack(side="left")
        
        # Apply button
        Button(control_frame, text="Apply", command=self.thresholdArea_Elimination).pack(fill="x", pady=10)

    def open_scale_dialog(self):
        """Open the contour scaling dialog."""
        dialog = self.open_dialog("Scale Contours", self.setup_scale_tab)
        return dialog
    
    def cancel_fill_operation(self):
        """Cancel the fill operation and clear preview."""
        if hasattr(self, 'preview_image'):
            del self.preview_image
        if hasattr(self, 'filtered_contours'):
            del self.filtered_contours
        self.update_display()

    def create_canvas_area(self, parent):
        """Create scrollable canvas for image display."""
        canvas_frame = Frame(parent)
        canvas_frame.pack(fill=BOTH, expand=True)
        
        # Add horizontal scrollbar
        h_scrollbar = Scrollbar(canvas_frame, orient=HORIZONTAL)
        h_scrollbar.pack(side="bottom", fill="x")
        
        # Add vertical scrollbar
        v_scrollbar = Scrollbar(canvas_frame, orient=VERTICAL)
        v_scrollbar.pack(side="right", fill="y")
        
        # Create canvas
        self.canvas = Canvas(canvas_frame, bg="white",
                            xscrollcommand=h_scrollbar.set,
                            yscrollcommand=v_scrollbar.set)
        self.canvas.pack(fill=BOTH, expand=True)
        
        # Configure scrollbars
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<Motion>", self.update_coordinates)
        
        # Add zoom controls
        zoom_frame = Frame(canvas_frame)
        zoom_frame.pack(side="bottom", fill="x")
        Button(zoom_frame, text="+", command=self.zoom_in).pack(side="left")
        Button(zoom_frame, text="-", command=self.zoom_out).pack(side="left")
        
        # Add coordinates display
        self.coord_display = Label(zoom_frame, text="X: 0, Y: 0")
        self.coord_display.pack(side="right")


    def save_profile(self):
        """Saves complete processing profile including ALL operations from history."""
        import os
        import json
        from datetime import datetime
        from tkinter import filedialog, messagebox

        # Verify we have operations to save
        if not self.operations_history:
            messagebox.showerror("Error", "No operations to save in history!")
            return

        # Prepare profile directory
        PROFILE_DIR = os.path.join(os.getcwd(), "profiles")
        os.makedirs(PROFILE_DIR, exist_ok=True)

        # Convert operations history to executable format
        operations = []
        for op in self.operations_history:
            # Create method name by converting display name to snake_case
            method_name = op['name'].lower().replace(' ', '_')
            
            operations.append({
                "method": method_name,
                "params": op.get('params', {}),  # Directly use saved parameters
                "display_name": op['name'],
                "metrics": op.get('metrics', {})  # Preserve any metrics
            })

        # Build complete profile data
        profile_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "source_script": "contour_editor.py",
                "original_image": self.image_path,
                "operations_count": len(operations)
            },
            "core_parameters": {
                "neighborhood_size": neighborhood_size,
                "color_spec": color_spec,
                "sv1": sv1,
                "sv2": sv2,
                "sv3": sv3,
                "sv4": sv4,
                "sv5": sv5
            },
            "operations": operations,
            "cli_template": (
                f"python contour_editor.py --load-profile PROFILE.json "
                f"--neighborhood {neighborhood_size} "
                f"--color-spec {color_spec} "
                f"--sv-values {sv1},{sv2},{sv3},{sv4},{sv5}"
            )
        }

        # Save to file
        try:
            profile_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Profile", "*.json")],
                initialdir=PROFILE_DIR,
                initialfile=f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            if not profile_path:
                return

            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)

            messagebox.showinfo(
                "Profile Saved",
                f"Successfully saved {len(operations)} operations:\n"
                f"{', '.join([op['display_name'] for op in operations])}\n\n"
                f"Path: {profile_path}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Profile save failed:\n{str(e)}")


        
            





    def init_ui(self):
        """Initialize the redesigned UI layout."""
        self.root.title("Contour Editor")
        self.root.geometry("1000x800")
        
        # Create main menu bar
        self.create_menu_bar()
        
        # Create main frame with canvas area
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Create canvas with scrollbars
        self.create_canvas_area(main_frame)
        
        # Create status bar
        self.status_bar = Label(self.root, text="Ready", bd=1, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")
        
        # Initialize display
        self.update_display()

if __name__ == "__main__":

    import sys

    out_dir = "Output/Results/"
    
  
   
    print(len(sys.argv))
    
    
    if len(sys.argv) != 10:
        print("Usage: python imaproc.py <excel_file>", sys.argv[1])
        sys.exit(1)
        
        
    excel_file = sys.argv[1]
    image_path = sys.argv[2]
   
    color_spec=sys.argv[3]
    neighborhood_size=int(sys.argv[4])
    
    sv1=float(sys.argv[5]) 
    sv2=float(sys.argv[6])
    sv3=float(sys.argv[7])
    sv4=float(sys.argv[8])
    sv5=float(sys.argv[9])
    #flg_hough=float(sys.argv[10])
    
    
    print("sv3:",sv3,"sv4:",sv4,"sv5:",sv5)
  
    """

    color_spec=1
    neighborhood_size=2
    
    sv1=3
    sv2=4
    sv3=5
    sv4=6
    sv5=7

 

    
    image_path="/home/surajit/CV/Unsupervised_Learning_New_NUMBA/341.jpg"
    neighborhood_size=7
    """
    
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))

    # Create output file name
    image_path = out_dir+f"0_{file_name}_{neighborhood_size}{file_extension}"

   
   
    """
    # Ask user to select an image file
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png *.jpg*.jpeg *.bmp *.tiff")]
    )
    """

    if image_path:  # Ensure the user selects a file
        editor = ContourEditor(image_path)
        editor.root.mainloop()  # Start the Tkinter event loop
    else:
        print("No image selected. Exiting.")

    
