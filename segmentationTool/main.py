
#pip install esda
#pip install libpysal




import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

#from pysal.lib.weights import lat2W
from libpysal.weights import lat2W
#from pysal.explore.esda.moran import Moran
from esda.moran import Moran

import tkinter as tk
from tkinter import Tk, Label, Button, Scale, filedialog, Canvas
from tkinter import filedialog, ttk
from tkinter import Tk, Menubutton, Menu, Frame, Toplevel
from tkinter.ttk import Label  # Import Label from ttk
from scipy.spatial import distance_matrix


import subprocess
import os
import sys



from PIL import Image, ImageTk, ImageEnhance 

from z_stat import*
#from z_denoising import*








def initialize_Image_Processing(color_spc, Neighbor):
    """
    Passes the generated Excel file to z_imaproc.py and initializes it.
    """
    global image_path, selected_groups
    # Define the Excel file name
    excel_file = "tool.xlsx"
    
    # Check if the file exists
    if not os.path.exists(excel_file):
        status_label.config(text=f"Error: File '{excel_file}' not found.")
        return

    # Get slider values
    sv1 =str(slider1_var.get())
    sv2 =str(slider2_var.get())
    sv3 =str(slider3_var.get())
    sv4 =str(slider4_var.get())
    sv5 =str(slider5_var.get())
    
    #print("xxxx" , image_path)
    
    # Define the command to run z_imaproc.py with the Excel file as an argument
    command = ["python", "z_imaproc.py", excel_file, image_path, color_spc, Neighbor, sv1, sv2, sv3, sv4, sv5]
    
    try:
        # Run the external script
        subprocess.run(command, check=True)
        #print(f"Successfully initialized image processing with file: {excel_file}")
        print(f"Process Completed Successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running z_imaproc.py: {e}")
    except FileNotFoundError:
        print("Error: z_imaproc.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


"""

def Post_Semantic_Segmentaion():
    global image_path
    command = ["python", "app_beta_vz.py", image_path]
    try:
        # Run the external script
        subprocess.run(command, check=True)
        #print(f"Successfully initialized image processing with file: {excel_file}")
        print(f"Process Completed Successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running z_binary_cleanup.py: {e}")
    except FileNotFoundError:
        print("Error: z_binary_cleanup.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")   
"""


def Post_Semantic_Segmentaion():
    """
    Passes the generated Excel file to z_post_seg_vz.py and initializes it.
    """
    global image_path, selected_groups
    # Define the Excel file name
    excel_file = "tool.xlsx"
    
    # Check if the file exists
    if not os.path.exists(excel_file):
        status_label.config(text=f"Error: File '{excel_file}' not found.")
        return

    # Get slider values
    sv1 =str(slider1_var.get())
    sv2 =str(slider2_var.get())
    sv3 =str(slider3_var.get())
    sv4 =str(slider4_var.get())
    sv5 =str(slider5_var.get())
    
    #print("xxxx" , image_path)
    
    # Define the command to run z_imaproc.py with the Excel file as an argument
    command = ["python", "z_post_seg_vz.py", excel_file, image_path, selected_color_space.get(), neighborhood_entry.get(), sv1, sv2, sv3, sv4, sv5]
    
    try:
        # Run the external script
        subprocess.run(command, check=True)
        #print(f"Successfully initialized image processing with file: {excel_file}")
        print(f"Process Completed Successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running z_post_seg_vz.py: {e}")
    except FileNotFoundError:
        print("Error: z_imaproc.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")





        





def call_binary_cleanup():
    global image_path
    command = ["python", "z_binary_cleanup.py", image_path]
    try:
        # Run the external script
        subprocess.run(command, check=True)
        #print(f"Successfully initialized image processing with file: {excel_file}")
        print(f"Process Completed Successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running z_binary_cleanup.py: {e}")
    except FileNotFoundError:
        print("Error: z_binary_cleanup.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")    

    
def call_AnalyzeCellData():
    global image_path,selected_groups_str
    #print("##############3", selected_groups_str)
    command = ["python", "z_AnalyzeCellData.py", image_path, selected_groups_str, neighborhood_entry.get()]
    try:
        # Run the external script
        subprocess.run(command, check=True)
        #print(f"Successfully initialized image processing with file: {excel_file}")
        print(f"Process Completed Successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running z_AnalyzeCellData.py: {e}")
    except FileNotFoundError:
        print("Error: z_AnalyzeCellData.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

        
def call_Auto_cell_profile():
    global image_path
    command = ["python", "z_Auto_cell_profile.py", image_path,neighborhood_entry.get()]
    try:
        # Run the external script
        subprocess.run(command, check=True)
        #print(f"Successfully initialized image processing with file: {excel_file}")
        print(f"Process Completed Successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running z_Auto_cell_profile.py: {e}")
    except FileNotFoundError:
        print("Error: z_Auto_cell_profile.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")   
    




def Run_Profile():
    global image_path
    command = ["python", "Run_Profile_consoleLog.py", image_path]
    try:
        # Run the external script
        subprocess.run(command, check=True)
        #print(f"Successfully initialized image processing with file: {excel_file}")
        print(f"Process Completed Successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running Run_Profile.py: {e}")
    except FileNotFoundError:
        print("Error: Run_Profile.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 



    

def call_Ero_Dia():
    global image_path
    command = ["python", "z_instance_seg.py", image_path]
    try:
        # Run the external script
        subprocess.run(command, check=True)
        #print(f"Successfully initialized image processing with file: {excel_file}")
        print(f"Process Completed Successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running z_Ero_Dia.py: {e}")
    except FileNotFoundError:
        print("Error: z_Ero_Dia.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")   

def Morfological_Correction():
    global image_path
    if image_path:
        command = ["python", "z_pixcel2.py", image_path]
    else:
        command = ["python", "z_pixcel2.py"]
        
    try:
        # Run the external script
        subprocess.run(command, check=True)
        #print(f"Successfully initialized image processing with file: {excel_file}")
        print(f"Process Completed Successfully...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running z_pixcel2.py: {e}")
    except FileNotFoundError:
        print("Error: z_pixcel2.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")       
    

'''
###Closing the current image file in GUI
def close_current_file():
    """
    Closes the currently opened image file and resets the application state.
    """
    global image, image_tk, points

    # Clear the image from the canvas
    canvas.delete("all")

    # Reset global variables
    image = None
    image_tk = None
    points.clear()

    # Update the text widget to reflect the cleared state
    update_text()

    # Update the status label
    status_label.config(text="Current file closed. Ready to open a new file.")

'''


'''
def close_current_file():
    """
    Closes the current Python script and runs z_imaproc.py.
    """
    
    if not image_path:
        print("Error: No image loaded.")
        return
    
    # Define the command to run z_imaproc.py using the current Python interpreter
    command = [sys.executable, "z_imaproc.py"]

    print("Terminating the current script and starting z_imaproc.py...")

    # Terminate the current script
    root.destroy()  # Close the Tkinter GUI
    sys.exit(0)     # Exit the script

    try:
        print("Closing the current script and starting image processing...")
        # Run z_imaproc.py
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to run z_imaproc.py. Details: {e}")
    except FileNotFoundError:
        print("Error: z_imaproc.py not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
'''









#################################################################################

# Initialize global variables
points = []
image = None
image_tk = None
image_path= None
selected_groups = set()

start_x, start_y = None, None
end_x, end_y = None, None  # Add these
rect_id = None
cropped_image = None
selection_active = False  # To track if region selection is active

selected_groups_str="Geometry"
valid_region_flg=1






# Define column groups

column_groups = {
    "XY": ["X", "Y"],
    
    "Mean":["RMN", "GMN", "BMN", "G1MN", "G2MN"],
    "Mdn":["G1MDN", "G2MDN"],
    "Md": ["G1MD", "G2MD"],
    "Skw": ["RW","GW", "BW", "GS1W","GS2W", "RK"],
    "Kurt":["GK", "BK", "GS1K","GS2K"],
    "RCoef": ["GS1RC","GS2RC"],
    
    "RGB": ["R", "G", "B", "G-1", "G-2"],
    "SD": ["RS", "GS", "BS", "GS1S","GS2S"],
    "Dep": ["RDep", "GDep", "BDep", "GS1Dep","GS2Dep"],
    "Dnsi": ["RDsn", "GDsn", "BDsn", "GS1Dsn","GS2Dsn"],

    "Dep_sd": ["RDpSD", "GDpSD", "BDpSD", "GS1DpSD","GS2DpSD"],
    "Dnsi_sd": ["RDnSD", "GDnSD", "BDnSD", "GS1DnSD","GS2DnSD"]
    
    
}





# Define Analysis Metrics
analysis_metric = {
        "General": ["Count", "Cell Density"],
        "Geometry": ["Perimeter", "Area", "Aspect Ratio"],
        "Shape and Geomentry": [
            "area", "filled_area", "convex_area", "bbox_area",
            "perimeter", "equivalent_diameter", "aspect_ratio", "circularity",
            "solidity", "roundness"
        ],
        "Morphology": [
            "eccentricity", "major_axis_length", "minor_axis_length", "convexity"
        ],
        "Topology": ["euler_number"],
        "Localization": ["centroid", "weighted_centroid", "bbox", "coords"],
        "Orientation": ["orientation"],
        "Intensity": ["mean_intensity", "min_intensity", "max_intensity"],
        "Moments": ["Hu Moments"],
        "Voronoi Entropy": ["Voronoi_Entropy"]
    }








# Initialize Tkinter root window
root = tk.Tk()
root.title("Image Analyzer")

# Flatten all columns
all_columns = [col for group in column_groups.values() for col in group]
column_vars = {col: tk.BooleanVar(value=True, master=root) for col in all_columns}






# Define slider variables
slider1_var = tk.DoubleVar(value=110)  # Default value for slider 1
slider2_var = tk.DoubleVar(value=0.273)  # Default value for slider 2
slider3_var = tk.DoubleVar(value=0)  # Default value for slider 3
slider4_var = tk.DoubleVar(value=0)  # Default value for slider 4
slider5_var = tk.DoubleVar(value=-1)  # Default value for slider 5
        


# Function to open the slider window
def open_slider_window(option):
    """
    Opens a top-level window with sliders based on the selected option.
    """
    # Create a new top-level window
    slider_window = tk.Toplevel(root)
    slider_window.title(f"{option} Options")

    slider_window.geometry("250x400")  # Adjust the size as needed

    # Position the top-level window relative to the parent window
    parent_x = root.winfo_x()  # X position of the parent window
    parent_y = root.winfo_y()  # Y position of the parent window
    slider_window.geometry(f"+{parent_x + 50}+{parent_y + 50}")  # Offset by 50 pixels

    # Make the top-level window modal and keep it on top
    slider_window.grab_set()  # Disable interaction with the main window
    slider_window.transient(root)  # Keep the top-level window on top of the main window

    # Add sliders based on the selected option
       
    if option == "HIP Actual Grascale Analysis":
        slider1 = tk.Scale(slider_window, from_=0, to=255, resolution=0.01, orient=tk.HORIZONTAL, label="Shift Gray", variable=slider1_var)
        slider1.pack(fill=tk.X, padx=5, pady=5)

        slider2 = tk.Scale(slider_window, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Span Gray", variable=slider2_var)
        slider2.pack(fill=tk.X, padx=5, pady=5)
        
        slider3 = tk.Scale(slider_window, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Black Msking Threshold", variable=slider3_var)
        slider3.pack(fill=tk.X, padx=5, pady=5)
        
        slider4 = tk.Scale(slider_window, from_=0, to=10, resolution=0.01, orient=tk.HORIZONTAL, label="Texture Contrast Threshold", variable=slider4_var)
        slider4.pack(fill=tk.X, padx=5, pady=5)
        
        slider5 = tk.Scale(slider_window, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Randomness index", variable=slider5_var)
        slider5.pack(fill=tk.X, padx=5, pady=5)

    '''    
    #"Reset" button to reset slider values to default
    reset_button = tk.Button(slider_window, text="Reset", command=lambda: reset_sliders(slider_window, option))
    reset_button.pack(pady=5)

    # "OK" button to confirm the slider values
    ok_button = tk.Button(slider_window, text="OK", command=lambda: on_ok_button_click(slider_window))
    ok_button.pack(pady=10)
    
    '''
    
    # Create a frame to hold the Reset and OK buttons
    button_frame = tk.Frame(slider_window)
    button_frame.pack(fill=tk.X, padx=5, pady=10)
   
    # Add a "Reset" button to reset slider values to default
    reset_button = tk.Button(button_frame, text="Reset", command=lambda: reset_sliders(slider_window, option))
    reset_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    # Add an "OK" button to confirm the slider values
    ok_button = tk.Button(button_frame, text="OK", command=lambda: on_ok_button_click(slider_window))
    ok_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)


def reset_sliders(slider_window, option):
    """
    Resets the slider values to their default values.
    """
    if option == "HIP Actual Grayscale Analysis":
        slider1_var.set(0)  # Default value for slider 1
        slider2_var.set(0.273)  # Default value for slider 2
        slider3_var.set(0)  # Default value for slider 3
        slider4_var.set(0)  # Default value for slider 4
        slider5_var.set(0)  # Default value for slider 5
    status_label.config(text="Sliders reset to default values.")


def on_ok_button_click(slider_window):
    """
    Called when the OK button is clicked in the slider window.
    """
    # Release the grab and re-enable the main window
    slider_window.grab_release()
    root.focus_set()  # Bring focus back to the main window
    slider_window.destroy()  # Close the slider window
    


# Modify the dropdown selection logic
def on_dropdown_select(*args):
    """
    Called when the user selects an option from the dropdown.
    """
    # Get the selected option
    selected_option = selected_color_space.get()


    # Check if the selected option requires sliders
    if selected_option in ["HIP Actual Grascale Analysis"]: 
        # Open the slider window with the selected option
        open_slider_window(selected_option)









DEFAULT_IMAGE_PATH = "pic2.png"



def resize_background_image(event=None):
    """
    Resize the background image to stretch horizontally and fit the canvas width.
    The height will remain unchanged to match the canvas height.
    The loaded image (if any) will be displayed on top of the background image.
    """
    global default_image, default_image_tk, image, image_tk

    # Get the current canvas dimensions
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # Ensure canvas dimensions are valid
    if canvas_width > 1 and canvas_height > 1:
        # Resize the background image (default image)
        if default_image is not None:
            # Stretch the image horizontally to match the canvas width
            new_width = canvas_width
            new_height = canvas_height  # Keep the height unchanged

            # Resize the background image using OpenCV
            resized_default_image = cv2.resize(default_image, (new_width, new_height))

            # Convert the resized background image to a format suitable for Tkinter
            resized_default_image_pil = Image.fromarray(resized_default_image)
            default_image_tk = ImageTk.PhotoImage(resized_default_image_pil)

        # Update the canvas with the resized background image
        canvas.delete("all")  # Clear the canvas

        # Display the resized background image
        if default_image is not None:
            canvas.create_image(0, 0, anchor=tk.NW, image=default_image_tk)

        # Display the loaded image (if any) at its original size on top of the background image
        if image is not None:
            # Convert the loaded image to a format suitable for Tkinter
            image_pil = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image_pil)

            # Display the loaded image at its original size on top of the background image
            canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

            # Update the scroll region to match the size of the loaded image
            canvas.config(scrollregion=(0, 0, image.shape[1], image.shape[0]))
    else:
        # If canvas dimensions are not valid, skip resizing
        status_label.config(text="Canvas dimensions not valid. Skipping resizing.")



'''
def open_image():
    """
    Open and display a new image on top of the background image.
    The loaded image will retain its original size, and scrollbars will be used to navigate it.
    """
    global image, image_tk, image_path

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    if file_path:
        # Debug: Print the file path
        print(f"Loading image from: {file_path}")

        # Load the image
        image = cv2.imread(file_path)
        
        # Debug: Print the type and shape of the image
        print(f"Image type: {type(image)}")
        if image is not None:
            print(f"Image shape: {image.shape}")

        # Check if the image was loaded successfully
        if image is None:
            status_label.config(text=f"Error: Unable to load image from {file_path}")
            return
        
        # Convert the image from BGR to RGB
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            status_label.config(text=f"Error converting image: {e}")
            return

        # Store the image path
        image_path = file_path

        # Enable the opacity slider after loading the image
        opacity_slider.config(state=tk.NORMAL)
        opacity_slider.set(1.0)  # Set slider to 1.0 (fully opaque)

        # Display the loaded image at its original size on top of the background image
        resize_background_image()

        # Clear any previous points and update the text widget
        points.clear()
        update_text()





'''


def open_image():
    """
    Open and display a new image on top of the background image.
    The loaded image will retain its original size, and scrollbars will be used to navigate it.
    """
    global image, image_tk, image_path

    #file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    file_path = filedialog.askopenfilename(title="Open Image",
                                           filetypes=[("Image Files", "*.png *.jpg *.bmp")])    
    
    if file_path:
        image_path = file_path  # Store the image path
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
        # Enable the opacity slider after loading the image
        opacity_slider.config(state=tk.NORMAL)
        opacity_slider.set(1.0)  # Set slider to 1.0 (fully opaque)

        
        # Display the loaded image at its original size on top of the background image
        resize_background_image()
        
        points.clear()
        update_text()

def default_image():
    """
    Load and display the default background image.
    """
    global default_image, default_image_tk

    file_path = DEFAULT_IMAGE_PATH  # Always use the default image
    default_image = cv2.imread(file_path)

    if default_image is not None:
        default_image = cv2.cvtColor(default_image, cv2.COLOR_BGR2RGB)

        # Bind the canvas resize event to the resize_background_image function
        canvas.bind("<Configure>", resize_background_image)

        # Schedule the initial display of the default image after the canvas is fully initialized
        root.after(100, resize_background_image)  # Delay to ensure canvas dimensions are valid
    else:
        status_label.config(text=f"Error: Unable to load image from {file_path}")

# Rest of your code remains unchanged...









'''
def resize_background_image(event=None):
    """
    Resize the background image to fit the canvas while maintaining aspect ratio.
    The loaded image (if any) will retain its original size, and scrollbars will be used to navigate it.
    """
    global default_image, default_image_tk, image, image_tk

    # Get the current canvas dimensions
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # Ensure canvas dimensions are valid
    if canvas_width > 1 and canvas_height > 1:
        # Resize the background image (default image)
        if default_image is not None:
            h, w = default_image.shape[:2]
            aspect_ratio = w / h

            if canvas_width / canvas_height > aspect_ratio:
                new_height = canvas_height
                new_width = int(aspect_ratio * new_height)
            else:
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)

            # Resize the background image using OpenCV
            resized_default_image = cv2.resize(default_image, (new_width, new_height))

            # Convert the resized background image to a format suitable for Tkinter
            resized_default_image_pil = Image.fromarray(resized_default_image)
            default_image_tk = ImageTk.PhotoImage(resized_default_image_pil)

        # Update the canvas with the resized background image
        canvas.delete("all")  # Clear the canvas

        # Display the resized background image
        if default_image is not None:
            canvas.create_image(0, 0, anchor=tk.NW, image=default_image_tk)

        # Display the loaded image (if any) at its original size
        if image is not None:
            # Convert the loaded image to a format suitable for Tkinter
            image_pil = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image_pil)

            # Display the loaded image at its original size
            canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

            # Update the scroll region to match the size of the loaded image
            canvas.config(scrollregion=(0, 0, image.shape[1], image.shape[0]))
    else:
        # If canvas dimensions are not valid, skip resizing
        print("Canvas dimensions not valid. Skipping resizing.")

def open_image():
    """
    Open and display a new image on top of the background image.
    The loaded image will retain its original size, and scrollbars will be used to navigate it.
    """
    global image, image_tk, image_path

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    if file_path:
        image_path = file_path  # Store the image path
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the loaded image at its original size
        resize_background_image()

        points.clear()
        update_text()

def default_image():
    """
    Load and display the default background image.
    """
    global default_image, default_image_tk

    file_path = DEFAULT_IMAGE_PATH  # Always use the default image
    default_image = cv2.imread(file_path)

    if default_image is not None:
        default_image = cv2.cvtColor(default_image, cv2.COLOR_BGR2RGB)

        # Bind the canvas resize event to the resize_background_image function
        canvas.bind("<Configure>", resize_background_image)

        # Schedule the initial display of the default image after the canvas is fully initialized
        root.after(100, resize_background_image)  # Delay to ensure canvas dimensions are valid
    else:
        print(f"Error: Unable to load image from {file_path}")

# Rest of your code remains unchanged...

'''








# Compute Statistics With a User defined Neighborhood.
def compute_stat(image, x,y, n):
    
    from scipy.stats import skew, kurtosis
    neighborhood_size=n
    h,w,_=image.shape
    R = image[:,:,0].astype(float)
    G = image[:,:,1].astype(float)
    B = image[:,:,2].astype(float)
    Gscale1=np.round(R/3+G/3+B/3) #round(0.299 * R + 0.587 * G + 0.114 * B)
    Gscale2=np.round(R/3+G/3+B/3)
   
    half_size = neighborhood_size // 2

    # Pad the channels with zeros to handle border pixels
    padded_R = np.pad(R, half_size, mode='constant', constant_values=0)
    padded_G = np.pad(G, half_size, mode='constant', constant_values=0)
    padded_B = np.pad(B, half_size, mode='constant', constant_values=0)
    padded_GS1 = np.pad(Gscale1, half_size, mode='constant', constant_values=0)
    padded_GS = np.pad(Gscale2, half_size, mode='constant', constant_values=0)
    

    # Get nxn neighborhood for each channel
    R_neighborhood = padded_R[y:y + neighborhood_size, x:x + neighborhood_size]
    G_neighborhood = padded_G[y:y + neighborhood_size, x:x + neighborhood_size]
    B_neighborhood = padded_B[y:y + neighborhood_size, x:x + neighborhood_size]
    GS1_neighborhood = padded_GS1[y:y + neighborhood_size, x:x + neighborhood_size]    
    GS_neighborhood = padded_GS[y:y + neighborhood_size, x:x + neighborhood_size]

    print("***********\n", R_neighborhood, "\n",G_neighborhood, "\n", B_neighborhood)
    
    def mode(arr):
        unique, counts = np.unique(arr, return_counts=True)
        max_count = np.max(counts)
        modes = unique[counts == max_count]
        return np.mean(modes)
    
    
    
    R_mn, G_mn, B_mn, Gscale1_mn, Gscale2_mn=np.mean(R_neighborhood), np.mean(G_neighborhood), np.mean(B_neighborhood), np.mean(GS1_neighborhood),np.mean(GS_neighborhood)
    R_mdn, G_mdn, B_mdn, Gscale1_mdn, Gscale2_mdn=np.median(R_neighborhood), np.median(G_neighborhood), np.median(B_neighborhood), np.median(GS1_neighborhood),np.median(GS_neighborhood)
    R_md, G_md, B_md, Gscale1_md, Gscale2_md=mode(R_neighborhood), mode(G_neighborhood), mode(B_neighborhood), mode(GS1_neighborhood), mode(GS_neighborhood)
  
    R_std, G_std, B_std, Gscale1_std, Gscale2_std =np.std(R_neighborhood), np.std(G_neighborhood), np.std(B_neighborhood), np.std(GS1_neighborhood),np.std(GS_neighborhood)
    R_sk, G_sk, B_sk, Gscale1_sk, Gscale2_sk =skew(R_neighborhood.flatten()), skew(G_neighborhood.flatten()), skew(B_neighborhood.flatten()),skew(GS1_neighborhood.flatten()), skew(GS_neighborhood.flatten())
    R_kurt, G_kurt, B_kurt, Gscale1_kurt, Gscale2_kurt =kurtosis(R_neighborhood.flatten()), kurtosis(G_neighborhood.flatten()), kurtosis(B_neighborhood.flatten()),kurtosis(GS1_neighborhood.flatten()), kurtosis(GS_neighborhood.flatten())
    
    #R_c, G_c, B_c, Gscale1_c, Gscale2_c =calculate_morans_i(R_neighborhood), calculate_morans_i(G_neighborhood), calculate_morans_i(B_neighborhood), calculate_morans_i(GS1_neighborhood),calculate_morans_i(GS_neighborhood)
    Gscale1_c = calculate_morans_i(GS1_neighborhood)
    Gscale2_c = calculate_morans_i(GS_neighborhood)

    R_d, G_d, B_d, Gscale1_d, Gscale2_d =calculate_variogram(R_neighborhood), calculate_variogram(G_neighborhood), calculate_variogram(B_neighborhood), calculate_variogram(GS1_neighborhood),calculate_variogram(GS_neighborhood)
    R_ds, G_ds, B_ds, Gscale1_ds, Gscale2_ds =adjacency_shift(R_neighborhood,0), adjacency_shift(G_neighborhood,0),adjacency_shift(B_neighborhood,0), adjacency_shift(GS1_neighborhood,0),adjacency_shift(GS_neighborhood,0)
  



    return (
        
    round(R_mn),round(G_mn), round(B_mn), round(Gscale1_mn),round(Gscale2_mn),        
    round(R_mdn),round(G_mdn), round(B_mdn), round(Gscale1_mdn),round(Gscale2_mdn),
    round(R_md),round(G_md), round(B_md), round(Gscale1_md),round(Gscale2_md),
    
    round(R_std, 2), round(G_std, 2), round(B_std, 2),
    round(Gscale1_std, 2), round(Gscale2_std, 2),
    round(R_sk, 2), round(G_sk, 2), round(B_sk, 2), round(Gscale2_sk, 2), round(Gscale2_sk, 2),
    round(R_kurt, 2), round(G_kurt, 2), round(B_kurt, 2), round(Gscale1_kurt, 2), round(Gscale2_kurt, 2),
    #round(R_c, 2), round(G_c, 2), round(B_c, 2), round(Gscale1_c, 2), round(Gscale2_c, 2),
    round(Gscale1_c, 2), round(Gscale2_c, 2),
    round(R_d, 2), round(G_d, 2), round(B_d, 2), round(Gscale1_d, 2), round(Gscale2_d, 2),  
    round(R_ds, 2), round(G_ds, 2), round(B_ds, 2), round(Gscale1_ds, 2), round(Gscale2_ds, 2)  

   )




# Capture pixel information
def get_pixel(event):
    global image, cropped_image, start_x, start_y, end_x, end_y
    if image is None:
        return
    
    x, y = int(canvas.canvasx(event.x)), int(canvas.canvasy(event.y))   #Convert coordinates for scrolling

    try:
        n = int(neighborhood_entry.get())
        if n < 3:
            n = 3
        elif n > min(image.shape[0], image.shape[1]):
            n = min(image.shape[0], image.shape[1])  # Limit to image dimensions
    except ValueError:
        n = 3


  # Check if a region is selected
    if selection_active and cropped_image is not None and start_x is not None and start_y is not None and end_x is not None and end_y is not None:
        # Convert global coordinates to local coordinates within the cropped region
        region_x = x - int(min(start_x, end_x))
        region_y = y - int(min(start_y, end_y))
        
        if 0 <= region_x < cropped_image.shape[1] and 0 <= region_y < cropped_image.shape[0]:
            r, g, b = cropped_image[region_y, region_x]
            print(f"Pixel at ({region_x}, {region_y}) in selected region: R={r}, G={g}, B={b}")
            status_label.config(text=f"Pixel at ({region_x}, {region_y}) in selected region: R={r}, G={g}, B={b}")
        else:
            print("Clicked outside the selected region.")
            status_label.config(text="Clicked outside the selected region.")
    else:
        # Retrieve pixel values from the entire image
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            r, g, b = image[y, x]
            print(f"Pixel at ({x}, {y}): R={r}, G={g}, B={b}")
            status_label.config(text=f"Pixel at ({x}, {y}): R={r}, G={g}, B={b}")






    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        r, g, b = image[int(y), int(x)]
        gs1 = int(0.299 * r + 0.587 * g + 0.114 * b)
        gs2 = int(0.333 * r + 0.333 * g + 0.333 * b)
        #rs, gs, bs, gs1s,gs2s, rw, gw, bw, gs2w,rk, gk, bk, gs2k, rc, gc, bc,gs1c,gs2c, rd,gd,bd,gs1d,gs2d = compute_stat(image, x,y, n)

        
        rmn, gmn, bmn, g1mn,g2mn,rmdn,gmdn, bmdn, g1mdn,g2mdn,rmd,gmd, bmd, g1md,g2md, rs, gs, bs, gs1s,gs2s, rw, gw, bw, gs1w,gs2w,rk, gk, bk, gs1k,gs2k, gs1c,gs2c,rd,gd,bd,gs1d,gs2d,rds,gds,bds,gs1ds,gs2ds = compute_stat(image, x,y, n)




    
      
        data_dict = {
            "X": int(x), "Y": int(y), "R": r, "G": g, "B": b, "G-1": gs1, "G-2": gs2,
            
            "RMN":rmn, "GMN":gmn, "BMN":bmn, "G1MN":g1mn, "G2MN":g2mn,
            "RMDN":rmdn, "GMDN":gmdn, "BMDN":bmdn, "G1MDN":g1mdn, "G2MDN":g2mdn,
            "RMD":rmd, "GMD":gmd, "BMD":bmd, "G1MD":g1md, "G2MD":g2md,
                            
            "RS":rs, "GS":gs, "BS":bs, "GS1S":gs1s,"GS2S":gs2s,
            #"RRC":rc, "GRC":gc, "BRC":bc,"GS1RC":gs1c, "GS2RC":gs2c,
            "RW":rw, "GW":gw, "BW":bw, "GS1W":gs1w,"GS2W":gs2w, "RK":rk, "GK":gk, "BK":bk, "GS1K":gs1k,"GS2K":gs2k,
            "GS1RC":gs1c, "GS2RC":gs2c,
            "RDep":rd,"GDep":gd,"BDep":bd,"GS1Dep":gs1d,"GS2Dep":gs2d,
            "RDsn":rds,"GDsn":gds,"BDsn":bds,"GS1Dsn":gs1ds,"GS2Dsn":gs2ds,

            "RDpSD":round(rd/rs,2),"GDpSD":round(gd/gs,2),"BDpSD":round(bd/bs,2),"GS1DpSD":round(gs1d/gs1s,2),"GS2DpSD":round(gs2d/gs2s,2),
            "RDnSD":round(rds/rs,2),"GDnSD":round(gds/gs,2),"BDnSD":round(bds/bs,2),"GS1DnSD":round(gs1ds/gs1s,2),"GS2DnSD":round(gs2ds/gs2s,2)
            
        }

        points.append(data_dict)
        update_text()

# Update text widget
def update_text():
    text_widget.delete(1.0, tk.END)

    selected_columns = [col for col in all_columns if column_vars[col].get()]
    text_widget.insert(tk.END, "\t".join(selected_columns) + "\n")
    text_widget.insert(tk.END, "-" * 80 + "\n")

    for point in points:
        display_data = [str(point[col]) for col in selected_columns if col in point]
        text_widget.insert(tk.END, "\t".join(display_data) + "\n")

# Clear output
def clear_output():
    points.clear()
    update_text()
    status_label.config(text="Output cleared")

# Save to Excel
def save_to_excel():
    if points:
        selected_columns = [col for col in all_columns if column_vars[col].get()]
        df = pd.DataFrame(points)[selected_columns]
        df.to_excel("tool.xlsx", index=False)
        status_label.config(text="Processing Image with current samples...")
    else:
        status_label.config(text="Processing Image with the samples used last time...")





# start_process function to use the cropped image
def start_process():
    global image_path, cropped_image

    progress_bar['value'] = 0  # Reset progress bar
    root.update_idletasks()  # Update the GUI

    # Check if a region is selected and valid
    if valid_region_flg==0 and cropped_image is not None:
        try:
            # Save the cropped image temporarily
            cv2.imwrite("subregion.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
            image_path = "subregion.jpg"  # Update the image path to the cropped image
            status_label.config(text="Processing the selected region...")
        except Exception as e:
            status_label.config(text=f"Error saving cropped image: {e}")
            return
    elif valid_region_flg==1:
        status_label.config(text="Please select a valid region... ")
    else:
        status_label.config(text="No region selected. Processing the entire image...")

    save_to_excel()

    # Simulate a process (e.g., a loop or task)
    for i in range(101):
        progress_bar['value'] = i  # Update progress bar value
        root.update_idletasks()  # Update the GUI
        root.after(50)  # Simulate a delay (50ms)
        status_label.config(text="Preparing to process data...")

    progress_bar['value'] = 100  # Ensure it reaches 100% at the end
    status_label.config(text="Look at console...")

    initialize_Image_Processing(selected_color_space.get(), neighborhood_entry.get())


 







# Function to update coordinates on mouse hover
def update_coordinates(event):
    x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)  # Convert to canvas coordinates
    coordinates_label.config(text=f"X: {int(x)}, Y: {int(y)}")





# Function to handle mouse click
def on_mouse_press(event):
    global start_x, start_y, rect_id
    if selection_active:  # Region selection mode
        # Clear the previous selection
        if rect_id:
            canvas.delete(rect_id)
            rect_id = None
        
        # Start a new selection
        start_x, start_y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="white", dash=(4, 4), width=2)
    else:  # Pixel feature retrieval mode
        get_pixel(event)  # Fetch pixel features on left-click
        
# Function to handle mouse drag
def on_mouse_drag(event):
    global start_x, start_y, rect_id
    if selection_active and start_x is not None and start_y is not None:  # Region selection mode
        cur_x, cur_y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        canvas.coords(rect_id, start_x, start_y, cur_x, cur_y)

def on_mouse_release(event):
    global start_x, start_y, end_x, end_y, rect_id, cropped_image, valid_region_flg

    if selection_active and start_x is not None and start_y is not None:
        end_x, end_y = canvas.canvasx(event.x), canvas.canvasy(event.y)

        # Ensure the coordinates are within the image bounds
        if image is not None:
            x1, y1 = int(min(start_x, end_x)), int(min(start_y, end_y))
            x2, y2 = int(max(start_x, end_x)), int(max(start_y, end_y))

            # Ensure the region is valid
            if x1 < x2 and y1 < y2 and x2 <= image.shape[1] and y2 <= image.shape[0]:
                # Crop the image
                cropped_image = image[y1:y2, x1:x2]
                status_label.config(text="Selected region saved as 'subregion.jpg'")
                valid_region_flg=0
            else:
                status_label.config(text="Invalid region selected. Please try again.")
                valid_region_flg=1
        else:
            status_label.config(text="No image loaded. Please open an image first.")
    else:
        status_label.config(text="No region selected. Please select a region first.")



  
    
# Create GUI elements
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Scrollable Canvas
canvas_frame = tk.Frame(frame)
canvas_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame, cursor="hand2")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scroll_x = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
scroll_y = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

canvas.config(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
canvas.bind("<Button-1>", get_pixel)

# Bind left-click event to on_mouse_press function
canvas.bind("<ButtonPress-1>", on_mouse_press)  # Left-click for region selection or pixel feature retrieval
canvas.bind("<B1-Motion>", on_mouse_drag)       # Left-click drag for region selection
canvas.bind("<ButtonRelease-1>", on_mouse_release)  # Left-click release for region selection
canvas.bind("<Button-3>", get_pixel)  # Right-click to retrieve pixel values




# Coordinates display label
coordinates_label = tk.Label(root, text="X: 0, Y: 0", fg="black")
coordinates_label.pack(side=tk.BOTTOM, pady=5)

# Bind motion event to the canvas
canvas.bind("<Motion>", update_coordinates)







dropdown_frame = tk.Frame(root)
dropdown_frame.pack(fill=tk.X, padx=5, pady=5)


# Single-select Dropdown for Color Space
color_spaces = ["HIP Actual Grascale Analysis"]
selected_color_space = tk.StringVar()
selected_color_space.set(color_spaces[0])  # Default selection

color_dropdown = tk.OptionMenu(dropdown_frame, selected_color_space, *color_spaces)
color_dropdown.pack(side=tk.LEFT, padx=5)

# Bind the dropdown selection event
selected_color_space.trace("w", on_dropdown_select)






# Multi-select Dropdown
selected_text = tk.StringVar()
selected_text.set("Type of Analysis")

dropdown_button = tk.Menubutton(dropdown_frame, textvariable=selected_text, indicatoron=True, borderwidth=1, direction="below")
menu = tk.Menu(dropdown_button, tearoff=0)


# Flatten all columns
all_columns = [col for group in column_groups.values() for col in group]
column_vars = {col: tk.BooleanVar(value=True, master=root) for col in all_columns}





# Toggle group selection
def toggle_group(group):
    global selected_groups_str
    if group in selected_groups:
        selected_groups.remove(group)
    else:
        selected_groups.add(group)
    """    
    selected_text.set(", ".join(selected_groups) if selected_groups else "Type os Analysis")
    selected_groups_str=selected_text.get()
    """
    status_label.config(text=", ".join(selected_groups))
    selected_groups_str=status_label.cget("text")
    
    #print(selected_text.get())
   
#selected_groups_str = ",".join(selected_groups)

for group in analysis_metric.keys():
    menu.add_checkbutton(label=group, onvalue=1, offvalue=0, command=lambda g=group: toggle_group(g))


dropdown_button.config(menu=menu)
dropdown_button.pack(side=tk.LEFT, padx=5)


# Add "Options" menu to the existing dropdown frame
button_menu = tk.Menubutton(dropdown_frame, text="Action", indicatoron=True, borderwidth=1, direction="below")
button_menu.pack(side=tk.LEFT, padx=5, pady=5)

menu = tk.Menu(button_menu, tearoff=0)
button_menu.config(menu=menu)



menu.add_command(label="Open Image", command=open_image)
menu.add_command(label="Clear Widget Output", command=clear_output)
menu.add_separator()

menu.add_command(label="Semantic Segmentation", command=start_process)
menu.add_command(label="Post Segmentation Processing", command=Post_Semantic_Segmentaion)
#menu.add_command(label="Clean Binary Image", command=call_binary_cleanup)
menu.add_command(label="Instance Segmentation Algo-1", command=call_AnalyzeCellData)
#menu.add_command(label="Instance Segmentation Algo-2", command=call_Ero_Dia)
#menu.add_command(label="Morfological Correction", command=Morfological_Correction)
menu.add_command(label="Report", command=call_Auto_cell_profile)
menu.add_separator()
menu.add_command(label="Run Profile", command=Run_Profile)






def toggle_selection():
    global selection_active, rect_id, end_x, end_y
    selection_active = not selection_active
    
    if selection_active:
        toggle_button.config(text="Clear Selection")
        status_label.config(text="Selection mode activated. Click and drag to select a region.")
    else:
        toggle_button.config(text="Toggle Selection")
        if rect_id:
            canvas.delete(rect_id)  # Clear the highlighted region
        status_label.config(text="Selection mode deactivated.")
        

toggle_button = tk.Button(dropdown_frame, text="Toggle Selection", command=toggle_selection)
toggle_button.pack(side=tk.LEFT, padx=5)


'''

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X)

open_button = tk.Button(button_frame, text="Open Image", command=open_image)
open_button.pack(side=tk.LEFT, padx=5, pady=5)


process_button = tk.Button(button_frame, text="Process", command=start_process)
process_button.pack(side=tk.LEFT, padx=5, pady=5)

clear_button = tk.Button(button_frame, text="Clear Output", command=clear_output)
clear_button.pack(side=tk.LEFT, padx=5, pady=5)
'''


Label_Neighborhood_Size = tk.Label(dropdown_frame, text="Neighborhood Size=>")
Label_Neighborhood_Size.pack(side='left', padx=1)
neighborhood_entry = tk.Entry(dropdown_frame, textvariable=tk.StringVar(value="7"),width=2)
neighborhood_entry.pack(side='left', padx=5)




# Add Zoom In and Zoom Out buttons to the dropdown_frame
zoom_in_button = tk.Button(dropdown_frame, text="+", command=lambda: zoom_image(1.25))  # Zoom In by 25%
zoom_in_button.pack(side=tk.LEFT, padx=5)

zoom_out_button = tk.Button(dropdown_frame, text="-", command=lambda: zoom_image(0.8))  # Zoom Out by 20%
zoom_out_button.pack(side=tk.LEFT, padx=5)

# Function to handle zooming
# Function to handle zooming
def zoom_image(scale_factor):
    global image, image_tk

    if image is not None:
        # Get the current dimensions of the loaded image
        h, w = image.shape[:2]

        # Calculate the new dimensions based on the scale factor
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)

        # Resize the loaded image using OpenCV
        resized_image = cv2.resize(image, (new_width, new_height))

        # Update the global image variable with the resized image
        image = resized_image

        # Convert the resized image to a format suitable for Tkinter
        resized_image_pil = Image.fromarray(resized_image)
        image_tk = ImageTk.PhotoImage(resized_image_pil)

        # Update the canvas with the resized loaded image
        canvas.delete("all")  # Clear the canvas

        # Display the resized background image (if any)
        if default_image is not None:
            canvas.create_image(0, 0, anchor=tk.NW, image=default_image_tk)

        # Display the resized loaded image on top of the background image
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

        # Update the scroll region to match the size of the resized loaded image
        canvas.config(scrollregion=(0, 0, new_width, new_height))
    else:
        status_label.config(text="No image loaded. Please open an image first.")




'''
# Add a slider for background image opacity
opacity_label = tk.Label(dropdown_frame, text="Opacity:")
opacity_label.pack(side=tk.LEFT, padx=5)

opacity_slider = tk.Scale(dropdown_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, length=100)
opacity_slider.set(1.0)  # Default opacity (fully visible)
opacity_slider.pack(side=tk.LEFT, padx=5)
'''
# Add a slider for background image opacity (no label, half length)
opacity_slider = tk.Scale(dropdown_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, length=50, showvalue=False, state=tk.DISABLED)  # Half length
opacity_slider.set(1.0)  # Default opacity (fully visible)
opacity_slider.pack(side=tk.LEFT, padx=5)





# Function to adjust background image opacity
def adjust_opacity(value):
    global default_image_tk

    if default_image is not None:
        # Convert the default image to a PIL image
        default_image_pil = Image.fromarray(default_image)

        # Adjust the brightness (simulate opacity) using ImageEnhance
        enhancer = ImageEnhance.Brightness(default_image_pil)
        faded_image_pil = enhancer.enhance(float(value))

        # Convert the faded image back to a Tkinter-compatible format
        default_image_tk = ImageTk.PhotoImage(faded_image_pil)

        # Update the canvas with the faded background image
        canvas.delete("all")  # Clear the canvas

        # Display the faded background image
        canvas.create_image(0, 0, anchor=tk.NW, image=default_image_tk)

        # Display the loaded image (if any) on top of the background image
        if image is not None:
            canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

        # Update the scroll region to match the size of the loaded image
        if image is not None:
            canvas.config(scrollregion=(0, 0, image.shape[1], image.shape[0]))
    else:
        status_label.config(text="No background image loaded.")

# Bind the slider to the adjust_opacity function
opacity_slider.config(command=adjust_opacity)

'''
var=tk.BooleanVar()
check1=tk.Checkbutton(dropdown_frame, text="Include Spatial Statistics", variable=var, command="Toggle_Chk")
check1.pack(padx=1, pady=5)
'''



'''
# Text Widget
text_widget = tk.Text(root, height=10, width=80)
text_widget.pack(fill=tk.X)
'''



# Text Widget with Scrollbars
text_frame = tk.Frame(root)
text_frame.pack(fill=tk.BOTH, expand=True)

# Add horizontal scrollbar
text_hscroll = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
text_hscroll.pack(side=tk.BOTTOM, fill=tk.X)

# Add vertical scrollbar
text_vscroll = tk.Scrollbar(text_frame)
text_vscroll.pack(side=tk.RIGHT, fill=tk.Y)

# Create text widget with scrollbars
text_widget = tk.Text(text_frame, 
                     height=10, 
                     wrap=tk.NONE,  # Disable word wrap to allow horizontal scrolling
                     xscrollcommand=text_hscroll.set,
                     yscrollcommand=text_vscroll.set)
text_widget.pack(fill=tk.BOTH, expand=True)

# Configure scrollbars
text_hscroll.config(command=text_widget.xview)
text_vscroll.config(command=text_widget.yview)











status_label = tk.Label(root, text="", fg="blue")
status_label.pack()


progress_frame = tk.Frame(root)
progress_frame.pack(fill=tk.X, padx=5, pady=5)

progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
progress_bar.pack(fill=tk.X, expand=True)















default_image()

root.mainloop()
