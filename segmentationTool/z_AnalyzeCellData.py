#!C:\Users\GOURAV ROY\Desktop\All Folders\Integrated Workflows\Somatic Cell Detection\CellMasking_FuzzyLogic\.venv\Scripts\python.exe

import cv2
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage.measure import shannon_entropy
from scipy.spatial import Voronoi
from scipy.stats import entropy
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
import os
import sys



# polygon area (shoelace)
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def analyze_and_label_objects(
    image_path, out_dir, output_file_suff, selected_features, base_name,
    min_area=10, mode="excel", font_scale=0.5, font_color=(255, 0, 0), font_thickness=2
):
    """
    Hybrid implementation using skimage.regionprops for most features,
    plus manual calculations for the rest. Keeps visualization and Excel export behavior.
    Returns: cell_data (dict of lists), cell_data2 (dict of globals)
    """

    # ensure out_dir exists
    os.makedirs(out_dir, exist_ok=True)

    # read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return -1

    # copies for drawing
    annotated_image = image.copy()
    bounding_box_image = image.copy()
    detected_contour_image = image.copy()

    # grayscale conversion (robust)
    if image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
        bounding_box_image = annotated_image.copy()
        detected_contour_image = annotated_image.copy()
    elif image.ndim == 2:
        gray = image
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
        bounding_box_image = annotated_image.copy()
        detected_contour_image = annotated_image.copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format: Ensure the input is grayscale or BGR color image.")

    # binary mask
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # contours (kept for drawing detected_contour_image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # regionprops
    labeled = label(binary)
    props = regionprops(labeled, intensity_image=gray)

    total_area = gray.shape[0] * gray.shape[1]

    # feature map (as you provided)
    feature_map = {
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

    # Build the list of requested output feature names (as-is) but also prepare normalized lookup
    selected_columns = ["ID"]
    selected_columns2 = []  # global features
    for key in (selected_features or "").split(","):
        key = key.strip()
        if not key:
            continue
        if key not in feature_map:
            # ignore unknown groups silently (or print a warning)
            continue
        if key in ["General", "Voronoi Entropy"]:
            selected_columns2.extend(feature_map[key])
        else:
            selected_columns.extend(feature_map[key])

    # create a normalized set for membership checks (lower, underscores)
    def norm(s):
        return str(s).strip().lower().replace(" ", "_")

    requested_norm = set(norm(c) for c in selected_columns)
    requested_globals_norm = set(norm(c) for c in selected_columns2)

    # Prepare data containers (we'll add keys dynamically to avoid mismatches)
    cell_data = {"ID": []}
    cell_data2 = {}

    centroids = []
    object_count = 0

    # --- Iterate regionprops and compute all required features ---
    for region in props:
        if region.area < min_area:
            continue
        object_count += 1

        # basic direct regionprops
        minr, minc, maxr, maxc = region.bbox
        area = region.area
        perimeter = getattr(region, "perimeter", 0)
        convex_area = getattr(region, "convex_area", 0)
        equivalent_diameter = getattr(region, "equivalent_diameter", 0)
        eccentricity = getattr(region, "eccentricity", 0)
        major_axis_length = getattr(region, "major_axis_length", 0)
        minor_axis_length = getattr(region, "minor_axis_length", 0)
        solidity = getattr(region, "solidity", 0)
        orientation_deg = np.degrees(region.orientation) if region.orientation is not None else 0.0
        centroid = tuple(region.centroid)
        weighted_centroid = tuple(region.weighted_centroid) if hasattr(region, "weighted_centroid") else (0, 0)
        # bounding-box area computed manually
        bbox_area = (maxr - minr) * (maxc - minc)
        mean_intensity = getattr(region, "mean_intensity", 0)
        min_intensity = getattr(region, "min_intensity", 0)
        max_intensity = getattr(region, "max_intensity", 0)
        hu_moments = getattr(region, "moments_hu", None)  # array-like or None

        # manual derived metrics
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter and perimeter > 0 else 0
        roundness = (4 * area) / (np.pi * (major_axis_length ** 2)) if major_axis_length and major_axis_length > 0 else 0
        convexity = area / convex_area if convex_area and convex_area > 0 else 0
        aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length and minor_axis_length > 0 else 0
        euler_number = getattr(region, "euler_number", None)

        # prepare the per-object row
        cell = {"ID": object_count}

        # add all requested features (we check normalized keys for robustness)
        if "area" in requested_norm or "area" in requested_norm:
            if "Area" in selected_columns or "area" in selected_columns:
                cell["Area"] = area
            else:
                # still allow normalized membership: add lower-cased key if that's how user asked
                pass

        # simpler approach: use the canonical column names you'll save
        def add_if_requested(col_name, value):
            # accepts col_name as final name, and also allows membership by normalized name
            if norm(col_name) in requested_norm:
                cell[col_name] = value

        add_if_requested("Area", area)
        add_if_requested("Perimeter", perimeter)
        add_if_requested("bbox_area", bbox_area)
        add_if_requested("convex_area", convex_area)
        add_if_requested("equivalent_diameter", equivalent_diameter)
        add_if_requested("eccentricity", eccentricity)
        add_if_requested("major_axis_length", major_axis_length)
        add_if_requested("minor_axis_length", minor_axis_length)
        add_if_requested("solidity", solidity)
        add_if_requested("orientation", orientation_deg)
        add_if_requested("centroid", centroid)
        add_if_requested("weighted_centroid", weighted_centroid)
        add_if_requested("euler_number", euler_number)
        add_if_requested("mean_intensity", float(mean_intensity))
        add_if_requested("min_intensity", float(min_intensity))
        add_if_requested("max_intensity", float(max_intensity))
        add_if_requested("circularity", float(circularity))
        add_if_requested("roundness", float(roundness))
        add_if_requested("convexity", float(convexity))
        add_if_requested("aspect_ratio", float(aspect_ratio))

        # Hu moments: expand into Hu1..Hu7 if requested as "Hu Moments" (normalize check)
        if "hu_moments" in requested_norm or norm("Hu Moments") in requested_norm or "hu_moments" in requested_norm:
            if hu_moments is not None:
                for i, hval in enumerate(np.atleast_1d(hu_moments).flatten()):
                    cell[f"Hu{i+1}"] = float(hval)

        # record centroid for Voronoi later
        centroids.append(centroid)

        # annotate bounding box + label on images
        cv2.rectangle(bounding_box_image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
        cv2.putText(annotated_image, str(object_count), (minc, minr - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

        # append this row to cell_data (create keys dynamically if necessary)
        for k, v in cell.items():
            if k not in cell_data:
                cell_data[k] = []
            cell_data[k].append(v)

    # draw all contours separately (keeps your previous behavior)
    for contour in contours:
        cv2.drawContours(detected_contour_image, [contour], -1, (255, 0, 0), 2)

    # --- global features ---
    concentration = object_count / total_area if total_area > 0 else 0
    # Voronoi entropy: compute from centroids if requested
    if "voronoi_entropy" in requested_globals_norm or norm("Voronoi_Entropy") in requested_globals_norm:
        voronoi_entropy_value = 0.0
        try:
            if len(centroids) >= 4:
                vor = Voronoi(np.array(centroids))
                # compute finite region polygon areas
                finite_areas = []
                for region_idx in vor.regions:
                    if not region_idx or -1 in region_idx:  # skip infinite regions
                        continue
                    poly = vor.vertices[region_idx]
                    if len(poly) >= 3:
                        a = polygon_area(np.array(poly))
                        if a > 0:
                            finite_areas.append(a)
                if finite_areas:
                    # normalize to probability distribution, compute Shannon entropy (base 2)
                    p = np.array(finite_areas) / np.sum(finite_areas)
                    voronoi_entropy_value = float(scipy_entropy(p, base=2))
            else:
                # fallback: use shannon_entropy on binary mask
                voronoi_entropy_value = float(shannon_entropy(binary))
        except Exception:
            # any failure fallback
            voronoi_entropy_value = float(shannon_entropy(binary))
        cell_data2["Voronoi_Entropy"] = voronoi_entropy_value

    # always add Count and Cell Density if requested
    if "count" in requested_globals_norm or "count" in (c.lower() for c in selected_columns2):
        cell_data2["Count"] = int(object_count)
    if "cell_density" in requested_globals_norm or "cell_density" in (c.lower() for c in selected_columns2):
        cell_data2["Cell Density"] = float(concentration)

    print(cell_data2)

    # ensure all lists in cell_data are equal length by padding with None
    if cell_data:
        max_len = max(len(v) for v in cell_data.values())
        for k in list(cell_data.keys()):
            while len(cell_data[k]) < max_len:
                cell_data[k].append(None)

    # --- output: console or excel ---
    if mode == "console":
        for i in range(object_count):
            print(f"Object {cell_data['ID'][i]}:")
            for key in selected_columns[1:]:
                print(f"  {key}: {cell_data[key][i]}")
    elif mode == "excel":
       
        excel_filename = output_file_suff+".xlsx"
        
        df = pd.DataFrame(cell_data)
        df.to_excel(excel_filename, index=False)
    
        print(f"Saved Data to {excel_filename} Successfully", out_dir+output_file_suff+"_annotated.jpg")
    
    plt.figure(figsize=(30, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(annotated_image)
    plt.title("Annotated Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(bounding_box_image)
    plt.title("Bounding Box Image")
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(30, 12))
    plt.imshow(detected_contour_image)
    plt.title("Detected Contour Image")
    plt.axis('off')
    plt.show()
    
    cv2.imwrite(output_file_suff+"_annotated.jpg", annotated_image)
    cv2.imwrite(output_file_suff+"_bounding_box.jpg", bounding_box_image)
    cv2.imwrite(output_file_suff+"_detected_contours.jpg", detected_contour_image)
    print("Images Saved Successfully")
    
    return cell_data


def main():

    image_path = sys.argv[1]
    option=sys.argv[2]
    neighborhood_size=sys.argv[3]
    print("Selected Options For Analysis:", option,"\n*******************")

    
    out_dir="Output/Results/"
    image_path="/home/surajit/CV/Unsupervised_Learning_NEW/341.jpg"
    option="Geometry,General,Voronoi Entropy"
    neighborhood_size=7

    
    if len(sys.argv) != 4:
       print("Usage: python imaproc.py <excel_file>", sys.argv[1])
       sys.exit(1)

    image_path = sys.argv[1]
    option=sys.argv[2]
    neighborhood_size=sys.argv[3]
    print("Selected Options For Analysis:", option,"\n*******************")


    

    
    
        
        
    # Extract file name and extension
    
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    # Create output file name
    #input_img = f"out_{file_name}.jpg"
    input_img = out_dir+f"0_{file_name}_{neighborhood_size}{file_extension}"
    output_file_suff = out_dir+f"0_{file_name}_{neighborhood_size}"
    analyze_and_label_objects(input_img, out_dir, output_file_suff, option, file_name)
    
    #image_original = cv2.imread(image_path)
    #analyze_and_label_objects(image_original)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", e)
    finally:
        #input("\nPress Enter to exit...")
        pass
    #cProfile.run('main()') #main()
