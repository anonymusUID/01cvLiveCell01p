import os
import pickle
import json
import cv2
import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sqlite3
from datetime import datetime
import sys
import re

class ProfileProcessor:
    def __init__(self, input_pkl_path, history_db_path):
        self.input_pkl_path = input_pkl_path
        self.history_db_path = history_db_path
        self.image = None
        self.contours_list = []
        self.operations = []
        self.current_profile = None

        # Initialize database connection
        self.db_conn = sqlite3.connect(history_db_path)
        self.create_history_table()

    def create_history_table(self):
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS operation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                operation_name TEXT,
                parameters TEXT,
                metrics TEXT,
                profile_name TEXT
            )
        ''')
        self.db_conn.commit()

    def load_input_data(self):
        try:
            print("[INFO] Loading input data...")
            with open(self.input_pkl_path, 'rb') as f:
                data = pickle.load(f)
                # Expecting a dict with keys 'image', optionally 'contours','operations','profile'
                self.image = data.get('image', None)
                if self.image is None:
                    raise ValueError("Input pickle does not contain key 'image'")
                if 'contours' in data:
                    self.contours_list = data['contours']
                if 'operations' in data:
                    self.operations = data['operations']
                if 'profile' in data:
                    self.current_profile = data['profile']
            if len(self.image.shape) == 2:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            print("[SUCCESS] load_input_data completed successfully")
        except Exception as e:
            print(f"[ERROR] load_input_data not completed: {e}. Going to next...")
            raise

    def detect_contours(self, spacing=8):
        # safe guard
        if self.image is None:
            return []
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        sampled_contours = []
        for contour in contours:
            if len(contour) == 0:
                continue
            reduced_contour = [contour[0][0]]
            for i in range(1, len(contour)):
                if np.linalg.norm(contour[i][0] - reduced_contour[-1]) >= spacing:
                    reduced_contour.append(contour[i][0])
            sampled_contours.append(np.array(reduced_contour))
        self.contours_list = sampled_contours
        return self.contours_list

    # --- helper for param key normalization ---
    def _normalize_key(self, key):
        # Common explicit mapping
        key_map = {
            "Min Area": "min_area",
            "Max Area": "max_area",
            "Min Circularity": "min_circularity",
            "Max Circularity": "max_circularity",
            "Mode": "mode",
            "isoperimetric_mode": "isoperimetric_mode",
            "Blur Kernel Size": "blur_kernel_size",
            "Threshold Area": "threshold_area",
            "Min Aspect Ratio": "min_aspect_ratio",
            "Max Aspect Ratio": "max_aspect_ratio",
            "Blur Type": "blur_type",
            "Fill Mode": "fill_mode",
            "Min Eccentricity": "min_eccentricity",
            "Max Eccentricity": "max_eccentricity",
        }
        if key in key_map:
            return key_map[key]
        # generic: strip, replace spaces and non-alnum with underscore, lower
        s = str(key).strip()
        s = re.sub(r'[^0-9a-zA-Z]+', '_', s)
        s = s.strip('_').lower()
        return s

    def _convert_value(self, v):
        # Convert numeric-looking strings to int/float
        if isinstance(v, str):
            v_strip = v.strip()
            # try int
            try:
                if re.fullmatch(r'-?\d+', v_strip):
                    return int(v_strip)
                if re.fullmatch(r'-?\d+\.\d*', v_strip):
                    return float(v_strip)
            except:
                pass
        return v

    def apply_operation(self, operation):
        op_name = operation.get('method') or operation.get('name') or ''
        params = operation.get('params', {}) or {}

        # Normalize keys
        params_fixed = {}
        for k, v in params.items():
            nk = self._normalize_key(k)
            params_fixed[nk] = self._convert_value(v)

        print(f"[INFO] Starting operation: {op_name} with params: {params_fixed}")

        # Map possible profile method names to actual methods
        operation_map = {
            # names that might appear in older profiles or display names
            'area_threshold_elimination': self.threshold_area_elimination,
            'threshold_area_elimination': self.threshold_area_elimination,
            'threshold_area': self.threshold_area_elimination,
            'aspect_ratio_elimination': self.aspect_ratio_elimination,
            'aspect_ratio': self.aspect_ratio_elimination,
            'eccentricity_elimination': self.eccentricity_elimination if hasattr(self, 'eccentricity_elimination') else None,
            'circularity_filter': self.circularity_filter,
            'isoperimetric_algo': self.circularity_filter,
            'isoperimetric': self.circularity_filter,
            'median_blur': self.apply_median_blur,
            'blur': self.apply_median_blur,
            'apply_blur': self.apply_median_blur if hasattr(self, 'apply_median_blur') else None,
            'fill_below_threshold': self.fill_below_threshold,
            'filtered_selective_fill': self.fill_below_threshold,
            'refine_contours': self.refine_contours,
            'erode': self.apply_erosion,
            'dilate': self.apply_dilation,
            'erosion': self.apply_erosion,
            'dilation': self.apply_dilation,
            'morphological_erode': self.apply_erosion,
            'morphological_dilate': self.apply_dilation,            
            
        }

        # try to find the callable in a case-insensitive manner if exact key not present
        target = operation_map.get(op_name)
        if target is None:
            # try lowercased keys
            target = operation_map.get(op_name.lower())
        if target is None:
            # final fallback: try to find by simple name substring
            for k, fn in operation_map.items():
                if fn is None:
                    continue
                if k in op_name.lower() or op_name.lower() in k:
                    target = fn
                    break

        if target:
            try:
                # call with params_fixed (it will ignore unknown kwargs if function signature doesn't accept them
                # but to be safe, only pass params that match function signature
                import inspect
                sig = inspect.signature(target)
                call_kwargs = {}
                for p in sig.parameters:
                    if p in params_fixed:
                        call_kwargs[p] = params_fixed[p]
                target(**call_kwargs)
                self.record_operation(operation)
                print(f"[SUCCESS] {op_name} completed successfully")
            except TypeError as e:
                # probably unexpected kwarg; try calling without kwargs
                try:
                    print(f"[WARNING] TypeError calling {op_name} with params: {e}. Retrying without kwargs...")
                    target()
                    self.record_operation(operation)
                    print(f"[SUCCESS] {op_name} completed successfully (no-arg fallback)")
                except Exception as e2:
                    print(f"[ERROR] {op_name} not completed after fallback: {e2}. Going to next operation...")
            except Exception as e:
                print(f"[ERROR] {op_name} not completed: {e}. Going to next operation...")
        else:
            print(f"[WARNING] Unknown operation '{op_name}', skipping...")

    # --- operations (same as your original ones, with parameter names matching normalized keys) ---
    def threshold_area_elimination(self, threshold_area=35, area_mode="fill"):
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image.copy()

        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if area_mode == "mark":
            result = self.image.copy()
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= threshold_area:
                    cv2.drawContours(result, [contour], -1, (0, 0, 255), thickness=3)
            self.image = result
        else:
            mask = np.zeros_like(gray_image, dtype=np.uint8)
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area >= threshold_area:
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                    if hierarchy is not None and len(hierarchy) > 0:
                        child = hierarchy[0][i][2]
                        while child != -1:
                            child_area = cv2.contourArea(contours[child])
                            if child_area >= threshold_area:
                                cv2.drawContours(mask, [contours[child]], -1, 0, thickness=cv2.FILLED)
                            child = hierarchy[0][child][0]

            if len(self.image.shape) == 3:
                self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
            else:
                self.image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        self.contours_list = self.detect_contours()

    def aspect_ratio_elimination(self, min_aspect_ratio=0.9, max_aspect_ratio=1.2, aspect_ratio_mode="fill"):
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()

        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if aspect_ratio_mode == "mark":
            self.image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            aspect_ratio = float(w) / h
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                if aspect_ratio_mode == "fill":
                    cv2.drawContours(self.image, [contour], -1, 0, thickness=cv2.FILLED)
                elif aspect_ratio_mode == "mark":
                    cv2.drawContours(self.image, [contour], -1, (0, 0, 255), thickness=3)

        self.contours_list = self.detect_contours()

    def circularity_filter(self, min_area=5, max_area=250, min_circularity=0.31, max_circularity=2.0, isoperimetric_mode="remove"):
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()

        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray, dtype=np.uint8)

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0:
                continue

            circularity = (4 * np.pi * area) / (perimeter ** 2)

            if (min_area <= area <= max_area and
                min_circularity <= circularity <= max_circularity):
                if isoperimetric_mode == 'remove':
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        if isoperimetric_mode == 'remove':
            if len(self.image.shape) == 3:
                self.image = cv2.bitwise_and(self.image, self.image, mask=cv2.bitwise_not(mask))
            else:
                self.image = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask))
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        self.contours_list = self.detect_contours()

    def apply_median_blur(self, blur_kernel_size=5):
        try:
            ks = int(blur_kernel_size)
        except:
            ks = 5
        if ks % 2 == 0:
            ks += 1
        # ensure image is proper dtype
        if self.image is not None:
            self.image = cv2.medianBlur(self.image, ks)
            self.contours_list = self.detect_contours()

    def fill_below_threshold(self, fill_threshold=100):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < fill_threshold:
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        if len(self.image.shape) == 3:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.image = np.where(mask_3ch > 0, 255, self.image)
        else:
            self.image = np.where(mask > 0, 255, self.image)

        self.contours_list = self.detect_contours()

    def refine_contours(self, force_constant=5.0, decay_exponent=2.0, gradient_threshold=30, epsilon=0.0001):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        edge_mask = (gradient > gradient_threshold).astype(np.uint8) * 255
        inverted_edges = cv2.bitwise_not(edge_mask)
        distance = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)

        new_contours = []
        for contour in self.contours_list:
            contour = contour.astype(np.float32)
            centroid = np.mean(contour, axis=0)
            dists = np.linalg.norm(contour - centroid, axis=1)
            d_max = np.max(dists) if np.max(dists) != 0 else 1

            refined_contour = []
            for pt in contour:
                x, y = int(pt[0]), int(pt[1])

                if y >= gradient.shape[0] or x >= gradient.shape[1] or y < 0 or x < 0:
                    refined_contour.append(pt)
                    continue

                if gradient[y, x] > gradient_threshold:
                    refined_contour.append(pt)
                    continue

                r = distance[y, x]
                step = force_constant / ((r + epsilon) ** decay_exponent)
                gx = grad_x[y, x]
                gy = grad_y[y, x]
                mag = np.sqrt(gx * gx + gy * gy)

                if mag == 0:
                    refined_contour.append(pt)
                    continue

                nx = gx / mag
                ny = gy / mag
                d = np.linalg.norm(pt - centroid)
                dielectric = 1.0 + (d / d_max)
                step_adjusted = step / dielectric
                new_x = pt[0] - nx * step_adjusted
                new_y = pt[1] - ny * step_adjusted
                refined_contour.append([new_x, new_y])

            refined_contour = np.array(refined_contour, dtype=np.float32)
            smoothed_contour = refined_contour.copy()

            for i in range(len(refined_contour)):
                prev_pt = refined_contour[i - 1]
                curr_pt = refined_contour[i]
                next_pt = refined_contour[(i + 1) % len(refined_contour)]
                smoothed_contour[i] = (prev_pt + curr_pt + next_pt) / 3.0

            if not cv2.isContourConvex(smoothed_contour):
                try:
                    hull = cv2.convexHull(smoothed_contour, returnPoints=True).squeeze()
                    hull_points = hull if len(hull.shape) == 2 else hull.reshape(-1, 2)
                    adjusted_contour = smoothed_contour.copy()

                    for i, pt in enumerate(smoothed_contour):
                        if cv2.pointPolygonTest(hull_points, (pt[0], pt[1]), False) < 0:
                            distances = np.linalg.norm(hull_points - pt, axis=1)
                            closest_idx = np.argmin(distances)
                            adjusted_contour[i] = hull_points[closest_idx]

                    smoothed_contour = adjusted_contour
                except Exception as e:
                    print(f"Error adjusting contour: {e}")

            new_contours.append(smoothed_contour.astype(np.int32))

        self.contours_list = new_contours
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for contour in self.contours_list:
            cv2.fillPoly(mask, [contour], 255)
        self.image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def record_operation(self, operation):
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO operation_history 
            (timestamp, operation_name, parameters, metrics, profile_name)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            operation.get('name', operation.get('method', 'unknown')),
            json.dumps(operation.get('params', {})),
            json.dumps(operation.get('metrics', {})),
            (self.current_profile.get('metadata', {}).get('source_script')
             if isinstance(self.current_profile, dict) else 'unknown')
        ))
        self.db_conn.commit()

    
    def process_profile(self, profile_path):
        #print(f"Current Profile Path: {profile_path}")
        self.load_input_data()

        if profile_path:
            #print(f"Loading profile from: {profile_path}")
            with open(profile_path, 'r') as f:
                self.current_profile = json.load(f)
            #print(f"Profile content: {self.current_profile}")
        else:
            print("No profile_path provided.")

        if not self.current_profile:
            raise ValueError("No processing profile available")

        for operation in self.current_profile.get('operations', []):
            self.apply_operation(operation)

        return self.image

    def save_output(self, output_path, image_array):
        try:
            if not output_path:
                base = os.path.splitext(os.path.basename(self.input_pkl_path))[0]
                output_path = os.path.join(
                    os.path.dirname(self.input_pkl_path),
                    f"{base}.pkl"
                )
            print("[INFO] Saving output data...")
            with open(output_path, 'wb') as f:
                pickle.dump(image_array, f)
            print(f"[SUCCESS] save_output completed successfully ({output_path})")
            
            ### ADD THESE 2 LINES ###
            png_path = os.path.splitext(output_path)[0] + ".png"  # Auto-generate PNG path
            cv2.imwrite(png_path, self.image)  # Save as PNG
            ##########################
            
            return output_path
        except Exception as e:
            print(f"[ERROR] save_output not completed: {e}")
            raise






    def apply_erosion(self, kernel_size=3):
        """Apply erosion morphological operation"""
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(gray, kernel, iterations=1)
        
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        else:
            self.image = eroded
            
        self.contours_list = self.detect_contours()
    
    def apply_dilation(self, kernel_size=3):
        """Apply dilation morphological operation"""
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        else:
            self.image = dilated
            
        self.contours_list = self.detect_contours()










    def close(self):
        if self.db_conn:
            self.db_conn.close()



    




def main(input_pkl_path, history_db_path, profile_path=None, output_pkl_path=None):
    processor = ProfileProcessor(input_pkl_path, history_db_path)
    try:
        output_image = processor.process_profile(profile_path)
        
        """
        # Display image only if GUI available
        if os.environ.get("DISPLAY") or sys.platform == "win32":
            cv2.imshow("Processed Image", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

        saved_path = processor.save_output(output_pkl_path, output_image)
        print(f"[INFO] Processing complete. Output saved to {saved_path}")
    except Exception as e:
        print(f"[ERROR] Error during processing: {str(e)}")
    finally:
        processor.close()





if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python y_post_seg_scriptSec.py <input_pkl_path> <history_db_path> <profile_path> [output_pkl_path]")
        sys.exit(1)

    input_pkl_path = sys.argv[1]
    history_db_path = sys.argv[2]
    profile_path = sys.argv[3]
    output_pkl_path = sys.argv[4] if len(sys.argv) > 4 else None

    print(f"Input pickle path: {input_pkl_path}")
    print(f"Input File Name: {os.path.splitext(os.path.basename(input_pkl_path))[0]}")
    print(f"History DB path: {history_db_path}")
    print(f"Profile path: {profile_path}")
    print(f"Input Profile Name: {os.path.splitext(os.path.basename(profile_path))[0]}")
    print(f"Output pickle path: {output_pkl_path if output_pkl_path else 'None'}")

    main(input_pkl_path, history_db_path, profile_path, output_pkl_path)

    #wait_for_esc()

    
