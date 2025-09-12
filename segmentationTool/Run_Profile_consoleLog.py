import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
from datetime import datetime

Neighbor = "7"

class ProfileManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Profile Manager")
        self.root.geometry("800x650")
        
        # Configuration
        self.profiles_dir = os.path.join(os.getcwd(), "profiles")
        self.default_input_dir = os.path.join(os.getcwd(), "Input")
        self.default_output_dir = os.path.join(os.getcwd(), "Output")
        
        # Create directories if needed
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.default_input_dir, exist_ok=True)
        os.makedirs(self.default_output_dir, exist_ok=True)
        
        # UI Setup
        self.create_widgets()
        self.load_profiles()
        
    def create_widgets(self):
        """Create all GUI components"""
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Directory selection
        dir_frame = tk.Frame(main_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(dir_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W)
        self.input_dir_entry = tk.Entry(dir_frame)
        self.input_dir_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.input_dir_entry.insert(0, self.default_input_dir)
        tk.Button(dir_frame, text="Browse", command=self.browse_input_dir).grid(row=0, column=2)
        
        tk.Label(dir_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W)
        
        # --- FIX 1: Remove state='readonly' to make the entry box editable ---
        self.output_dir_entry = tk.Entry(dir_frame)
        
        self.output_dir_entry.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.output_dir_entry.insert(0, self.default_output_dir)
        
        # --- FIX 2: Uncomment this line to show the "Browse" button ---
        tk.Button(dir_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2)
        
        dir_frame.columnconfigure(1, weight=1)
        
        # Profile list with Treeview
        list_frame = tk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create Treeview with three columns
        self.tree = ttk.Treeview(list_frame, columns=("filename", "created", "ops"), show="headings")
        self.tree.heading("#0", text="Profile Name", command=lambda: self.sort_treeview(0, False))
        self.tree.heading("filename", text="Filename", command=lambda: self.sort_treeview(1, False))
        self.tree.heading("created", text="Created", command=lambda: self.sort_treeview(2, False))
        self.tree.heading("ops", text="Operations", command=lambda: self.sort_treeview(3, False))
        
        # Configure column widths
        self.tree.column("#0", width=150, anchor=tk.W)
        self.tree.column("filename", width=150, anchor=tk.W)
        self.tree.column("created", width=150, anchor=tk.W)
        self.tree.column("ops", width=100, anchor=tk.W)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.tree.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        
        # Profile details
        detail_frame = tk.Frame(main_frame)
        detail_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(detail_frame, text="Profile Details:").pack(anchor=tk.W)
        self.detail_text = tk.Text(detail_frame, wrap=tk.WORD, height=10)
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(button_frame, text="Refresh", command=self.load_profiles).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Delete", command=self.delete_profile, bg="#ff9999").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run on Single Image", command=self.run_single, bg="#99ff99").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run Batch Process", command=self.run_batch, bg="#99ccff").pack(side=tk.LEFT, padx=5)
        
        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self.show_profile_details)
    
    def browse_input_dir(self):
        """Select input directory"""
        dir_path = filedialog.askdirectory(initialdir=self.default_input_dir)
        if dir_path:
            self.input_dir_entry.delete(0, tk.END)
            self.input_dir_entry.insert(0, dir_path)
    
    def browse_output_dir(self):
        """Select output directory"""
        dir_path = filedialog.askdirectory(initialdir=self.default_output_dir)
        if dir_path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, dir_path)
    
    def load_profiles(self):
        """Load all JSON profiles from the profiles directory"""
        self.tree.delete(*self.tree.get_children())
        
        try:
            profile_files = [f for f in os.listdir(self.profiles_dir) if f.endswith('.json')]
            profile_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.profiles_dir, x)), reverse=True)
            
            for filename in profile_files:
                with open(os.path.join(self.profiles_dir, filename), 'r') as f:
                    profile = json.load(f)
                
                profile_name = os.path.splitext(filename)[0]
                created = profile.get('metadata', {}).get('created', 'Unknown')
                ops_count = len(profile.get('operations', []))
                
                # Insert with separate columns
                self.tree.insert("", "end", text=profile_name, 
                               values=(filename, created, f"{ops_count} operations"))
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not load profiles:\n{str(e)}")
    
    def sort_treeview(self, col, reverse):
        """Sort treeview by column"""
        # Get all items from the tree
        items = [(self.tree.set(item, col), item) for item in self.tree.get_children('')]
        
        # Sort the items
        items.sort(reverse=reverse)
        
        # Rearrange items in sorted positions
        for index, (val, item) in enumerate(items):
            self.tree.move(item, '', index)
        
        # Reverse sort next time
        self.tree.heading(col, command=lambda: self.sort_treeview(col, not reverse))
    
    def show_profile_details(self, event):
        """Show details of the selected profile"""
        selection = self.tree.selection()
        if not selection:
            return
            
        profile_name = self.tree.item(selection[0], "text")
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        
        try:
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            details = []
            
            # Metadata
            details.append("=== METADATA ===")
            details.append(f"Created: {profile.get('metadata', {}).get('created', 'Unknown')}")
            details.append(f"Original Image: {profile.get('metadata', {}).get('original_image', 'Unknown')}")
            details.append(f"Operations: {len(profile.get('operations', []))}")
            
            # Core Parameters
            details.append("\n=== CORE PARAMETERS ===")
            core_params = profile.get('core_parameters', {})
            for param, value in core_params.items():
                details.append(f"{param}: {value}")
            
            # Operations
            details.append("\n=== OPERATIONS ===")
            for op in profile.get('operations', []):
                details.append(f"\n{op.get('display_name', 'Unnamed Operation')}")
                for param, value in op.get('params', {}).items():
                    details.append(f"  {param}: {value}")
            
            self.detail_text.delete(1.0, tk.END)
            self.detail_text.insert(tk.END, "\n".join(details))
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not read profile:\n{str(e)}")
    
    def delete_profile(self):
        """Delete the selected profile"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "No profile selected")
            return
            
        profile_name = self.tree.item(selection[0], "text")
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        
        try:
            os.remove(profile_path)
            self.load_profiles()
            self.detail_text.delete(1.0, tk.END)
            messagebox.showinfo("Success", f"Deleted profile: {profile_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete profile:\n{str(e)}")

    def run_single(self):
        """Run the full pipeline on a single selected image with proper console output"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "No profile selected")
            return
            
        profile_name = self.tree.item(selection[0], "text")
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        
        # Select input image
        input_image = filedialog.askopenfilename(
            initialdir=self.input_dir_entry.get(),
            title="Select Input Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif")]
        )
        if not input_image:
            return
            
        loc_specifier = os.path.splitext(os.path.basename(input_image))[0]
        
        # Prepare output directory
        output_dir = os.path.join(self.output_dir_entry.get(), loc_specifier)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load profile parameters
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            core_params = profile.get('core_parameters', {})
            
            # Get all parameters from profile with defaults
            color_spec = str(core_params.get("color_spec", 1))
            neighborhood = str(core_params.get("neighborhood", 7))
            sv1 = str(core_params.get("sv1", 0))
            sv2 = str(core_params.get("sv2", 0))
            sv3 = str(core_params.get("sv3", 0))
            sv4 = str(core_params.get("sv4", 0))
            sv5 = str(core_params.get("sv5", 0))
            excel_file = "tool.xlsx"
            
            # Pipeline steps
            steps = [
                {
                    "name": "y_imaprocSec.py",
                    "cmd": ["python", "y_imaprocSec.py", excel_file, input_image, color_spec, neighborhood, sv1, sv2, sv3, sv4, sv5, loc_specifier,output_dir],
                    "output": os.path.join(output_dir, "imaproc_output.pkl")
                },
                {
                    "name": "y_post_segSec.py",
                    "cmd": ["python", "y_post_segSec.py", os.path.join(output_dir, loc_specifier+".pkl"), "operational_history.db", profile_path],
                    "output": os.path.join(output_dir, "alpha_output.pkl")
                },
                {
                    "name": "y_AnalyzeCellDataSec.py",
                    "cmd": ["python", "y_AnalyzeCellDataSec.py", os.path.join(output_dir, loc_specifier+".pkl")],
                    "output": os.path.join(output_dir, "results.xlsx")
                },
                {
                    "name": "y_Auto_cell_profileSec.py",
                    "cmd": ["python", "y_Auto_cell_profileSec.py", os.path.join(output_dir, loc_specifier+".xlsx"), os.path.join(output_dir, loc_specifier+".html")],
                    "output": os.path.join(output_dir, "report.html")
                }
            ]
            
            # Execute each step
            for step in steps:
                print(f"\nRunning {step['name']}...")
                try:
                    subprocess.run(step['cmd'], check=True)
                    print(f"{step['name']} completed successfully")
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running {step['name']}: {e}")
                    messagebox.showerror("Error", f"{step['name']} failed. See console for details.")
                    return
                except FileNotFoundError:
                    print(f"Error: {step['name']} not found.")
                    messagebox.showerror("Error", f"Script not found: {step['name']}")
                    return
                except Exception as e:
                    print(f"An unexpected error occurred in {step['name']}: {e}")
                    messagebox.showerror("Error", f"Unexpected error in {step['name']}")
                    return
            
            print("\nProcess Completed Successfully!")
            messagebox.showinfo("Success", f"Pipeline completed!\nResults saved to:\n{output_dir}")
        
        except Exception as e:
            print(f"\nPipeline initialization failed: {e}")
            messagebox.showerror("Error", f"Pipeline failed to initialize:\n{str(e)}")

    def run_batch(self):
        """Run the full pipeline on all images in the input directory with progress UI."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "No profile selected")
            return

        profile_name = self.tree.item(selection[0], "text")
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        input_dir = self.input_dir_entry.get()
        base_output_dir = self.output_dir_entry.get()

        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", f"Input directory not found:\n{input_dir}")
            return
        os.makedirs(base_output_dir, exist_ok=True)

        image_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ]
        if not image_files:
            messagebox.showwarning("Warning", "No image files found in input directory.")
            return

        # Load profile parameters
        try:
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            core_params = profile.get('core_parameters', {})
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load profile:\n{str(e)}")
            return

        color_spec = str(core_params.get("color_spec", 1))
        neighborhood = str(core_params.get("neighborhood", 7))
        sv1 = str(core_params.get("sv1", 0))
        sv2 = str(core_params.get("sv2", 0))
        sv3 = str(core_params.get("sv3", 0))
        sv4 = str(core_params.get("sv4", 0))
        sv5 = str(core_params.get("sv5", 0))
        excel_file = "tool.xlsx"

        # === Progress window ===
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Batch Processing")
        progress_win.geometry("400x120")
        progress_win.grab_set()

        tk.Label(progress_win, text="Batch Processing in Progress...").pack(pady=5)
        status_label = tk.Label(progress_win, text="", anchor="w")
        status_label.pack(fill=tk.X, padx=10)

        progress_bar = ttk.Progressbar(progress_win, length=350, mode="determinate", maximum=len(image_files))
        progress_bar.pack(pady=5)

        cancel_flag = {"stop": False}

        def cancel_batch():
            cancel_flag["stop"] = True
            status_label.config(text="Cancelling after current image...")
        tk.Button(progress_win, text="Cancel", command=cancel_batch).pack(pady=5)

        self.root.update()

        processed = 0
        errors = 0

        for idx, image_file in enumerate(image_files, start=1):
            if cancel_flag["stop"]:
                break

            input_image = os.path.join(input_dir, image_file)
            loc_specifier = os.path.splitext(image_file)[0]
            output_dir = os.path.join(base_output_dir, loc_specifier)
            os.makedirs(output_dir, exist_ok=True)

            status_label.config(text=f"Processing {image_file} ({idx}/{len(image_files)})")
            progress_bar["value"] = idx - 1
            self.root.update_idletasks()

            steps = [
                {
                    "name": "y_imaprocSec.py",
                    "cmd": ["python", "y_imaprocSec.py", excel_file, input_image, color_spec, neighborhood, sv1, sv2, sv3, sv4, sv5, loc_specifier,output_dir]
                },
                {
                    "name": "y_post_segSec.py",
                    "cmd": ["python", "y_post_segSec.py", os.path.join(output_dir, loc_specifier + ".pkl"), "operational_history.db", profile_path]
                },
                {
                    "name": "y_AnalyzeCellDataSec.py",
                    "cmd": ["python", "y_AnalyzeCellDataSec.py", os.path.join(output_dir, loc_specifier + ".pkl")]
                },
                {
                    "name": "y_Auto_cell_profileSec.py",
                    "cmd": ["python", "y_Auto_cell_profileSec.py", os.path.join(output_dir, loc_specifier + ".xlsx"), os.path.join(output_dir, loc_specifier + ".html")]
                }
            ]

            try:
                for step in steps:
                    print(f"Running {step['name']} for {image_file}...")
                    subprocess.run(step['cmd'], check=True)
                processed += 1
                print(f"✔ Completed: {image_file}")
            except subprocess.CalledProcessError as e:
                print(f"✘ Failed {image_file} on {step['name']}: {e}")
                errors += 1
            except Exception as e:
                print(f"✘ Unexpected error on {image_file}: {e}")
                errors += 1

            progress_bar["value"] = idx
            self.root.update_idletasks()

        progress_win.destroy()
        messagebox.showinfo(
            "Batch Complete",
            f"Processed {processed} images\n{errors} errors occurred.\nResults saved in:\n{base_output_dir}"
        )
        print(f"\nBatch completed. {processed} succeeded, {errors} failed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProfileManager(root)
    root.mainloop()