import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, scrolledtext

import ezdxf
import numpy as np
from ezdxf.math import Vec3


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & GLOBAL SETUP
# -----------------------------------------------------------------------------
def resource_path(relative_path):
    """
    Get the absolute path to a resource. This function is crucial for ensuring
    that data files (like icons) are found correctly, whether the script is
    running in development mode or as a bundled PyInstaller executable.
    """
    try:
        # PyInstaller creates a temporary folder and stores its path in `sys._MEIPASS`.
        base_path = sys._MEIPASS
    except AttributeError:
        # If `sys._MEIPASS` doesn't exist, the script is not running as a bundled app.
        # In this case, use the script's current directory as the base path.
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Defines the number of segments used to approximate a full circle for each quality level.
QUALITY_LEVELS = {"DRAFT": 40, "LOW": 80, "MEDIUM": 120, "HIGH": 240, "ULTRA": 480}


# -----------------------------------------------------------------------------
# CORE DXF PROCESSING LOGIC
# -----------------------------------------------------------------------------
def get_all_points(msp, tessellation_segments: int):
    """
    Iterates through all entities in a modelspace, tessellates (flattens) curves
    into a series of vertices, and yields each vertex as a Vec3 object.
    This function handles different DXF entity types.
    """
    for entity in msp:
        entity_type = entity.dxftype()
        try:
            if entity_type == "LINE":
                yield entity.dxf.start
                yield entity.dxf.end
            elif entity_type == "CIRCLE":
                # Approximate the circle with a series of straight line segments.
                center = entity.dxf.center
                radius = entity.dxf.radius
                angles = np.linspace(0, 2 * np.pi, tessellation_segments)
                for angle in angles:
                    yield Vec3(
                        center.x + radius * np.cos(angle),
                        center.y + radius * np.sin(angle),
                    )
            elif entity_type == "ARC":
                # Approximate the arc with line segments.
                center, radius = entity.dxf.center, entity.dxf.radius
                start_angle_rad, end_angle_rad = np.deg2rad(
                    entity.dxf.start_angle
                ), np.deg2rad(entity.dxf.end_angle)
                # Ensure the angle range is calculated correctly (e.g., from 350 to 10 degrees).
                if end_angle_rad < start_angle_rad:
                    end_angle_rad += 2 * np.pi

                arc_angle_span = abs(end_angle_rad - start_angle_rad)
                num_segments = max(
                    2, int(tessellation_segments * arc_angle_span / (2 * np.pi))
                )
                angles = np.linspace(start_angle_rad, end_angle_rad, num_segments)
                for angle in angles:
                    yield Vec3(
                        center.x + radius * np.cos(angle),
                        center.y + radius * np.sin(angle),
                    )
            elif entity_type in ["POLYLINE", "LWPOLYLINE"]:
                # This 'with' block is required for compatibility with very old ezdxf versions
                # that return a context manager instead of a direct iterator.
                with entity.points() as points_iterator:
                    for p in points_iterator:
                        # This explicit conversion is needed for old ezdxf versions where the
                        # iterator yields tuples (x, y, bulge...) instead of Vec3 objects.
                        # We only take the first two elements (x, y) to create the Vec3.
                        yield Vec3(p[0], p[1])
        except (AttributeError, ValueError, TypeError):
            # Gracefully skip any entities that are unsupported or have errors.
            continue


def find_drawing_properties(msp, tessellation_segments: int):
    """
    Finds the geometric center and the maximum radius of the drawing.
    The center is calculated as the center of the drawing's bounding box.
    """
    print("Analyzing drawing properties...")
    all_points = list(get_all_points(msp, tessellation_segments))
    if not all_points:
        return Vec3(0, 0, 0), 0.0

    # Step 1: Get the bounding box of all points.
    min_x, max_x = min(p.x for p in all_points), max(p.x for p in all_points)
    min_y, max_y = min(p.y for p in all_points), max(p.y for p in all_points)

    # Step 2: Calculate the center of the bounding box.
    center = Vec3(min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2)
    print(f"  - Auto-detected center: ({center.x:.2f}, {center.y:.2f})")

    # Step 3: Find the maximum distance from the center to any point in the drawing.
    max_radius = max((p - center).magnitude for p in all_points)
    print(f"  - Auto-detected source radius: {max_radius:.2f} units (assuming mm)")
    return center, max_radius


def transform_point(
    point: Vec3, center: Vec3, scale_factor: float, target_radius: float
) -> Vec3:
    """
    Transforms a single point from a polar-like coordinate system (on the disc)
    to a Cartesian coordinate system (on the unrolled rectangle).
    This function is the mathematical core of the unrolling logic.
    """
    # Get the point's position relative to the center of the disc.
    relative_vec = point - center

    # Calculate polar coordinates: radius (distance) and angle.
    radius = relative_vec.magnitude
    angle = np.arctan2(relative_vec.y, relative_vec.x)

    # Normalize the angle to the range [0, 2*pi] to avoid negative X coordinates.
    if angle < 0:
        angle += 2 * np.pi

    # Calculate the new Cartesian coordinates for the unrolled plane.
    # The new Y-coordinate is the scaled radial distance.
    new_y = radius * scale_factor
    # The new X-coordinate is the arc length at the target radius (angle in radians * radius).
    new_x = angle * target_radius

    return Vec3(new_x, new_y, 0)


def run_unroll_process(input_file, quality, target_radius):
    """
    Orchestrates the entire DXF unrolling process from file loading to saving.
    This function is designed to be run in a separate thread to keep the GUI responsive.
    """
    try:
        # --- 1. Setup ---
        input_path = Path(input_file)
        # Generate the output filename automatically (e.g., "input_unrolled.dxf").
        output_file = input_path.with_name(
            f"{input_path.stem}_unrolled{input_path.suffix}"
        )
        tessellation_segments = QUALITY_LEVELS.get(
            quality.upper(), QUALITY_LEVELS["MEDIUM"]
        )

        # --- 2. Loading ---
        print(f"Loading DXF file: {input_path.name}")
        in_doc = ezdxf.readfile(input_path)
        msp_in = in_doc.modelspace()

        # --- 3. Analysis ---
        # The process is hardcoded to always auto-detect drawing properties.
        disk_center, source_radius = find_drawing_properties(
            msp_in, tessellation_segments
        )
        if source_radius == 0:
            print("Error: Source radius is 0. Cannot perform scaling.")
            return

        scale_factor = target_radius / source_radius
        print(
            f"Scaling to new radius {target_radius} mm with factor {scale_factor:.4f}"
        )

        # --- 4. Transformation ---
        out_doc = ezdxf.new()
        msp_out = out_doc.modelspace()

        # Set the units of the output DXF file to millimeters.
        # 4 is the standard DXF code for millimeters.
        out_doc.header["$INSUNITS"] = 4
        print("  - Output DXF units set to millimeters.")

        print("Processing and transforming entities...")
        processed_count = 0
        for entity in msp_in:
            points = list(get_all_points([entity], tessellation_segments))
            if points:
                is_closed = getattr(entity.dxf, "flags", 0) & 1
                transformed_points = [
                    transform_point(p, disk_center, scale_factor, target_radius)
                    for p in points
                ]
                if len(transformed_points) >= 2:
                    msp_out.add_lwpolyline(transformed_points, close=is_closed)
                    processed_count += 1
        print(f"Successfully processed {processed_count} entities.")

        # --- 5. Border Creation ---
        # The process is hardcoded to always create a smart border along the geometry edges.
        print("Closing contours along geometry edges...")
        left_points, right_points = [], []
        width = 2 * np.pi * target_radius
        tolerance = 1e-6  # Tolerance for floating point comparisons.

        for entity in msp_out:
            if entity.dxftype() == "LWPOLYLINE":
                with entity.points() as points_iterator:
                    vertices = [Vec3(p[0], p[1]) for p in points_iterator]
                if not vertices:
                    continue

                first_pt, last_pt = vertices[0], vertices[-1]

                # Check if the start/end points of the polyline lie on the left or right boundary.
                if abs(first_pt.x) < tolerance:
                    left_points.append(first_pt)
                if abs(first_pt.x - width) < tolerance:
                    right_points.append(first_pt)

                if not entity.is_closed:
                    if abs(last_pt.x) < tolerance:
                        left_points.append(last_pt)
                    if abs(last_pt.x - width) < tolerance:
                        right_points.append(last_pt)

        # Sort the boundary points by their y-coordinate before connecting them into a line.
        left_points.sort(key=lambda p: p.y)
        right_points.sort(key=lambda p: p.y)

        if len(left_points) >= 2:
            msp_out.add_lwpolyline(left_points)
            print(f"  - Added left border connecting {len(left_points)} points.")
        if len(right_points) >= 2:
            msp_out.add_lwpolyline(right_points)
            print(f"  - Added right border connecting {len(right_points)} points.")

        # --- 6. Saving ---
        out_doc.saveas(output_file)
        print("-" * 40)
        print(f"🎉 SUCCESS! Unrolled drawing saved to:\n{output_file}")
        print("-" * 40)
    except Exception as e:
        # Catch any unexpected errors and display them in the GUI console.
        print("\n" + "=" * 40)
        print(f"AN ERROR OCCURRED:\n{e}")
        print("=" * 40)


# -----------------------------------------------------------------------------
# GRAPHICAL USER INTERFACE (GUI)
# -----------------------------------------------------------------------------
class UnrollerApp:
    """Main application class for the Tkinter GUI."""

    def __init__(self, root):
        self.root = root
        self.root.title("DXF Unroller")

        # Set the window icon using the resource_path helper function.
        try:
            icon_path = resource_path("icon.png")
            icon_image = tk.PhotoImage(file=icon_path)
            self.root.iconphoto(True, icon_image)
        except Exception as e:
            print(f"Warning: Could not load window icon. {e}")

        self.root.geometry("500x450")

        # --- GUI Layout ---
        # The layout is organized into frames for better structure.
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame for file selection widgets.
        file_frame = ttk.LabelFrame(main_frame, text="Input File", padding="10")
        file_frame.pack(fill=tk.X, expand=True)
        self.filepath_var = tk.StringVar()
        self.filepath_entry = ttk.Entry(
            file_frame, textvariable=self.filepath_var, state="readonly", width=50
        )
        self.filepath_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.browse_button = ttk.Button(
            file_frame, text="Browse...", command=self.browse_file
        )
        self.browse_button.pack(side=tk.LEFT)

        # Frame for settings widgets (quality and radius).
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, expand=True, pady=10)
        ttk.Label(settings_frame, text="Quality:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.quality_var = tk.StringVar(value="MEDIUM")
        self.quality_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.quality_var,
            values=list(QUALITY_LEVELS.keys()),
            state="readonly",
        )
        self.quality_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(settings_frame, text="Target Outer Radius (mm):").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.radius_var = tk.StringVar(value="100.0")
        self.radius_entry = ttk.Entry(settings_frame, textvariable=self.radius_var)
        self.radius_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        settings_frame.columnconfigure(1, weight=1)

        # Main action button.
        self.run_button = ttk.Button(
            main_frame,
            text="Unroll DXF",
            command=self.start_unrolling,
            style="Accent.TButton",
        )
        self.run_button.pack(fill=tk.X, expand=True, pady=10)
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        # Frame for the status console output.
        console_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True)
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=10)
        self.console.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        """Callback function for the 'Browse...' button. Opens a file dialog."""
        filepath = filedialog.askopenfilename(
            title="Select a DXF File",
            filetypes=[("DXF Files", "*.dxf"), ("All files", "*.*")],
        )
        if filepath:
            self.filepath_var.set(filepath)

    def start_unrolling(self):
        """
        Callback for the 'Unroll DXF' button. It validates user input and
        starts the unrolling process in a separate thread.
        """
        input_file = self.filepath_var.get()
        quality = self.quality_var.get()
        try:
            target_radius = float(self.radius_var.get())
        except ValueError:
            self.log_to_console("Error: 'Target Outer Radius' must be a valid number.")
            return
        if not input_file:
            self.log_to_console("Error: Please select an input DXF file.")
            return

        # Disable the button to prevent the user from running the process multiple times.
        self.run_button.config(state="disabled")
        self.console.delete("1.0", tk.END)
        self.log_to_console("Starting process...")

        # Run the core logic in a thread to keep the GUI from freezing.
        thread = threading.Thread(
            target=self.run_process_in_thread, args=(input_file, quality, target_radius)
        )
        thread.start()

    def run_process_in_thread(self, input_file, quality, target_radius):
        """
        Wrapper function that runs in the new thread. It redirects stdout to
        the GUI console and re-enables the run button upon completion.
        """
        old_stdout = sys.stdout
        sys.stdout = self.Redirector(self.log_to_console)

        run_unroll_process(input_file, quality, target_radius)

        sys.stdout = old_stdout
        # Safely schedule the GUI update from the main thread.
        self.root.after(0, self.run_button.config, {"state": "normal"})

    def log_to_console(self, text):
        """
        Schedules a text insertion into the console widget. Using root.after()
        makes this method thread-safe, allowing it to be called from any thread.
        """
        self.root.after(0, self._insert_text, text)

    def _insert_text(self, text):
        """Internal method that performs the actual text insertion in the GUI thread."""
        self.console.insert(tk.END, str(text) + "\n")
        self.console.see(tk.END)

    class Redirector:
        """
        A helper class to redirect a stream (like sys.stdout) to a callback function.
        This allows capturing `print()` statements and showing them in the GUI.
        """

        def __init__(self, callback):
            self.callback = callback
            self.buffer = ""

        def write(self, text):
            self.buffer += text
            if "\n" in self.buffer:
                lines = self.buffer.split("\n")
                for line in lines[:-1]:
                    self.callback(line)
                self.buffer = lines[-1]

        def flush(self):
            # Flushes any remaining text in the buffer.
            if self.buffer:
                self.callback(self.buffer)
                self.buffer = ""


if __name__ == "__main__":
    # Create the main Tkinter window and the application instance.
    root = tk.Tk()
    app = UnrollerApp(root)
    # Start the Tkinter event loop, which waits for user actions.
    root.mainloop()
