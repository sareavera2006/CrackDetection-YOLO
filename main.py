import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from crackDetection import CrackAssessment
from crackMeasurement import CrackAnalysis


class CrackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("A YOLOv8-Based Computer Vision System for Automated Crack Detection and Dimension Measurement")
        self.root.geometry("1150x780")
        self.root.minsize(950, 680)

        # Initialize models 
        try:
            self.detector = CrackAssessment()
            self.analyzer = CrackAnalysis()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to load detection models:\n{e}")

        self.current_image_path = None
        self.original_cv_img = None
        self.output_visualization = None

        # Build UI layout
        self._create_widgets()

    def _create_widgets(self):
        # ------------------ Sidebar (Controls & Results) ------------------ #
        sidebar = ttk.Frame(self.root, padding="15")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Control Section
        ttk.Label(sidebar, text="Controls", font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        self.btn_browse = ttk.Button(sidebar, text="Select Image", command=self.load_image)
        self.btn_browse.pack(fill=tk.X, pady=4)

        self.btn_analyze = ttk.Button(sidebar, text="Run Analysis", command=self.run_analysis, state=tk.DISABLED)
        self.btn_analyze.pack(fill=tk.X, pady=4)

        self.btn_save = ttk.Button(sidebar, text="Save Overlay Image", command=self.save_visualization, state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=4)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=12)

        # ------------------ Classification Assessment Section ------------------ #
        ttk.Label(sidebar, text="Assessment", font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0, 8))
        
        # Color-Coded Status Badge
        ttk.Label(sidebar, text="Status:", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(0, 2))
        
        badge_frame = ttk.Frame(sidebar, width=220, height=28)
        badge_frame.pack_propagate(False)
        badge_frame.pack(anchor="w", padx=6, pady=(0, 8))

        self.lbl_badge = tk.Label(
            badge_frame, 
            text="---", 
            font=("Helvetica", 10, "bold"), 
            bg="#7f8c8d", 
            fg="white", 
            relief="flat"
        )
        self.lbl_badge.pack(fill=tk.BOTH, expand=True)

        # ---------------- Custom Canvas Gauge Meter ---------------- #
        self.canvas_gauge = tk.Canvas(sidebar, width=232, height=60, highlightthickness=0)
        self.canvas_gauge.pack(anchor="w", pady=(0, 8))
        self._draw_confidence_gauge(None)

        # Raw Probabilities
        self.lbl_prob_crack = ttk.Label(sidebar, text="  • Crack Probability: --", font=("Helvetica", 9))
        self.lbl_prob_crack.pack(anchor="w", pady=1)

        self.lbl_prob_noncrack = ttk.Label(sidebar, text="  • Non-Crack Probability: --", font=("Helvetica", 9))
        self.lbl_prob_noncrack.pack(anchor="w", pady=1)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=12)

        # ------------------ Measurement Results Section ------------------ #
        ttk.Label(sidebar, text="Measurements", font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0, 8))

        self.lbl_area = ttk.Label(sidebar, text="Crack Area: -- px²", font=("Helvetica", 10))
        self.lbl_area.pack(anchor="w", pady=2)

        self.lbl_length = ttk.Label(sidebar, text="Crack Length: -- px", font=("Helvetica", 10))
        self.lbl_length.pack(anchor="w", pady=2)

        self.lbl_max_width = ttk.Label(sidebar, text="Max Width: -- px", font=("Helvetica", 10))
        self.lbl_max_width.pack(anchor="w", pady=2)

        self.lbl_avg_width = ttk.Label(sidebar, text="Average Width: -- px", font=("Helvetica", 10))
        self.lbl_avg_width.pack(anchor="w", pady=2)

        # ------------------ Side-by-Side Image Display Panel ------------------ #
        display_frame = ttk.Frame(self.root, padding="10")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Left Container: Original Image
        frame_orig = ttk.LabelFrame(display_frame, text=" Original Image ", padding="2")
        frame_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_img_orig = ttk.Label(frame_orig, text="No image selected", anchor="center")
        self.lbl_img_orig.pack(fill=tk.BOTH, expand=True)

        # Right Container: Analysis Overlay
        frame_proc = ttk.LabelFrame(display_frame, text=" Analysis Overlay ", padding="2")
        frame_proc.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_img_proc = ttk.Label(frame_proc, text="Run analysis to generate overlay", anchor="center")
        self.lbl_img_proc.pack(fill=tk.BOTH, expand=True)

        # ------------------ Analysis Overlay Legend (Inside Right Container) ------------------ #
        self.legend_frame = ttk.Frame(frame_proc)
        
        ttk.Label(self.legend_frame, text="Legend:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 8))

        # Blue Swatch
        lbl_blue_swatch = tk.Label(self.legend_frame, bg="#0066FF", width=2, height=1, relief="solid", bd=1)
        lbl_blue_swatch.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(self.legend_frame, text="Segmented Crack", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 12))

        # Red Swatch
        lbl_red_swatch = tk.Label(self.legend_frame, bg="#FF0000", width=2, height=1, relief="solid", bd=1)
        lbl_red_swatch.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(self.legend_frame, text="Skeleton Path", font=("Helvetica", 9)).pack(side=tk.LEFT)

        # Keep hidden on startup
        self.legend_frame.pack_forget()

    def _draw_confidence_gauge(self, score=None):
        """Draws a 5-stage color spectrum bar with a dynamic triangle pointer on Canvas."""
        self.canvas_gauge.delete("all")

        w, h = 232, 60
        x1, y1 = 6, 18
        x2, y2 = 226, 36 
        bar_width = x2 - x1

        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1e8449"]
        seg_w = bar_width / 5

        self.canvas_gauge.create_text(w / 2, 8, text="CONFIDENCE", font=("Helvetica", 9, "bold"), fill="#2c3e50")

        # Draw 5-Segment Bar
        for i, col in enumerate(colors):
            sx1 = x1 + i * seg_w
            sx2 = x1 + (i + 1) * seg_w
            self.canvas_gauge.create_rectangle(sx1, y1, sx2, y2, fill=col, outline="")

        self.canvas_gauge.create_rectangle(x1, y1, x2, y2, outline="#7f8c8d", width=1)

        # Inner Text Labels
        self.canvas_gauge.create_text(x1 + 14, (y1 + y2) / 2, text="LOW", font=("Helvetica", 7, "bold"), fill="white")
        self.canvas_gauge.create_text(x2 - 16, (y1 + y2) / 2, text="HIGH", font=("Helvetica", 7, "bold"), fill="white")

        # Triangle Pointer & Percentage Display
        if score is not None:
            clamped_score = max(0.0, min(100.0, float(score)))
            px = x1 + (clamped_score / 100.0) * bar_width

            # Black Upward-Pointing Triangle
            self.canvas_gauge.create_polygon(
                px, y2 + 2,
                px - 6, y2 + 11,
                px + 6, y2 + 11,
                fill="#2c3e50", outline=""
            )
            # Numeric Score Below Pointer
            self.canvas_gauge.create_text(px, y2 + 18, text=f"{clamped_score:.1f}%", font=("Helvetica", 8, "bold"), fill="#2c3e50")
        else:
            self.canvas_gauge.create_text(w / 2, y2 + 15, text="--%", font=("Helvetica", 8, "bold"), fill="#7f8c8d")

    def load_image(self):
        file_types = [("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        path = filedialog.askopenfilename(title="Select Target Image", filetypes=file_types)

        if not path:
            return

        self.current_image_path = path
        self.original_cv_img = cv2.imread(path)
        self.output_visualization = None

        # Render original image to left canvas panel
        self._render_cv_image_to_label(self.original_cv_img, self.lbl_img_orig)

        # Clear overlay preview panel & reset status indicators
        self.lbl_img_proc.config(image="", text="Run analysis to generate overlay")
        self._reset_labels()

        # Enable analysis button
        self.btn_analyze.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)

    def run_analysis(self):
        if not self.current_image_path:
            return

        # Classification
        assessment_res = self.detector.assessImage(self.current_image_path)
        class_label = assessment_res["assessment"]
        confidence = assessment_res["confidence_percent"]

        # Update Gauge Meter with calculated confidence
        self._draw_confidence_gauge(confidence)

        # Update Raw Probabilities Display
        raw_probs = assessment_res.get("raw_probabilities", {})
        self._update_probability_labels(raw_probs)

        # Branching based on assessment
        if class_label == "NONCRACKED":
            # Green Badge for Non-Cracked
            self.lbl_badge.config(text="NO CRACK DETECTED", bg="#27ae60")
            
            self._reset_measurement_labels()
            self.output_visualization = None
            
            # Reset right canvas frame & hide legend
            self.lbl_img_proc.config(image="", text="No crack detected within image.")
            self.legend_frame.pack_forget()
            self.btn_save.config(state=tk.DISABLED)

        elif class_label == "CRACKED":
            # Red Badge for Cracked
            self.lbl_badge.config(text="⚠ CRACK DETECTED", bg="#e74c3c")

            # Run segmentation & measurement
            measurements = self.analyzer.analyzeCrackPixels(
                self.current_image_path, 
                threshold_method='otsu'
            )

            # Update measurement metrics
            self.lbl_area.config(text=f"Crack Area: {measurements['crack_area_pixels']} px²")
            self.lbl_length.config(text=f"Crack Length: {measurements['crack_length_pixels']} px")
            self.lbl_max_width.config(text=f"Max Width: {measurements['max_width_pixels']} px")
            self.lbl_avg_width.config(text=f"Average Width: {measurements['avg_width_pixels']} px")

            # Store and display overlay image on right canvas panel
            self.output_visualization = measurements["visualization"]
            self._render_cv_image_to_label(self.output_visualization, self.lbl_img_proc)

            # Reveal legend at bottom of right container
            self.legend_frame.pack(side=tk.BOTTOM, pady=(8, 0))

            self.btn_save.config(state=tk.NORMAL)

    def _update_probability_labels(self, raw_probs):
        """Helper to parse and format raw probabilities onto GUI labels."""
        crack_prob_str = "--"
        noncrack_prob_str = "--"

        for class_name, prob_val in raw_probs.items():
            formatted_val = f"{prob_val * 100:.2f}%" if isinstance(prob_val, float) else str(prob_val)
            if "NON" in class_name.upper():
                noncrack_prob_str = formatted_val
            else:
                crack_prob_str = formatted_val

        self.lbl_prob_crack.config(text=f"  • Crack Probability: {crack_prob_str}")
        self.lbl_prob_noncrack.config(text=f"  • Non-Crack Probability: {noncrack_prob_str}")

    def _render_cv_image_to_label(self, cv_img, label_widget):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        target_w, target_h = 500, 600

        # Calculate scaling ratio (scales UP small images, scales DOWN large images)
        img_w, img_h = pil_img.size
        scale = min(target_w / img_w, target_h / img_h)

        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized_img)

        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img

    def save_visualization(self):
        if self.output_visualization is None:
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")]
        )
        if save_path:
            cv2.imwrite(save_path, self.output_visualization)
            messagebox.showinfo("Saved", f"Overlay saved successfully to:\n{save_path}")

    def _reset_measurement_labels(self):
        self.lbl_area.config(text="Crack Area: -- px²")
        self.lbl_length.config(text="Crack Length: -- px")
        self.lbl_max_width.config(text="Max Width: -- px")
        self.lbl_avg_width.config(text="Average Width: -- px")

    def _reset_labels(self):
        self.lbl_badge.config(text="---", bg="#7f8c8d")
        self._draw_confidence_gauge(None)
        self.lbl_prob_crack.config(text="  • Crack Probability: --")
        self.lbl_prob_noncrack.config(text="  • Non-Crack Probability: --")
        self._reset_measurement_labels()
        self.legend_frame.pack_forget()


def main():
    root = tk.Tk()
    app = CrackApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()