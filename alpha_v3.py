import csv
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import sys
from Grain import Grano
from tooltips import ToolTip


class GrainDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Declaración de la lista de granos vacía
        self.grain_data = []
        self.iconbitmap(r".\iconos\icons8-granos-16.ico")
        self.title('Grain Detector')
        self.state('zoomed')  # This will maximize the window

        # Menu bar
        self.menu_bar = tk.Menu(self)

        # File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open", command=self.open_image)
        self.file_menu.add_command(label="Exit", command=self.exit_program)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=self.show_about_dialog)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Show State", command=self.show_state_dialog)

        self.config(menu=self.menu_bar)

        # Control panel
        self.controls = tk.Frame(self, width=200)
        self.controls.grid(row=0, column=0, sticky="ns", rowspan=1000)

        # Longitud Mínima
        self.length_frame = tk.Frame(self.controls)
        self.length_frame.pack(pady=5)
        self.length_label = tk.Label(self.length_frame, text="Longitud Mínima")
        self.length_label.grid(row=0, column=0)
        self.length_entry = tk.Entry(self.length_frame, width=5)
        self.length_entry.insert(0, "10")
        self.length_entry.grid(row=0, column=1)

        # self.length_entry.trace_add("write", self.statusbar.config(text=str(self)))
        # self.length_entry.bind('<Return>', self.statusbar.config(text=str(self)))

        # Grosor Mínimo
        self.width_frame = tk.Frame(self.controls)
        self.width_frame.pack(pady=5)
        self.width_label = tk.Label(self.width_frame, text="Grosor Mínimo")
        self.width_label.grid(row=0, column=0)
        self.width_entry = tk.Entry(self.width_frame, width=5)
        self.width_entry.insert(0, "10")
        self.width_entry.grid(row=0, column=1)

        # Umbral
        self.threshold_frame = tk.Frame(self.controls)
        self.threshold_frame.pack(pady=5)
        self.threshold_label = tk.Label(self.threshold_frame, text="Umbral")
        self.threshold_label.grid(row=0, column=0)

        self.threshold_value_var = tk.StringVar()
        self.threshold_value_var.set("128")
        self.threshold_entry = tk.Entry(self.threshold_frame, width=5, textvariable=self.threshold_value_var)
        self.threshold_entry.grid(row=0, column=1)

        self.threshold_slider = tk.Scale(self.controls, from_=0, to=255, orient=tk.HORIZONTAL,
                                         command=self.update_threshold_entry)
        self.threshold_slider.set(128)
        self.threshold_slider.pack()
        self.threshold_entry.bind('<Return>', self.update_threshold_slider)

        # Detectar granos
        self.detect_grains_button = tk.Button(self.controls, text="Detectar granos", command=self.detect_grains)
        self.detect_grains_button.pack()

        # Focus
        self.detect_icon = ImageTk.PhotoImage(Image.open(r".\iconos\icons8-trazo-rectangular-30.png"))
        self.detect_button = tk.Button(self.controls, text="Focus", command=self.toggle_focus_mode,
                                       image=self.detect_icon, height=50, width=50)
        self.detect_button.pack()
        self.focus_button_toolTip = ToolTip(self.detect_button,
                                            "Actualice el área donde quiera que se detecten los granos")

        # Measure
        self.measure_icon = ImageTk.PhotoImage(Image.open(r".\iconos\icons8-regla-16.png"))
        self.measure_button = tk.Button(self.controls, text="Measure", command=self.toggle_measure_mode,
                                        image=self.measure_icon, height=50, width=50)
        self.measure_button.pack()
        self.measure_button_toolTip = ToolTip(self.measure_button,
                                              "Medir la regla para conocer las dimensiones reales de la imagen")

        self.zoom_factor = 1  # Inicialización del factor de zoom

        self.zoom_frame = tk.Frame(self.controls)
        self.zoom_frame.pack(pady=5)  # Espacio vertical para separar de otros widgets

        self.zoom_in_icon = ImageTk.PhotoImage(Image.open(r".\iconos\icons8-acercar-48.png"))
        self.zoom_in_button = tk.Button(self.zoom_frame, text="Zoom in", command=self.zoom_in, image=self.zoom_in_icon,
                                        height=40, width=40)  # Ajuste de dimensiones
        self.zoom_in_button.pack(side='left', padx=5)  # Alineación a la izquierda con un poco de espacio horizontal
        self.zoom_in_button_toolTip = ToolTip(self.zoom_in_button, "Acercar Zoom")

        self.zoom_out_icon = ImageTk.PhotoImage(Image.open(r".\iconos\icons8-alejar-48.png"))
        self.zoom_out_button = tk.Button(self.zoom_frame, text="Zoom out", command=self.zoom_out,
                                         image=self.zoom_out_icon, height=40, width=40)  # Ajuste de dimensiones
        self.zoom_out_button.pack(side='left', padx=5)  # Alineación a la izquierda con un poco de espacio horizontal
        self.zoom_out_button_toolTip = ToolTip(self.zoom_out_button, "Alejar Zoom")

        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=1, sticky="nsew", rowspan=1000)

        # Configure the stretching properties
        self.grid_columnconfigure(0, weight=0)  # Control panel column doesn't stretch
        self.grid_columnconfigure(1, weight=1)  # Canvas column stretches
        self.grid_rowconfigure(0, weight=1)  # First row stretches to fill vertical space

        self.filename = None

        # Barra de estado
        self.statusbar = tk.Label(self, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                  height=2)  # anchor W hace que el texto quede a la izquierda
        self.statusbar.grid(row=1001, column=0, columnspan=2, sticky="ew")

        # Bindeo de eventos al canvas
        self.canvas.bind("<ButtonPress-1>", self.start_action)
        self.canvas.bind("<B1-Motion>", self.update_action)
        self.canvas.bind("<ButtonRelease-1>", self.end_action)

        # Inicialización del ROI
        self.start_x = None
        self.start_y = None
        self.current_rectangle = None
        self.selection_rectangle = None
        self.focus_mode = False

        self.image_dimensions_label = tk.Label(self.controls)
        self.image_dimensions_label.pack()
        self.image = None  # To store the current image

        self.reset_button = tk.Button(self.controls, text="Reset", command=self.reset)
        self.reset_button.pack()

        self.distance_label = tk.Label(self.controls)
        self.distance_label.pack()

        # Contador de semillas
        self.grain_count_label = tk.Label(self.controls, text="Semillas detectadas: 0")
        self.grain_count_label.pack()

        # Boton Analisis
        self.perform_analysis_button = tk.Button(self.controls, text="Análisis", command=self.perform_analysis)
        self.perform_analysis_button.pack(pady=10)  # Espacio vertical para separar de otros widgets

        self.measure_mode = False  # New variable to track the measure mode
        self.start_line_point = None  # New variable to store the start point of the line
        self.end_line_point = None  # New variable to store the end point of the line
        self.current_line = None  # New variable to store the current line drawn on the canvas
        self.granos = 0

        self.statusbar.config(text=str(self))

    # --------------------AQUI EMPIEZA LAS DEFINICIONES DE LAS FUNCIONES DE LA APLICACIÓN--------------------#

    def update_threshold_entry(self, value):
        self.threshold_value_var.set(value)
        self.statusbar.config(text=str(self))

    def update_threshold_slider(self, event):
        try:
            value = int(self.threshold_value_var.get())
            if 0 <= value <= 255:
                self.threshold_slider.set(value)
                self.statusbar.config(text=str(self))
        except ValueError:
            pass

    def show_state_dialog(self):
        state_win = tk.Toplevel(self)
        state_win.title("State Information")
        state_win.geometry("300x400")  # puedes ajustar esto según tus necesidades

        details = [
            f"Title: {self.title()}",
            f"State: {self.state()}",
            f"File: {self.filename if self.filename else 'No file loaded'}",
            f"Zoom Factor: {self.zoom_factor}",
            f"Focus Mode: {'On' if self.focus_mode else 'Off'}",
            f"Measure Mode: {'On' if self.measure_mode else 'Off'}",
            f"Threshold: {self.threshold_slider.get()}",
            f"Length: {self.length_entry.get()}",
            f"Width: {self.width_entry.get()}"
        ]

        for detail in details:
            label = tk.Label(state_win, text=detail)
            label.pack(pady=10)  # pequeño espacio entre las líneas

        close_button = tk.Button(state_win, text="Close", command=state_win.destroy)
        close_button.pack(pady=20)

    def __str__(self):
        details = [
            f"Title: {self.title()}",
            # f"State: {self.state()}",
            f"File: {self.filename if self.filename else 'No file loaded'}",
            # f"Zoom Factor: {self.zoom_factor}",
            # f"Focus Mode: {'On' if self.focus_mode else 'Off'}",
            # f"Measure Mode: {'On' if self.measure_mode else 'Off'}",
            f"Threshold: {self.threshold_slider.get()}",
            f"Length: {self.length_entry.get()}",
            f"Width: {self.width_entry.get()}",
            f"Selection Rectangle: {self.selection_rectangle}",
        ]
        return ' | '.join(details)

    def start_action(self, event):
        if self.focus_mode:
            self.start_rectangle(event)
        elif self.measure_mode:
            self.start_line(event)

    def update_action(self, event):
        if self.focus_mode:
            self.update_rectangle(event)
        if self.measure_mode:
            self.update_line(event)

    def update_line(self, event):
        if not self.measure_mode or self.current_line is None:
            return

        self.end_line_point = (event.x, event.y)
        self.canvas.coords(self.current_line, self.start_line_point[0], self.start_line_point[1],
                           self.end_line_point[0], self.end_line_point[1])

    def end_action(self, event):
        if self.focus_mode:
            self.end_rectangle(event)
        elif self.measure_mode:
            self.end_line(event)

    def start_line(self, event):
        if not self.measure_mode:
            return

        self.start_line_point = (event.x, event.y)
        self.current_line = self.canvas.create_line(
            self.start_line_point[0], self.start_line_point[1], self.start_line_point[0], self.start_line_point[1],
            fill='blue'
        )

    def end_line(self, event):
        if not self.measure_mode or self.current_line is None:
            return

        # Calculate the distance in pixels and show it to the user
        distance_pixels = math.sqrt((self.end_line_point[0] - self.start_line_point[0]) ** 2 + (
                self.end_line_point[1] - self.start_line_point[1]) ** 2)
        distance_cm = distance_pixels / 10  # Cuantos pixeles hay en un centimetro

        self.distance_label.config(text=f'Distancia: {distance_cm:.2f} px/cm')  # Update the label text

        self.measure_mode = False
        Grano.px_per_cm = distance_cm  # Pasarle al atributo de la clase Grano cuantos píxeles tiene un centimetro

    def toggle_measure_mode(self):
        self.measure_mode = not self.measure_mode
        if self.measure_mode:
            self.focus_mode = False

    def reset(self):
        # Limpia el canvas
        self.canvas.delete("all")
        # Reinicia todas las variables a su estado inicial
        self.start_x = None
        self.start_y = None
        self.current_rectangle = None
        self.selection_rectangle = None
        self.focus_mode = False
        self.zoom_factor = 1
        self.image_dimensions_label.config(text="")
        self.image = None
        self.length_entry.delete(10, tk.END)
        self.width_entry.delete(10, tk.END)
        self.threshold_slider.set(128)
        self.distance_label.config(text=f'Distancia:')
        self.grain_count_label.config(text=f"Semillas detectadas: 0")
        self.statusbar.config(text=str(self))
        self.statusbar.config(text=str(self))

    def start_rectangle(self, event):
        if not self.focus_mode:
            return

        self.start_x = event.x
        self.start_y = event.y
        self.current_rectangle = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline='red'
        )

    def update_rectangle(self, event):
        if not self.focus_mode or self.current_rectangle is None:
            return

        self.canvas.coords(self.current_rectangle, self.start_x, self.start_y, event.x, event.y)

    def end_rectangle(self, event):
        if not self.focus_mode or self.current_rectangle is None:
            return

        self.selection_rectangle = self.canvas.bbox(self.current_rectangle)
        self.focus_mode = False

    def toggle_focus_mode(self):
        self.focus_mode = not self.focus_mode
        if self.focus_mode:
            self.measure_mode = False

    def zoom_in(self):
        self.zoom_factor *= 1.25  # Aumenta el zoom en un 25%
        self.redraw_image()

    def zoom_out(self):
        self.zoom_factor /= 1.25  # Disminuye el zoom en un 25%
        self.redraw_image()

    def redraw_image(self):
        if self.image is not None:
            height, width, _ = self.image.shape
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)
            resized_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            tk_img = ImageTk.PhotoImage(Image.fromarray(image))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            self.canvas.image = tk_img

    def open_image(self, filename=None):
        if filename is None:
            self.filename = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.filename:
            self.image = cv2.imread(self.filename)
            # Resize and center the image
            self.image = self.resize_image(self.image)
            # Update image dimensions label
            height, width, _ = self.image.shape
            self.image_dimensions_label.config(text=f"Tamaño: {width}x{height} px")
            # Convert the image to RGB for displaying in Tkinter (OpenCV uses BGR)
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # Convert the processed image to a format compatible with Tkinter
            tk_img = ImageTk.PhotoImage(Image.fromarray(image))
            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            self.canvas.image = tk_img
            self.statusbar.config(text=str(self))
        else:
            self.image = None

    def show_about_dialog(self):
        about_text = "Este programa está hecho por Alejandro Villalba Fernández y dirigido por el profesor Sergio Gálvez. \n Trabajo Fin de Grado Curso 2022/2023\n Grado de Ingeniería de Computadores.\n Universidad de Málaga"

        dialog = tk.Toplevel()
        dialog.title("Acerca de")
        dialog.geometry("980x260")

        label = tk.Label(dialog, text=about_text)
        label.pack(padx=20, pady=20)

        image_frame = tk.Frame(dialog)
        image_frame.pack()

        # Cargamos la imagen
        image_paths = [r".\iconos\Alejandro Villalba.jpeg", r".\iconos\iconoUMA.png"]

        for image_path in image_paths:
            # Load the image
            image = Image.open(image_path)

            # Calculate aspect ratio
            aspect_ratio = image.width / image.height

            # Fixed width
            new_width = 150
            new_height = int(new_width / aspect_ratio)

            # If the new height exceeds 122, fix the height and calculate the width
            if new_height > 122:
                new_height = 122
                new_width = int(new_height * aspect_ratio)

            # Resize the image
            image = image.resize((new_width, new_height), Image.LANCZOS)

            # Convert the image to a format Tkinter can use
            photo = ImageTk.PhotoImage(image)

            # Create a label for the image
            image_label = tk.Label(image_frame, image=photo)
            image_label.image = photo  # Keep a reference to the image so it's not garbage collected
            image_label.pack(side=tk.LEFT)

    def grain_detector(self, image):
        min_length = int(self.length_entry.get())
        min_width = int(self.width_entry.get())
        threshold_value = self.threshold_slider.get()
        # Convertir la imagen a escala de grises
        into_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Aplicar un umbral adaptativo para binarizar la imagen
        _, threshold = cv2.threshold(into_gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Realizar operaciones morfológicas para eliminar el ruido y cerrar los contornos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img_processed = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos
        outlines, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar los contornos basado en la longitud y anchura mínima especificada
        grains = []
        grain_data = []
        for outline in outlines:
            x, y, w, h = cv2.boundingRect(outline)
            if w >= min_width and h >= min_length:
                grains.append(outline)
                rect = cv2.minAreaRect(outline)
                center, (width, height), angle = rect
                # print(rect)

                # Creamos una instancia del objeto Grain y la añadimos a la lista
                grain = Grano(center, width, height, angle, outline)
                grain_data.append(grain)

        # Para sacar la información en la consola
        for g in grain_data:
            print(g)

        # Actualizar el cuadro de texto con el número de semillas detectadas
        self.grain_count_label.config(text=f"Semillas detectadas: {len(grains)}")

        # Dibujar los contornos detectados en la imagen original
        cv2.drawContours(image, grains, -1, (0, 255, 0), 2)

        # Redimensionar la imagen para adaptarla a la pantalla del ordenador
        # resized_image = cv2.resize(image, (1980, 1080))

        # Actualiza self.grain_data y solo devuelve la imagen:
        self.grain_data = grain_data

        return image, grain_data

    def detect_grains(self):
        if self.image is not None:
            self.open_image(self.filename)
            processed_image = self.image.copy()

            # Check if there is a selected rectangle
            if self.selection_rectangle is not None:
                x1, y1, x2, y2 = self.selection_rectangle

                # Scale the coordinates of the ROI based on the current zoom factor
                x1 = int(x1 / self.zoom_factor)
                y1 = int(y1 / self.zoom_factor)
                x2 = int(x2 / self.zoom_factor)
                y2 = int(y2 / self.zoom_factor)

                print(f'Coordinates of ROI: x1={x1}, y1={y1}, x2={x2}, y2={y2}')

                # Extract the region of interest (ROI) from the image
                roi = self.image[y1:y2, x1:x2]
                print(f'Shape of ROI before processing: {roi.shape}')

                # Process the ROI
                processed_roi, _ = self.grain_detector(roi)
                print(f'Shape of ROI after processing: {processed_roi.shape}')

                # Check if the shapes of the original ROI and processed ROI match
                if roi.shape != processed_roi.shape:
                    print(
                        'Warning: Shapes of original ROI and processed ROI do not match. Resizing processed ROI to match original ROI.')
                    processed_roi = cv2.resize(processed_roi, (roi.shape[1], roi.shape[0]))

                # Insert the processed ROI back into the image
                processed_image[y1:y2, x1:x2] = processed_roi

                # Reset the selected rectangle
                # self.selection_rectangle = None

            # Resize and center the image
            image = self.resize_image(processed_image)

            # Update image dimensions label
            height, width, _ = image.shape
            self.image_dimensions_label.config(text=f"Dimensions: {width}x{height}")

            # Convert the image to RGB for displaying in Tkinter (OpenCV uses BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the processed image to a format compatible with Tkinter
            tk_img = ImageTk.PhotoImage(Image.fromarray(image))

            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            self.canvas.image = tk_img
            self.statusbar.config(
                text=str(self))  # Cada vez que se pulse al botón "Detectar Granos" se actualizará el status bar

    def resize_image(self, img):
        # Obtén las dimensiones del canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calcula la relación de aspecto del canvas
        canvas_aspect = canvas_width / canvas_height

        # Calcula la relación de aspecto de la imagen
        image_height, image_width = img.shape[:2]
        image_aspect = image_width / image_height

        if image_aspect > canvas_aspect:
            # Si la relación de aspecto de la imagen es mayor que la del canvas
            # entonces ajusta la imagen a la anchura del canvas
            scale_ratio = canvas_width / image_width
        else:
            # Si no, ajusta la imagen a la altura del canvas
            scale_ratio = canvas_height / image_height

        # Si la imagen es más grande que el canvas, redimensiona
        if scale_ratio < 1:
            new_width = int(image_width * scale_ratio)
            new_height = int(image_height * scale_ratio)
            resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            # Si la imagen es más pequeña que el canvas, deja la imagen como está
            resized_image = img

        return resized_image

    def perform_analysis(self):
        # Verificar si hay datos de grano
        if not hasattr(self, 'grain_data') or not self.grain_data:
            messagebox.showwarning("Warning", "No grains detected yet. Please detect grains first.")
            return
        if Grano.px_per_cm is None:
            messagebox.showwarning("Warning", "No pixels per centimeter detected. Please use Measure function first.")
            return

        # Solicitar al usuario el nombre del archivo CSV
        csv_filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not csv_filename:
            return

        # Definir la cabecera para el archivo CSV
        header = [
            "Center", "Width (Cm)", "Height (Cm)", "Contour Area (Cm^2)", "Contour Perimeter (Cm)",
            "Solidity", "Fit Ellipse", "Rectangle Fill", "Bounding Rectangle Perimeter",
            "Equivalent Diameter (Cm)", "Circulation Factor", "Compactness", "Elongation",
            "Aspect Ratio", "Ratio Surface of Volume", "Mean RGB", "NDIrg",
            "NDIrb", "NDIgb", "Mean HSV", "GLCM contrast", "GLCM Dissimilarity", "Homogeneity",
            "ASM", "Correlation"
        ]

        # Crear el archivo CSV y escribir los datos
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            print(f"{self.grain_data}")
            for grain in self.grain_data:
                try:
                    data = [
                        grain.center, grain.width / Grano.px_per_cm, grain.height / Grano.px_per_cm,
                                      grain.contourArea() / (Grano.px_per_cm ** 2),
                                      grain.contourPerimeter() / Grano.px_per_cm,
                        grain.solidity(), grain.fitEllipse(),
                        grain.rectangleFill(),
                        grain.boundingRectanglePerimeter(), grain.equivalentDiameter() / Grano.px_per_cm,
                                      grain.circulationFactor() / Grano.px_per_cm, grain.compactness(),
                        grain.elongation(), grain.aspectRatio(),
                        grain.ratioSurfaceVolume(),
                        grain.meanRGB(self.image), grain.NDIrg(self.image),
                        grain.NDIrb(self.image), grain.NDIgb(self.image),
                        grain.mean_hsv(self.image), grain.glcm_contrast(self.image),
                        grain.glcm_dissimilarity(self.image),
                        grain.homogeneity_v2(self.image), grain.ASM_v2(self.image), grain.correlation_v2(self.image)
                    ]
                    writer.writerow(data)
                except Exception as e:
                    print(f"Error processing grain: {e}")

        messagebox.showinfo("Analysis Completed", f"Data saved to {csv_filename}")

    def exit_program(self):
        self.destroy()


if __name__ == "__main__":
    app = GrainDetectorApp()
    app.mainloop()