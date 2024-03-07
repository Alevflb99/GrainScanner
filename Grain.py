import math
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class Grano:
    px_per_cm = None
    def __init__(self, center, width, height, angle, contour):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.contour = contour

    def __str__(self):
        return f"Grain(center={self.center}, width={self.width}, height={self.height}, angle={self.angle})"

    def contourArea(self):
        return cv2.contourArea(self.contour)

    def contourPerimeter(self):
        return cv2.arcLength(self.contour, True)

    def solidity(self):
        hull = cv2.convexHull(self.contour)
        hull_area = cv2.contourArea(hull)
        return float(self.contourArea()) / hull_area

    def fitEllipse(self):
        return cv2.fitEllipse(self.contour)

    def rectangleFill(self):
        # Crear una máscara de ceros del tamaño del rectángulo delimitador
        mask = np.zeros((int(self.height), int(self.width)), dtype=np.uint8)

        # Transladar el contorno a la esquina superior izquierda de la máscara
        shifted_contour = self.contour - [int(self.center[0] - self.width / 2), int(self.center[1] - self.height / 2)]

        # Dibujar el contorno relleno en la máscara
        cv2.drawContours(mask, [shifted_contour], -1, (255), thickness=cv2.FILLED)

        # Contar cuántos píxeles están ocupados por el grano
        grain_pixels = np.sum(mask == 255)

        # Calcular el relleno
        fill_ratio = grain_pixels / (self.width * self.height)
        return fill_ratio

    def boundingRectanglePerimeter(self):
        perimeter = self.contourPerimeter()
        brp= perimeter/(2*(self.height+self.width))
        return brp

    def equivalentDiameter(self):
        area = self.contourArea()
        return math.sqrt(4*area/math.pi)

    def circulationFactor(self):
        perimeter=self.contourPerimeter()
        return perimeter/math.pi

    def compactness(self):
        area = self.contourArea()
        perimeter = self.contourPerimeter()
        return (4*math.pi*area)/(perimeter**2)

    def elongation(self):
        M = max(self.width, self.height)
        m = min(self.width, self.height)

        if M + m == 0:  # Para evitar la división por cero
            return 0  # En este caso, deberíamos tratarlo como un objeto puntual, por lo que la elongación sería 0

        return (M - m) / (M + m)

    def aspectRatio(self):
        L = max(self.width, self.height)
        B = min(self.width, self.height)
        if B == 0:
                return float("inf")

        return L/B

    def feret_diameter(self):  #Esta funcion está definida para ayudarnos más adelante
        max_distance = 0
        for i in range(len(self.contour)):
            for j in range(i + 1, len(self.contour)):
                distance = cv2.norm(self.contour[i] - self.contour[j])
                if distance > max_distance:
                    max_distance = distance
        return max_distance

    def ratioSurfaceVolume(self):
        A = self.contourArea()  # El área del contorno

        # Calcular el radio usando el diámetro equivalente
        #r = self.feret_diameter() / 2
        r = self.equivalentDiameter() / 2

        # Calcular el volumen cúbico usando la fórmula de la esfera
        M = (4 / 3) * math.pi * r ** 3

        if M == 0:  # Para evitar la división por cero
            return float('inf')

        return A / (M**3)
#-----------------------------------------COLORES-----------------------------------------#

    def meanRGB(self, origImg):
        # Usar el contorno para extraer la región de interés (ROI) de la imagen
        mask = np.zeros(origImg.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, (255), thickness=cv2.FILLED)

        mean_val = cv2.mean(origImg, mask=mask)

        # mean_val seguirá el orden BGR debido a la convención de OpenCV
        return {"blue": mean_val[0], "green": mean_val[1], "red": mean_val[2]}

    def NDIrg(self, original_image):
        mean_values = self.meanRGB(original_image)

        R = mean_values["red"]
        G = mean_values["green"]

        # Evitar la división por cero añadiendo un pequeño valor epsilon
        epsilon = 1e-7
        ndi_rg = abs(R - G) / (R + G + epsilon)

        return ndi_rg


    def NDIrb(self, original_image):
        mean_values = self.meanRGB(original_image)

        R = mean_values["red"]
        B = mean_values["blue"]

        # Evitar la división por cero añadiendo un pequeño valor epsilon
        epsilon = 1e-7
        ndi_rb = abs(R - B) / (R + B + epsilon)

        return ndi_rb

    def NDIgb(self, original_image):
        mean_values = self.meanRGB(original_image)

        G = mean_values["green"]
        B = mean_values["blue"]

        # Evitar la división por cero añadiendo un pequeño valor epsilon
        epsilon = 1e-7
        ndi_gb = abs(G - B) / (G + B + epsilon)

        return ndi_gb

    def mean_hsv(self, original_image):
        # Convertir la imagen BGR a HSV
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Máscara binaria del contorno del grano
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, (255), thickness=cv2.FILLED)

        # Calcular el valor medio de cada canal HSV
        mean_h, mean_s, mean_v, _ = cv2.mean(hsv_image, mask=mask)

        return {"hue": mean_h, "saturation": mean_s, "value": mean_v}


    def glcm_contrast(self, origImg):
        grayscale_image = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
        # Usar solo los píxeles dentro del contorno del grano para calcular GLCM
        mask = np.zeros(grayscale_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, (255), thickness=cv2.FILLED)

        # Aplicar la máscara a la imagen en escala de grises
        roi = cv2.bitwise_and(grayscale_image, grayscale_image, mask=mask)

        # Calcular la matriz GLCM
        glcm = graycomatrix(roi, [1], [0], 256, symmetric=True, normed=True)

        # Calcular el contraste
        levels = glcm.shape[0]
        contrast = 0
        for i in range(levels):
            for j in range(levels):
                contrast += glcm[i, j, 0, 0] * (i - j) ** 2

        return contrast

    def glcm_dissimilarity(self, origImg):
        grayscale_image = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
        # Usar solo los píxeles dentro del contorno del grano para calcular GLCM
        mask = np.zeros(grayscale_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, (255), thickness=cv2.FILLED)

        # Aplicar la máscara a la imagen en escala de grises
        roi = cv2.bitwise_and(grayscale_image, grayscale_image, mask=mask)

        # Calcular la matriz GLCM
        glcm = graycomatrix(roi, [1], [0], 256, symmetric=True, normed=True)

        # Calcular la disimilitud
        levels = glcm.shape[0]
        dissimilarity = 0
        for i in range(levels):
            for j in range(levels):
                dissimilarity += glcm[i, j, 0, 0] * abs(i - j)

        return dissimilarity


    def homogeneity_v2(self, origImg):
        grayscale_image = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)

        # Máscara binaria del contorno del grano
        mask = np.zeros(grayscale_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, (255), thickness=cv2.FILLED)

        # Aplicar la máscara a la imagen en escala de grises
        roi = cv2.bitwise_and(grayscale_image, grayscale_image, mask=mask)

        # Calcular la matriz GLCM
        glcm = graycomatrix(roi, [1], [0], symmetric=True, normed=True)

        # Calcula la homogeneidad usando greycoprops
        homogeneity_value = graycoprops(glcm, 'homogeneity')[0, 0]

        return homogeneity_value



    def ASM_v2(self, origImg):  # También se le llama Energía
        grayscale_image = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)

        # Máscara binaria del contorno del grano
        mask = np.zeros(grayscale_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, (255), thickness=cv2.FILLED)

        # Aplicar la máscara a la imagen en escala de grises
        roi = cv2.bitwise_and(grayscale_image, grayscale_image, mask=mask)

        # Calcular la matriz GLCM
        glcm = graycomatrix(roi, [1], [0], symmetric=True, normed=True)

        # Calcula la energía usando greycoprops
        energy_value = graycoprops(glcm, 'energy')[0, 0]

        # La energía es la raíz cuadrada del ASM, así que elevamos al cuadrado para obtener el ASM
        asm_value = energy_value ** 2

        return asm_value



    def correlation_v2(self, origImg):
        grayscale_image = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)

        # Máscara binaria del contorno del grano
        mask = np.zeros(grayscale_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, (255), thickness=cv2.FILLED)

        # Aplicar la máscara a la imagen en escala de grises
        roi = cv2.bitwise_and(grayscale_image, grayscale_image, mask=mask)

        # Calcular la matriz GLCM
        glcm = graycomatrix(roi, [1], [0], symmetric=True, normed=True)

        # Calcula la correlación usando greycoprops
        correlation_value = graycoprops(glcm, 'correlation')[0, 0]

        return correlation_value
