import cv2
import numpy as np

# Cargar la red YOLO preentrenada
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Cargar las clases del modelo YOLO
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Cargar el video desde un archivo
video_path = "tfvideo.mp4"
cap = cv2.VideoCapture(video_path)

# Configuración para contar un objeto específico
target_class = "person"  

# Obtener las dimensiones del video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Salida del video
output_path = "output_video.avi"  
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Bucle principal para procesar cada frame del video
while True:
    count = 0 

    # Leer un frame del video
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener las dimensiones del frame
    height, width = frame.shape[:2]

    # Preprocesamiento de la imagen para YOLO
    # normalizamos el frame - cambiamos el formato de BGR a RGB - no recortamos la imagen
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Realizar la detección de objetos con YOLO
    detections = net.forward(layer_names)

    # Iterar sobre todas las detecciones
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            # obtenemos la confianza
            confidence = scores[class_id]

            # Verificar confianza
            if confidence > 0.4 and classes[class_id] == target_class:
                count += 1  # Incrementar el contador

                # Obtener las coordenadas del cuadro delimitador
                center_x, center_y = int(obj[0] * width), int(obj[1] * height)
                w, h = int(obj[2] * width), int(obj[3] * height)
                x, y = center_x - w // 2, center_y - h // 2

                # Dibujar el cuadro delimitador y mostrar la etiqueta
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f'{classes[class_id]}: {confidence:.2f}'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Contador
    cv2.putText(frame, f'{target_class}: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Escribir el frame resultante en el video de salida
    out.write(frame)

    # Mostrar el frame resultante
    cv2.imshow('Object Detection', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
out.release()
cv2.destroyAllWindows()