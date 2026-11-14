import cv2
import pygame
from BodyTrackingModule import BodyDetector 

def main():
    # Inicializar el motor de audio
    pygame.mixer.init()
    
    # 2. Cargar tus archivos .wav
    sonido_do = pygame.mixer.Sound('do2.wav')
    sonido_re = pygame.mixer.Sound('re2.wav')
    sonido_mi = pygame.mixer.Sound('mi2.wav')
    sonido_fa = pygame.mixer.Sound('fa2.wav')
    sonido_sol = pygame.mixer.Sound('sol2.wav')
    sonido_la = pygame.mixer.Sound('la2.wav')
    sonido_si = pygame.mixer.Sound('si2.wav')
    sonido_dod = pygame.mixer.Sound('dod2.wav')
    
    # 3. Asignar un canal exclusivo para el brazo izquierdo
    canal_izquierdo = pygame.mixer.Channel(0)
    
    cap = cv2.VideoCapture('0')
    detector = BodyDetector()

    zona_actual = 0 

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findBody(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0:
            cy = lmList[15][2] 
            h, w, c = img.shape 
            y_invertida = h - cy 
            
            intensidad = int((y_invertida / h) * 100)
            intensidad = max(0, min(100, intensidad)) 
            
            # Mapeo matemático directo de 0-100% a 8 zonas (12.5% cada una)
            nueva_zona = int(intensidad / 12.5) + 1
            nueva_zona = min(8, nueva_zona) # Clamping de seguridad al límite superior
                
            # 4. Lógica de disparo y reproducción en el canal
            if nueva_zona != zona_actual:
                print(f"🎵 Activando Nivel {nueva_zona}")
                
                if nueva_zona == 1:
                    canal_izquierdo.play(sonido_do)
                elif nueva_zona == 2:
                    canal_izquierdo.play(sonido_re)
                elif nueva_zona == 3:
                    canal_izquierdo.play(sonido_mi)
                elif nueva_zona == 4:
                    canal_izquierdo.play(sonido_fa)
                elif nueva_zona == 5:
                    canal_izquierdo.play(sonido_sol)
                elif nueva_zona == 6:
                    canal_izquierdo.play(sonido_la)
                elif nueva_zona == 7:
                    canal_izquierdo.play(sonido_si)
                elif nueva_zona == 8:
                    canal_izquierdo.play(sonido_dod)
                
                zona_actual = nueva_zona

        cv2.imshow("Prueba Brazo Izquierdo", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
