import cv2
import pygame
from BodyTrackingModule import BodyDetector


def main():

    # ==============================
    # INICIALIZAR AUDIO
    # ==============================

    pygame.mixer.init()

    sonido_do = pygame.mixer.Sound("do2.wav")
    sonido_re = pygame.mixer.Sound("re2.wav")
    sonido_mi = pygame.mixer.Sound("mi2.wav")
    sonido_fa = pygame.mixer.Sound("fa2.wav")
    sonido_sol = pygame.mixer.Sound("sol2.wav")
    sonido_la = pygame.mixer.Sound("la2.wav")
    sonido_si = pygame.mixer.Sound("si2.wav")
    sonido_dod = pygame.mixer.Sound("dod2.wav")

    canal_izquierdo = pygame.mixer.Channel(0)

    sonidos = [
        sonido_do,
        sonido_re,
        sonido_mi,
        sonido_fa,
        sonido_sol,
        sonido_la,
        sonido_si,
        sonido_dod
    ]


    # ==============================
    # ABRIR WEBCAM
    # ==============================

    cap = cv2.VideoCapture(0)

    detector = BodyDetector()

    zona_actual = 0


    # ==============================
    # BUCLE PRINCIPAL
    # ==============================

    while True:

        success, img = cap.read()

        if not success:
            print("No se pudo leer la webcam")
            break


        # Detectar cuerpo
        img = detector.findBody(img)

        # Obtener posiciones del cuerpo
        lmList = detector.findPosition(img)


        # ==============================
        # DETECTAR MUÑECA IZQUIERDA
        # ==============================

        if len(lmList) > 9:

            # En YOLO Pose:
            # 9 = muñeca izquierda
            cy = lmList[9][2]

            h, w, c = img.shape


            # ==============================
            # INVERTIR EJE Y
            # ==============================

            y_invertida = h - cy


            # ==============================
            # CALCULAR INTENSIDAD
            # ==============================

            intensidad = (y_invertida / h) * 100

            intensidad = max(0, min(100, intensidad))


            # ==============================
            # DIVIDIR EN 8 NIVELES
            # ==============================

            nueva_zona = int(intensidad / 12.5) + 1

            # Evitar que llegue a 9
            nueva_zona = min(8, nueva_zona)


            # ==============================
            # MOSTRAR DATOS EN PANTALLA
            # ==============================

            cv2.putText(
                img,
                f"Y: {cy}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Intensidad: {int(intensidad)}%",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Nivel: {nueva_zona}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )


            # ==============================
            # CAMBIO DE NIVEL Y SONIDO
            # ==============================

            if nueva_zona != zona_actual:

                print(
                    f"Y: {cy} | "
                    f"Intensidad: {int(intensidad)}% | "
                    f"Nivel: {nueva_zona}"
                )

                print(f"🎵 Activando Nivel {nueva_zona}")

                canal_izquierdo.play(
                    sonidos[nueva_zona - 1]
                )

                zona_actual = nueva_zona


        # ==============================
        # MOSTRAR WEBCAM
        # ==============================

        cv2.imshow("Prueba Brazo Izquierdo", img)


        # Presiona Q para salir
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # ==============================
    # CERRAR
    # ==============================

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()