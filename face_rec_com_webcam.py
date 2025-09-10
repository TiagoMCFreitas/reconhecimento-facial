import cv2
import face_recognition
import numpy as np


class FacialRecognitionSystem:
    def __init__(self):
        self._known_names: list[str] = []
        self._known_encodings: list[np.ndarray] = []
        self._capture_device = None
        self._running = True

    def _initialize_camera(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        return capture

    def register_user(self, name: str) -> bool:
        self._capture_device = self._initialize_camera()

        while self._running:
            ret, frame = self._capture_device.read()
            if not ret:
                return False

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(
                face_locations, face_encodings
            ):
                scaled_coords = [coord * 4 for coord in [top, right, bottom, left]]
                self._draw_registration_box(frame, scaled_coords)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    self._known_encodings.append(encoding)
                    self._known_names.append(name)
                    self._cleanup()
                    return True
                elif key == ord("q"):
                    break

            cv2.imshow("Registro de Usuario", frame)

        self._cleanup()
        return False

    def start_recognition(self):
        self._capture_device = self._initialize_camera()
        frame_count = 0
        process_rate = 2

        # Adiciona mensagem de instrução
        print("\nPressione 'q' para voltar ao menu principal")

        while self._running:
            ret, frame = self._capture_device.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % process_rate == 0:
                face_data = self._process_frame(frame)
                self._display_recognition_results(frame, face_data)

            # Adiciona texto de instrução na tela
            cv2.putText(
                frame,
                "Pressione 'q' para voltar",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Sistema de Reconhecimento Facial", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._cleanup()
                return  # Retorna ao menu principal

        self._cleanup()

    def _process_frame(self, frame) -> tuple[list, list]:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        return face_locations, face_encodings

    def _display_recognition_results(self, frame, face_data):
        face_locations, face_encodings = face_data

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            scaled_coords = [coord * 4 for coord in [top, right, bottom, left]]
            name = self._identify_face(face_encoding)
            self._draw_recognition_box(frame, scaled_coords, name)

    def _identify_face(self, face_encoding) -> str:
        if not self._known_encodings:
            return "Desconhecido"

        matches = face_recognition.compare_faces(self._known_encodings, face_encoding)
        distances = face_recognition.face_distance(self._known_encodings, face_encoding)

        if not any(matches):
            return "Desconhecido"

        best_match_index = np.argmin(distances)
        return (
            self._known_names[best_match_index]
            if matches[best_match_index]
            else "Desconhecido"
        )

    def _draw_registration_box(self, frame, coords):
        top, right, bottom, left = coords
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Pressione 'c' para capturar",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def _draw_recognition_box(self, frame, coords, name):
        top, right, bottom, left = coords
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
        )
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    def _cleanup(self):
        if self._capture_device:
            self._capture_device.release()
        cv2.destroyAllWindows()
        self._running = True


def main():
    system = FacialRecognitionSystem()

    while True:
        print("\nSistema de Reconhecimento Facial v1.0")
        print("[1] Registrar novo usuário")
        print("[2] Iniciar reconhecimento")
        print("[3] Sair")

        opcao = input("Selecione uma opção: ")

        if opcao == "1":
            nome = input("Nome do usuário: ")
            if system.register_user(nome):
                print(f"Usuário {nome} registrado com sucesso!")
            else:
                print("Registro cancelado.")

        elif opcao == "2":
            system.start_recognition()

        elif opcao == "3":
            print("Encerrando sistema...")
            break

        else:
            print("Opção inválida!")


if __name__ == "__main__":
    main()
