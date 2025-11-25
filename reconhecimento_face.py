from bibliotecas import cv2, os, np
from reconhecimento_head import ReconhecimentoHead

class ReconhecimentoFace:
    """
    Classe que gerencia o cadastro de rostos e o reconhecimento facial usando LBPH.
    Usa as configurações e variáveis do ReconhecimentoHead.
    """

    def __init__(self, head: ReconhecimentoHead):
        self.head = head
        self.banco = head.banco
        self.BASE_DIR = head.BASE_DIR
        self.DB_FILE = head.DB_FILE
        self.face_cascade = head.face_cascade

    # ---------------- Cadastro de pessoas ----------------
    def cadastrar(self, nome: str, idade: str):
        """
        Captura fotos da pessoa, incluindo variações de ângulo e flip horizontal,
        para melhorar o reconhecimento quando ela olha para os lados.
        """
        cap = cv2.VideoCapture(0)   #Abre a câmera
        pasta_fotos = os.path.join(self.BASE_DIR, nome) #Cria pasta
        os.makedirs(pasta_fotos, exist_ok=True)

        # Conta quantas fotos já existem na pasta
        fotos_existentes = len([f for f in os.listdir(pasta_fotos) if f.endswith(".jpg")])
        limite_fotos = fotos_existentes + self.head.fotos_por_vez
        fotos = fotos_existentes

        print(">>> Olhe para a câmera, vire a cabeça para os lados... (Q para sair)")

        while fotos < limite_fotos:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostos = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.head.scaleFactor,
                minNeighbors=self.head.minNeighbors,
                minSize=self.head.minSize
            )

            for (x, y, w, h) in rostos:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                rosto = gray[y:y+h, x:x+w]

                # Salvar rosto original
                cv2.imwrite(os.path.join(pasta_fotos, f"{fotos}.jpg"), rosto)
                # Salvar flip horizontal para simular olhar para os lados
                cv2.imwrite(os.path.join(pasta_fotos, f"{fotos}_flip.jpg"), cv2.flip(rosto, 1))
                fotos += 2  # Contou duas fotos (original + flip)

            cv2.putText(frame, f"Fotos: {fotos - fotos_existentes}/{self.head.fotos_por_vez}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Cadastro", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):   #Aperta q para sair
                break

        cap.release()
        cv2.destroyAllWindows()

        # Atualiza banco
        self.banco[nome] = {"idade": idade, "num_fotos": fotos}
        self.head.salvar_banco()
        print(f"Cadastro concluído! Total de fotos: {fotos}")

    # ---------------- Treinar reconhecedor LBPH ----------------
    def treinar_reconhecedor(self):
        """
        Treina o reconhecedor LBPH usando todas as fotos cadastradas.
        Retorna o recognizer e o dicionário de labels.
        """
        faces = []          # Lista de imagens de rosto
        labels = []         # Lista de pessoas correspondentes a cada rosto
        label_dict = {}     # Dicionário para mapear nomes a pessoas numéricas
        current_label = 0   # Contador de pessoas

        # Carrega todas as fotos do banco
        for nome in self.banco:
            pasta_fotos = os.path.join(self.BASE_DIR, nome)
            arquivos = [f for f in os.listdir(pasta_fotos) if f.endswith(".jpg")]
            if nome not in label_dict:
                label_dict[nome] = current_label
                current_label += 1
            for f in arquivos:
                img = cv2.imread(os.path.join(pasta_fotos, f), cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(label_dict[nome])

        if len(faces) == 0:
            return None, None

        recognizer = cv2.face.LBPHFaceRecognizer_create()       # Cria o reconhecedor LBPH
        recognizer.train(faces, np.array(labels))               # Treina com as faces e pessoas
        return recognizer, {v: k for k, v in label_dict.items()} 

    # ---------------- Reconhecimento ----------------
    def reconhecer(self):
        """
        Captura da câmera em tempo real e reconhece rostos cadastrados.
        Mostra quadrado em volta do rosto com nome e idade.
        """
        recognizer, label_reverse = self.treinar_reconhecedor()
        if recognizer is None:
            print("Nenhuma pessoa cadastrada!")
            return

        cap = cv2.VideoCapture(0)
        print(">>> Pressione Q para sair do reconhecimento.")

        # Loop de captura
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converte para escala de cinza
            rostos = self.face_cascade.detectMultiScale(   # Coloca quadrado em volta do rosto
                gray, scaleFactor=1.2, minNeighbors=5
            )

            # Para cada rosto detectado, tenta reconhecer
            for (x, y, w, h) in rostos:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Desenha retângulo
                rosto = gray[y:y+h, x:x+w]                               # Recorta o rosto
                try:
                    label, conf = recognizer.predict(rosto)
                    if conf > 70:  # limite para considerar desconhecido
                        texto = "Desconhecido"
                    else:
                        nome = label_reverse[label]
                        idade = self.banco[nome]["idade"]
                        texto = f"{nome}, {idade} anos" if idade else nome
                except:
                    texto = "Desconhecido"
                # Escreve o nome acima do rosto
                cv2.putText(frame, texto, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Reconhecimento", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Reconhecimento encerrado.")
