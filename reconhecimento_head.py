from bibliotecas import os, json, cv2

class ReconhecimentoHead:
    """
    Classe que gerencia todas as configurações e variáveis globais do sistema de reconhecimento facial.
    Inclui diretórios, banco de dados, classificador de rosto e parâmetros de captura.
    """

    def __init__(self, base_dir="faces"):
        # ---------------- Diretórios e banco ----------------
        self.BASE_DIR = base_dir                  # Pasta base onde todas as fotos serão salvas
        os.makedirs(self.BASE_DIR, exist_ok=True)

        self.DB_FILE = os.path.join(self.BASE_DIR, "data.json")  # Arquivo JSON para guardar informações
        if os.path.exists(self.DB_FILE):
            with open(self.DB_FILE, "r") as f:
                self.banco = json.load(f)  # Carrega banco existente
        else:
            self.banco = {}  # Inicializa banco vazio

        # ---------------- Classificador de rosto ----------------
        # Cascade do OpenCV para detectar rostos
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # ---------------- Configurações de captura ----------------
        self.fotos_por_vez = 250       # Quantas fotos capturar por cadastro
        self.scaleFactor = 1.1         # Para detectMultiScale: aumenta/reduz a escala da imagem
        self.minNeighbors = 3          # Para detectMultiScale: quantos vizinhos um retângulo precisa
        self.minSize = (50, 50)        # Para detectMultiScale: tamanho mínimo do rosto detectado

    # Salva o banco de dados atualizado
    def salvar_banco(self):
        with open(self.DB_FILE, "w") as f:
            json.dump(self.banco, f, indent=4)
