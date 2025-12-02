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
        os.makedirs(pasta_fotos, exist_ok=True) # Cria a pasta se não existir

        # Conta quantas fotos já existem na pasta
        fotos_existentes = len([f for f in os.listdir(pasta_fotos) if f.endswith(".jpg")]) # Conta fotos
        limite_fotos = fotos_existentes + self.head.fotos_por_vez # Define limite
        fotos = fotos_existentes # Contador de fotos

        print(">>> Olhe para a câmera, vire a cabeça para os lados... (Q para sair)") #Instruções

        while fotos < limite_fotos: # Loop até atingir o limite
            ret, frame = cap.read() # Captura frame
            if not ret: # Se não capturou, sai do loop
                break # Sai do loop

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converte para escala de cinza
            rostos = self.face_cascade.detectMultiScale( # Detecta rostos
                gray, # Imagem em escala de cinza
                scaleFactor=self.head.scaleFactor, # Fator de escala
                minNeighbors=self.head.minNeighbors, # Vizinhos mínimos para detecção 
                minSize=self.head.minSize # Tamanho mínimo do rosto
            )

            for (x, y, w, h) in rostos: # Para cada rosto detectado
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Desenha retângulo
                rosto = gray[y:y+h, x:x+w] # Recorta o rosto

                # Salvar rosto original
                cv2.imwrite(os.path.join(pasta_fotos, f"{fotos}.jpg"), rosto) # Salva a foto
                # Salvar flip horizontal para simular olhar para os lados 
                cv2.imwrite(os.path.join(pasta_fotos, f"{fotos}_flip.jpg"), cv2.flip(rosto, 1)) # Salva a foto invertida
                fotos += 2  # Contou duas fotos (original + flip) # Incrementa contador

            cv2.putText(frame, f"Fotos: {fotos - fotos_existentes}/{self.head.fotos_por_vez}", (10, 30), # Mostra contador de fotos
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) # Desenha o texto na tela
            cv2.imshow("Cadastro", frame) # Mostra o frame

            if cv2.waitKey(1) & 0xFF == ord("q"):   #Aperta q para sair
                break # Sai do loop

        cap.release() # Libera a câmera
        cv2.destroyAllWindows() # Fecha as janelas

        # Atualiza banco
        self.banco[nome] = {"idade": idade, "num_fotos": fotos} # Adiciona pessoa ao banco
        self.head.salvar_banco() # Salva o banco atualizado
        print(f"Cadastro concluído! Total de fotos: {fotos}") # Confirmação de cadastro

    # ---------------- Treinar reconhecedor LBPH ----------------
    def treinar_reconhecedor(self): # -> tuple:
        """
        Treina o reconhecedor LBPH usando todas as fotos cadastradas. 
        Retorna o recognizer e o dicionário de labels.
        """
        faces = []          # Lista de imagens de rosto
        labels = []         # Lista de pessoas correspondentes a cada rosto
        label_dict = {}     # Dicionário para mapear nomes a pessoas numéricas
        current_label = 0   # Contador de pessoas

        # Carrega todas as fotos do banco
        for nome in self.banco: # Para cada pessoa no banco
            pasta_fotos = os.path.join(self.BASE_DIR, nome) # Caminho da pasta da pessoa
            arquivos = [f for f in os.listdir(pasta_fotos) if f.endswith(".jpg")] # Lista de fotos
            if nome not in label_dict: # Se a pessoa não está no dicionário
                label_dict[nome] = current_label # Adiciona ao dicionário
                current_label += 1 # Incrementa o contador de pessoas
            for f in arquivos: # Para cada foto da pessoa
                img = cv2.imread(os.path.join(pasta_fotos, f), cv2.IMREAD_GRAYSCALE) # Lê a imagem em escala de cinza
                faces.append(img) # Adiciona a imagem à lista de faces
                labels.append(label_dict[nome]) # Adiciona o rótulo correspondente

        if len(faces) == 0: # Se não há fotos cadastradas
            return None, None # Retorna None

        recognizer = cv2.face.LBPHFaceRecognizer_create()       # Cria o reconhecedor LBPH
        recognizer.train(faces, np.array(labels))               # Treina com as faces e pessoas
        return recognizer, {v: k for k, v in label_dict.items()} # Retorna o reconhecedor e o dicionário invertido

    # ---------------- Reconhecimento ----------------
    def reconhecer(self):
        """
        Captura da câmera em tempo real e reconhece rostos cadastrados.
        Mostra quadrado em volta do rosto com nome e idade.
        """
        recognizer, label_reverse = self.treinar_reconhecedor() # Treina o reconhecedor
        if recognizer is None: # Se não há fotos cadastradas
            print("Nenhuma pessoa cadastrada!")
            return

        cap = cv2.VideoCapture(0) # Abre a câmera
        print(">>> Pressione Q para sair do reconhecimento.")

        # Loop de captura
        while True: # Loop infinito
            ret, frame = cap.read() # Captura frame
            if not ret: # Se não capturou, sai do loop
                break # Sai do loop

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converte para escala de cinza
            rostos = self.face_cascade.detectMultiScale(   # Coloca quadrado em volta do rosto
                gray, scaleFactor=1.2, minNeighbors=5 #, minSize=(100, 100)
            )

            # Para cada rosto detectado, tenta reconhecer
            for (x, y, w, h) in rostos: # Para cada rosto detectado
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Desenha retângulo
                rosto = gray[y:y+h, x:x+w]                               # Recorta o rosto
                try: # Tenta reconhecer o rosto
                    label, conf = recognizer.predict(rosto) # Reconhece o rosto
                    if conf > 70:  # limite para considerar desconhecido
                        texto = "Desconhecido" 
                    else:
                        nome = label_reverse[label] # Obtém o nome pelo rótulo
                        idade = self.banco[nome]["idade"]
                        texto = f"{nome}, {idade} anos" if idade else nome
                except:
                    texto = "Desconhecido" 
                # Escreve o nome acima do rosto
                cv2.putText(frame, texto, (x, y-10), # Posição do texto
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Desenha o texto na tela

            cv2.imshow("Reconhecimento", frame) # Mostra o frame
            if cv2.waitKey(1) & 0xFF == ord("q"): # Aperta q para sair
                break

        cap.release() # Libera a câmera
        cv2.destroyAllWindows() # Fecha as janelas
        print("Reconhecimento encerrado.") # Confirmação de encerramento
