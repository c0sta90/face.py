# Reconhecimento Facial com OpenCV e LBPH

## 1. Descrição do Projeto
Este projeto implementa um sistema de **reconhecimento facial em tempo real** usando Python e OpenCV. Ele permite:

- Cadastrar pessoas com fotos capturadas da câmera, incluindo variações de ângulo.  
- Armazenar informações (nome, idade e número de fotos) em um **banco de dados JSON**.  
- Reconhecer rostos em tempo real, mostrando **nome + idade** ao redor do rosto.  
- Melhorar o reconhecimento adicionando novas fotos para a mesma pessoa.

O sistema é modularizado em **classes** e arquivos separados para fácil manutenção.

---

## 2. Estrutura de Arquivos

reconhecimento_project/
-   │
-   ├── reconhecimento_head.py  # Classe para gerenciar banco e configurações
-   ├── reconhecimento_face.py  # Classe para cadastro e reconhecimento
-   ├── main.py                 # Menu principal para executar o programa
-   ├── faces/                  # Pasta para armazenar fotos
-   └── requirements.txt        # Bibliotecas necessárias


---

## 3. Bibliotecas Utilizadas

| Biblioteca       | Função no Projeto |
|-----------------|-----------------|
| `cv2`           | Captura vídeo, processa imagens, detecta rostos, desenha retângulos, mostra janela e salva fotos. |
| `os`            | Cria e gerencia pastas e arquivos, verifica existência de arquivos. |
| `json`          | Salva e lê informações das pessoas cadastradas (nome, idade, número de fotos). |
| `numpy`         | Manipula arrays de imagens e labels para treinamento do LBPH. |
| `time`          | Pausas entre capturas de fotos para variar expressões. |

---

## 4. Classes do Projeto

### 4.1 `ReconhecimentoHead`
Responsável por **gerenciar variáveis globais e configurações** do sistema:

- `BASE_DIR` → pasta onde fotos serão salvas  
- `DB_FILE` → arquivo JSON com informações do banco  
- `banco` → dicionário carregado do JSON com os dados das pessoas  
- `face_cascade` → classificador Haar Cascade do OpenCV  
- Configurações de detecção: `scaleFactor`, `minNeighbors`, `minSize`  
- `fotos_por_vez` → número de fotos capturadas por cadastro  
- Método `salvar_banco()` → salva o banco atualizado no arquivo JSON  

---

### 4.2 `ReconhecimentoFace`
Responsável pelo **cadastro e reconhecimento facial**:

- `cadastrar(nome, idade)` → captura fotos da pessoa, aplica variações de ângulo e flip horizontal, salva em pastas e atualiza banco.  
- `treinar_reconhecedor()` → treina o reconhecedor LBPH usando todas as fotos cadastradas.  
- `reconhecer()` → captura vídeo em tempo real, reconhece rostos e mostra **nome + idade** ao redor de cada rosto.

**Melhorias implementadas:**

- Captura fotos de vários ângulos e flips horizontais para melhorar reconhecimento quando a pessoa olha para os lados.  
- Usa parâmetros de detecção configuráveis do `ReconhecimentoHead`.  
- Atualiza o banco JSON automaticamente.

---

## 5. Como Usar

### 5.1 Instalação de Dependências
Recomenda-se criar um **ambiente virtual**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-python opencv-contrib-python numpy
