from reconhecimento_head import ReconhecimentoHead
from reconhecimento_face import ReconhecimentoFace

def main():
    head = ReconhecimentoHead()
    face_rec = ReconhecimentoFace(head)

    while True:
        print("\n--- MENU ---")
        print("1 - Cadastrar pessoa")
        print("2 - Reconhecer pessoa")
        print("3 - Sair")
        op = input("Escolha: ")

        if op == "1":
            nome = input("Nome: ")
            idade = input("Idade: ")
            face_rec.cadastrar(nome, idade)
        elif op == "2":
            face_rec.reconhecer()
        elif op == "3":
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    main()
