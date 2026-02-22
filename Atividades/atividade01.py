import random
class Neuronio:

    def __init__(self, n_parametros, eta = 0.05):
        self.eta = eta
        self.w = [random.uniform(-1,1) for _ in range(n_parametros)]
        self.b = random.uniform(-1,1)

    def ativacao(self, soma):
        if soma >= 0.0:
            return 1
        else:
            return 0

    def teste(self, x):
        soma = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.b
        return self.ativacao(soma)


    def treinamento(self, dados, n_loops = 100):
        for _ in range(n_loops):
            for x, esperado in dados:
                y = self.teste(x)
                erro = esperado - y

                for i in range(len(self.w)):
                    self.w[i] += self.eta * erro * x[i]

                self.b += self.eta * erro


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dados_and = [
        ([0,0], 0),
        ([1,0], 0),
        ([0,1], 0),
        ([1,1], 1),
    ]

    dados_or = [
        ([0,0], 0),
        ([1,0], 1),
        ([0,1], 1),
        ([1,1], 1),
    ]

    dados_custom = [
        ([0,0], 0),
        ([0,1], 0),
        ([1,0], 1),
        ([1,1], 1),
    ]

    dados_xor = [
        ([0,0], 0),
        ([0,1], 1),
        ([1,0], 1),
        ([1,1], 0),
    ]

    print("Treinando AND:")
    n_and = Neuronio(2)
    n_and.treinamento(dados_and, n_loops=100)
    for x, _ in dados_and:
        print(x, " -> ", n_and.teste(x))


    print("\nTreinando OR:")
    n_or = Neuronio(2)
    n_or.treinamento(dados_or, n_loops=100)
    for x, _ in dados_or:
        print(x, " -> ", n_or.teste(x))

    print ("\nTreinando A ou (A e B)")
    n_custom = Neuronio(2)
    n_custom.treinamento(dados_custom, n_loops=100)
    for x, _ in dados_custom:
        print(x, " -> ", n_custom.teste(x))

    print("\nTreinando XOR") #não funciona, independente do numero de loops que eu tente adicionar na chamada,
    # o resultado sempre muda, mas nunca está correto
    n_xor = Neuronio(2)
    n_xor.treinamento(dados_xor, n_loops=500)
    for x, _ in dados_xor:
        print(x, " -> ", n_xor.teste(x))
