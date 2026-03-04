import numpy as np
import random
import pandas as pd


class Neuronio:
    def __init__(self, n_parametros, eta=0.05):
        self.eta = eta
        # Iniciamos pesos e bias aleatoriamente
        self.w = [random.uniform(-1, 1) for _ in range(n_parametros)]
        self.b = random.uniform(-1, 1)

    def ativacao(self, soma):
        # Função de degrau unitário
        return 1 if soma >= 0 else 0

    def teste(self, x):
        # Calcula a soma ponderada: (w1*x1 + w2*x2 + ... + wn*xn) + b
        soma = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.b
        return self.ativacao(soma)

    def treinamento(self, dados, n_loops=100):
        for _ in range(n_loops):
            for row in dados:
                x = row[:-1]  # Entradas (todas as colunas exceto a última)
                esperado = row[-1]  # Saída desejada (última coluna)

                y = self.teste(x)
                erro = esperado - y

                # Ajuste de pesos e bias se houver erro
                if erro != 0:
                    for i in range(len(self.w)):
                        self.w[i] += self.eta * erro * x[i]
                    self.b += self.eta * erro

    def calcular_acertos(self, dados):
        acertos = 0
        total = len(dados)

        for row in dados:
            x = row[:-1]
            esperado = row[-1]
            previsao = self.teste(x)

            if previsao == esperado:
                acertos += 1

        taxa = (acertos / total) * 100
        return taxa


def carregar_e_limpar_dados(arquivo):
    """
    Função para tratar os CSVs que possuem pontos como separadores de milhar
    e normalizar os valores para o neurônio conseguir processar.
    """
    try:
        # Lê como string para evitar erro imediato de conversão
        df = pd.read_csv(arquivo, dtype=str)

        # Remove os pontos das colunas de entrada (x1, x2) e converte para float
        # Assume-se que as duas primeiras colunas são as entradas
        colunas_entrada = df.columns[:2]
        for col in colunas_entrada:
            df[col] = df[col].str.replace('.', '', regex=False).astype(float)

        # Converte a coluna de classe para int
        df.iloc[:, -1] = df.iloc[:, -1].astype(int)

        # Normalização: Essencial para lidar com números na casa dos quatrilhões
        # Transforma os valores para a escala entre 0 e 1
        for col in colunas_entrada:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val != 0:
                df[col] = (df[col] - min_val) / (max_val - min_val)

        return df.to_numpy()
    except Exception as e:
        print(f"Erro ao carregar o arquivo {arquivo}: {e}")
        return None


# --- Execução Principal ---
if __name__ == '__main__':
    # 1. Carregamento dos datasets
    # O arquivo 'and.csv' geralmente é simples, mas usaremos a mesma função por segurança
    dados_and = carregar_e_limpar_dados('and.csv')
    dados_linear = carregar_e_limpar_dados('dataset_linearmente_separavel.csv')
    dados_nao_linear = carregar_e_limpar_dados('dataset_nao_linearmente_separavel.csv')

    datasets = [
        ("PORTA LÓGICA AND", dados_and),
        ("LINEARMENTE SEPARÁVEL", dados_linear),
        ("NÃO LINEARMENTE SEPARÁVEL", dados_nao_linear)
    ]

    for nome, dados in datasets:
        if dados is not None:
            print(f"\n=== Testando: {nome} ===")

            # Criar neurônio com 2 entradas
            n = Neuronio(n_parametros=2)

            # Acurácia antes do treino
            acc_antes = n.calcular_acertos(dados)
            print(f"Acurácia inicial: {acc_antes:.2f}%")

            # Treinamento
            n.treinamento(dados, n_loops=500)

            # Acurácia após o treino
            acc_depois = n.calcular_acertos(dados)
            print(f"Acurácia final: {acc_depois:.2f}%")

            # Mostrar alguns exemplos de predição
            print("Exemplos (Input -> Predição):")
            for row in dados[:4]:  # Mostra apenas os 4 primeiros para não encher a tela
                x_ex = row[:-1]
                print(f"  {x_ex} -> {n.teste(x_ex)} (Esperado: {int(row[-1])})")