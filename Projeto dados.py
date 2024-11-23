import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

class ModeloComparacao:
    def __init__(self):
        self.df = None
        self.svm_model = SVC()
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestClassifier(random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.

        Parâmetros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.
        """
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)

    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.
        """
        print(self.df.head())

        if self.df.isnull().sum().any():
            print("Valores ausentes detectados, preenchendo com média.")
            self.df.fillna(self.df.mean(), inplace=True)

        self.df['Species'] = self.df['Species'].astype('category').cat.codes

    def PlotarGrafico(self):
        """
        Plota um gráfico de dispersão para as variáveis SepalLengthCm e SepalWidthCm.
        """
        sns.scatterplot(data=self.df, x='SepalLengthCm', y='SepalWidthCm', hue='Species', palette='viridis')
        plt.title("Gráfico de Dispersão - Sepal Length vs Sepal Width")
        plt.xlabel("Comprimento da Sépala (cm)")
        plt.ylabel("Largura da Sépala (cm)")
        plt.legend(title="Espécies")
        plt.show()

    def DividirDados(self):
        """
        Divide os dados em conjuntos de treino e teste.
        """
        X = self.df.drop("Species", axis=1)
        y = self.df["Species"]

        # Usa train_test_split para dividir os dados
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print("Dados divididos em treino e teste.")

    def TreinarAvaliarModelos(self):
        """
        Treina e avalia os modelos SVM, Regressão Linear e Random Forest, comparando seus desempenhos.
        """
        # Treinamento do modelo SVM
        self.svm_model.fit(self.X_train, self.y_train)
        svm_predictions = self.svm_model.predict(self.X_test)
        svm_accuracy = accuracy_score(self.y_test, svm_predictions)
        print(f"Acurácia do modelo SVM: {svm_accuracy:.2f}")

        # Treinamento do modelo de Regressão Linear
        self.lr_model.fit(self.X_train, self.y_train)
        lr_predictions = self.lr_model.predict(self.X_test)

        # Convertendo previsões contínuas para classes arredondadas para calcular a acurácia
        lr_predictions_rounded = lr_predictions.round()
        lr_accuracy = accuracy_score(self.y_test, lr_predictions_rounded)
        lr_mse = mean_squared_error(self.y_test, lr_predictions)
        print(f"Acurácia (com previsões arredondadas) do modelo de Regressão Linear: {lr_accuracy:.2f}")
        print(f"Erro Médio Quadrado do modelo de Regressão Linear: {lr_mse:.2f}")

        # Treinamento do modelo Random Forest
        self.rf_model.fit(self.X_train, self.y_train)
        rf_predictions = self.rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_predictions)
        print(f"Acurácia do modelo Random Forest: {rf_accuracy:.2f}")

        # Comparação dos resultados
        print("\nResumo da Comparação:")
        print(f"SVM Acurácia: {svm_accuracy:.2f}")
        print(f"Regressão Linear Acurácia: {lr_accuracy:.2f}")
        print(f"Random Forest Acurácia: {rf_accuracy:.2f}")

        best_model = max((svm_accuracy, "SVM"), (lr_accuracy, "Regressão Linear"), (rf_accuracy, "Random Forest"))
        print(f"\nMelhor modelo: {best_model[1]} com acurácia de {best_model[0]:.2f}")

    def Train(self, path):
        """
        Função principal para o fluxo de treinamento e avaliação dos modelos.
        """
        self.CarregarDataset(path)
        self.TratamentoDeDados()
        self.PlotarGrafico()  # Chamando o método para exibir o gráfico
        self.DividirDados()   # Dividindo os dados em treino e teste
        self.TreinarAvaliarModelos()  # Comparando os modelos


# Exemplo de uso
path = '/iris.data'
modelo = ModeloComparacao()
modelo.Train(path)

#    SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
# 0            5.1           3.5            1.4           0.2  Iris-setosa
# 1            4.9           3.0            1.4           0.2  Iris-setosa
# 2            4.7           3.2            1.3           0.2  Iris-setosa
# 3            4.6           3.1            1.5           0.2  Iris-setosa
# 4            5.0           3.6            1.4           0.2  Iris-setosa


# Dados divididos em treino e teste.
# Acurácia do modelo SVM: 1.00
# Acurácia (com previsões arredondadas) do modelo de Regressão Linear: 1.00
# Erro Médio Quadrado do modelo de Regressão Linear: 0.04
# Acurácia do modelo Random Forest: 1.00

# Resumo da Comparação:
# SVM Acurácia: 1.00
# Regressão Linear Acurácia: 1.00
# Random Forest Acurácia: 1.00
#Porém dado que temos poucos dados o modelo SVM é melhor para o treinamento do que os outros modelos
# Melhor modelo: SVM com acurácia de 1.00
