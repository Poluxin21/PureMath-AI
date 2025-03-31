# ANOTAÇÕES DE APRENDIZADO

## Neurônios e Matrizes
- **Neuronios** são representados como **vetores**.
- **Matrizes** são utilizadas como **pesos dos neurônios**.

## Cálculo de uma Rede Neural
A rede neural é composta por três etapas principais:
1. **Feed Forward**: Passagem das entradas pelas camadas da rede até gerar a saída.
2. **Backpropagation**: Ajuste dos pesos baseado no erro da rede.
3. **MSE (Mean Squared Error)**: Função de erro que calcula a média dos quadrados das diferenças entre as saídas reais e as previstas.

## Algoritmos de Randomização para Pesos de Neurônios
- **Xavier Normal**: Método de inicialização de pesos que ajuda a evitar problemas de gradientes vanishing/exploding.

---

## Trabalhando com Imagens
- **Imagens** são representadas como **matrizes de pixels**.
- **Dica para iniciar**: Comece com imagens pequenas (por exemplo, 28x28 pixels, como o MNIST de dígitos escritos à mão).

---

# MELHORIAS PARA IMPLEMENTAR

## Implementar Camadas Convolucionais (CNN)
Em reconhecimento de imagens, uma **rede neural convolucional (CNN)** funciona melhor do que uma rede totalmente conectada.

- Use **filtros (kernels)** para detectar padrões, como bordas e texturas.
- A operação matemática principal aqui é a **convolução**.

## Pooling (Redução de Dimensão)
Após a convolução, usamos **Max Pooling** ou **Average Pooling** para reduzir a dimensão da matriz sem perder informações importantes.

- **Max Pooling** pega o maior valor em uma região (por exemplo, 2x2 pixels).
- **Average Pooling** faz a média dos valores.

Esses métodos reduzem o custo computacional e ajudam a melhorar a **generalização**.

## Classificação com Softmax
- O **Softmax** é utilizado para transformar as saídas da rede neural em probabilidades, permitindo que a rede faça classificações baseadas nas probabilidades de cada classe.

---

# Processamento de Linguagem Natural (NLP)

## Modelos de Texto
1. **Bag of Words (BoW)**: Conta quantas vezes cada palavra aparece no texto.
2. **Word Embeddings (Vetores de Palavras)**: Representa palavras em um espaço de vetores, permitindo capturar semelhanças semânticas.

## Implementar um Perceptron para Classificação de Texto
Se quiser classificar frases (por exemplo: detectar sentimentos), pode usar uma rede neural simples:
- **Entrada**: vetor com as palavras da frase.
- **Saída**: **softmax** com classes como "positivo", "negativo", "neutro".

## Implementar RNN ou LSTM
Se quiser processar textos sequenciais, como prever a próxima palavra ou entender a sequência de um texto, será necessário usar uma **Rede Neural Recorrente (RNN)** ou **LSTM (Long Short-Term Memory)**.
