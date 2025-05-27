# rna-classificacao-inteligencia-artificial
Projeto de classificação com Redes Neurais Artificiais usando TensorFlow – atividade prática da disciplina de IA..

Este projeto foi desenvolvido como parte das atividades práticas da disciplina de Inteligência Artificial do curso de Análise e Desenvolvimento de Sistemas.

## 💡 Objetivo

Construir e treinar uma Rede Neural Artificial (RNA) para resolver um problema de classificação binária, a partir de um dataset com atributos numéricos.

## 🧰 Tecnologias e Bibliotecas

- Python 3.x  
- pandas, numpy  
- scikit-learn (`train_test_split`, `StandardScaler`)  
- tensorflow.keras (Sequential, Dense, Adam)  
- matplotlib.pyplot (visualização)

## 📊 Estrutura da Rede Neural

- Camada 1: 16 neurônios, ativação ReLU  
- Camada 2: 8 neurônios, ativação ReLU  
- Saída: 1 neurônio, ativação Sigmoid  
- Otimizador: Adam  
- Função de perda: Binary Crossentropy

## 📈 Resultados

- Média de acurácia no teste: **91,67%**  
- Treinamento estável, com pequenas oscilações abaixo de 2%  
- Aplicação de boas práticas: normalização, estratificação, prevenção de overfitting  

## 📷 Exemplo de visualização

![Curva de aprendizado](resultados/grafico_aprendizado.png)

## 🧪 Código

O código principal está em `rede_neural.py`. Basta executar após instalar as bibliotecas com:

```bash
pip install -r requirements.txt
python rede_neural.py
