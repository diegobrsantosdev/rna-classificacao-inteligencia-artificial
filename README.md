# RNA-classification-artificial-intelligence  
Classification project with Artificial Neural Networks using TensorFlow – practical activity for the AI course.

This project was developed as part of the practical activities of the Artificial Intelligence course in the Systems Analysis and Development program.

## 💡 Objective

Build and train an Artificial Neural Network (ANN) to solve a binary classification problem using a dataset with numerical attributes.

## 🧰 Technologies and Libraries

- Python 3.x  
- pandas, numpy  
- scikit-learn (`train_test_split`, `StandardScaler`)  
- tensorflow.keras (Sequential, Dense, Adam)  
- matplotlib.pyplot (visualization)

## 📊 Neural Network Architecture

- Layer 1: 16 neurons, ReLU activation  
- Layer 2: 8 neurons, ReLU activation  
- Output: 1 neuron, Sigmoid activation  
- Optimizer: Adam  
- Loss function: Binary Crossentropy

## 📈 Results

- Average test accuracy: **91.67%**  
- Stable training with minor fluctuations under 2%  
- Application of best practices: normalization, stratification, overfitting prevention  


## 🧪 Code

The main code is in `rede_neural.py`. Just run it after installing the libraries with:

```bash
pip install -r requirements.txt
python rede_neural.py
```

👨‍💻 Author
Diego Melo Bezerra dos Santos
🔗 github.com/diegobrsantosdev

