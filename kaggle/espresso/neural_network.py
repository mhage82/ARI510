from sklearn.neural_network import MLPClassifier

from run1 import SEED, run


model = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, max_iter=1500, random_state=SEED)
run(model, "run2_neural_network.csv")
