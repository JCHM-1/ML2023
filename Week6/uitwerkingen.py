import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# OPGAVE 1a
def plot_image(img, label):
    # Deze methode krijgt een matrix mee (in img) en een label dat correspondeert met het 
    # plaatje dat in de matrix is weergegeven. Zorg ervoor dat dit grafisch wordt weergegeven.
    # Maak gebruik van plt.cm.binary voor de cmap-parameter van plt.imgshow.

    # YOUR CODE HERE
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(label)
    plt.colorbar()
    plt.grid(False)
    plt.show()


# OPGAVE 1b
def scale_data(X):
    # Deze methode krijgt een matrix mee waarin getallen zijn opgeslagen van 0..m, en hij 
    # moet dezelfde matrix retourneren met waarden van 0..1. Deze methode moet werken voor 
    # alle maximale waarde die in de matrix voorkomt.
    # Deel alle elementen in de matrix 'element wise' door de grootste waarde in deze matrix.

    # YOUR CODE HERE
    max_value = np.amax(X)
    scaled_X = X / max_value

    return scaled_X


# OPGAVE 1c
def build_model():
    # Deze methode maakt het keras-model dat we gebruiken voor de classificatie van de mnist
    # dataset. Je hoeft deze niet abstract te maken, dus je kunt er van uitgaan dat de input
    # layer van dit netwerk alleen geschikt is voor de plaatjes in de opgave (wat is de 
    # dimensionaliteit hiervan?).
    # Maak een model met een input-laag, een volledig verbonden verborgen laag en een softmax
    # output-laag. Compileer het netwerk vervolgens met de gegevens die in opgave gegeven zijn
    # en retourneer het resultaat.

    # Het staat je natuurlijk vrij om met andere settings en architecturen te experimenteren.

    # YOUR CODE HERE
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input-laag voor 28x28 plaatjes
        tf.keras.layers.Dense(128, activation=tf.nn.relu),  # Verborgen laag met ReLU-activatie
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Output-laag met softmax-activatie voor classificatie
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# OPGAVE 2a
def conf_matrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix:
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
    
    # YOUR CODE HERE
    confusion = tf.math.confusion_matrix(labels, pred)
    return confusion

    

# OPGAVE 2b
def conf_els(conf, labels): 
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)
 
    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html
 
    # YOUR CODE HERE
    conf = np.array(conf)  
    conf = np.array(conf, dtype=int)  

    conf_els_list = []  

    for i in range(len(labels)):
        category = labels[i]  # Huidige categorie
        tp = conf[i, i]  # True Positives
        fp = np.sum(conf[:, i]) - tp  # False Positives
        fn = np.sum(conf[i, :]) - tp  # False Negatives
        tn = np.sum(np.delete(np.delete(conf, i, axis=0), i, axis=1))  # True Negatives

        conf_els_list.append((category, tp, fp, fn, tn))

    return conf_els_list

# OPGAVE 2c
def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    # Bereken de totale waarden voor TP, FP, FN en TN over alle labels
    total_tp = sum(tp for _, tp, _, _, _ in metrics)
    total_fp = sum(fp for _, _, fp, _, _ in metrics)
    total_fn = sum(fn for _, _, _, fn, _ in metrics)
    total_tn = sum(tn for _, _, _, _, tn in metrics)

    # Bereken de metrieken op basis van de totale waarden
    tpr = total_tp / (total_tp + total_fn)  # True Positive Rate (Recall)
    ppv = total_tp / (total_tp + total_fp)  # Positive Predictive Value (Precision)
    tnr = total_tn / (total_tn + total_fp)  # True Negative Rate
    fpr = total_fp / (total_tn + total_fp)  # False Positive Rate

    return {'tpr': tpr, 'ppv': ppv, 'tnr': tnr, 'fpr': fpr}
