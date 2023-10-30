import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    image_matrix = np.reshape(nrVector, (20, 20), order='F')

    plt.matshow(image_matrix, cmap='gray')
    plt.show()
    

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.
    return 1 / (1 + np.exp(-z))



# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van 
    # y en m
    y = np.array(y)  # Zorg ervoor dat y een array is

    # De breedte van de matrix is max(y) + 1
    width = max(y) + 1

    # Maak arrays voor rijen, kolommen en gegevens
    rows = list(range(len(y))
    cols = [i - 1 for i in y]  # Zet y om naar een lijst met 0-based indices
    data = [1] * len(y)

    # Maak de csr_matrix en zet deze om naar een gewone array
    y_matrix = csr_matrix((data, (rows, cols)), shape=(len(y), width)).toarray()

    return y_matrix

    

# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predict_number(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    m, n = X.shape
    a1 = np.c_[np.ones((m, 1)), X]

    # Bereken de activatie in de tweede laag (verborgen laag)
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)

    # Voeg enen toe aan de matrix a2, dit is de input voor de laatste laag in het netwerk
    a2 = np.c_[np.ones((m, 1)), a2]

    # Bereken de output van het netwerk aan de buitenste laag
    z3 = a2.dot(Theta2.T)
    h = sigmoid(z3)

    return h




# ===== deel 2: =====
def compute_cost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix. 
    m, n = X.shape
    y_matrix = get_y_matrix(y, m)

    # Voer een voorwaartse propagatie uit om de voorspelling te krijgen
    h = predict_number(Theta1, Theta2, X)

    # Bereken de kostfunctie
    J = (-1 / m) * np.sum(y_matrix * np.log(h) + (1 - y_matrix) * np.log(1 - h))

    return J



# ==== OPGAVE 3a ====
def sigmoid_gradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.

    g = sigmoid(z) * (1 - sigmoid(z))
    return g

# ==== OPGAVE 3b ====
def nn_check_gradients(Theta1, Theta2, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

        m, n = X.shape
    y_matrix = get_y_matrix(y, m)
    Delta2 = np.zeros(Theta1.shape)
    Delta3 = np.zeros(Theta2.shape

    for i in range(m):
        # Stap 1: Neem een voorbeeld uit de dataset
        xi = X[i]
        yi = y_matrix[i]

        # Stap 2: Voer voorwaartse en achterwaartse propagatie uit
        a1 = np.c_[1, xi]
        z2 = a1.dot(Theta1.T)
        a2 = sigmoid(z2)
        a2 = np.c_[1, a2]
        z3 = a2.dot(Theta2.T)
        h = sigmoid(z3)

        # Stap 3: Bereken de fout aan de uitvoerlaag
        delta3 = h - yi

        # Stap 4: Bereken de fout aan de verborgen laag
        delta2 = delta3.dot(Theta2) * sigmoid_gradient(np.c_[1, z2])
        delta2 = delta2[0, 1:]  # Verwijder het extra bias term element

        # Stap 5: Update Delta-matrices
        Delta3 += delta3.reshape(-1, 1).dot(a2.reshape(1, -1))
        Delta2 += delta2.reshape(-1, 1).dot(a1.reshape(1, -1))

    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m

    return Delta2_grad, Delta3_grad
