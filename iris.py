from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# 1. Завантаження датасету Iris
iris = load_iris()

X = iris.data
y = iris.target

# Назви класів
classes = iris.target_names

# 2. Розділення даних на навчальні та тестові
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Створення та навчання моделі KNN
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

# 4. Прогнозування
y_pred = knn.predict(X_test)

# 5. Підрахунок правильних прогнозів
correct = 0

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct += 1

# Accuracy
accuracy = correct / len(y_test)

print("===== ACCURACY =====")
print("Кількість правильних прогнозів:", correct)
print("Кількість тестових даних:", len(y_test))
print("Accuracy:", round(accuracy, 4))


# 6. Матриця помилок
matrix = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(matrix)


# 7. Метрики для кожного класу
print("\n===== МЕТРИКИ КЛАСИФІКАЦІЇ =====")

for i in range(len(classes)):

    # TP, FP, FN, TN
    TP = matrix[i][i]

    FP = sum(matrix[:, i]) - TP

    FN = sum(matrix[i, :]) - TP

    TN = matrix.sum() - TP - FP - FN

    # Precision
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    # Recall
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # F1-score
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    # Виведення результатів
    print("\nКлас:", classes[i])

    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))