import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def filter_data(path):
    df = pd.read_csv(path, sep=',')

    work_data = df.values

    indices = []
    for i in range(0, len(work_data)):
        for j in range(0, len(work_data[i])):
            if work_data[i][j] == '?':
                indices.append((i, j))

    for index in indices:
        df.loc[index[0], df.columns[index[1]]] = 0

    df.to_csv(path, index=False)


if __name__ == "__main__":

    # Фильтрация данных
    filter_data('./dataset.csv')
    raw_data = pd.read_csv("dataset.csv")

    # Отсекание некоррелируемых столбцов
    dropped_data = raw_data.drop(['YearEnd', 'ManagerExp', 'TeamExp', 'Project'], axis=1)

    scaler = StandardScaler()
    scaler.fit(dropped_data)
    scaled_features = scaler.transform(dropped_data)
    scaled_data = pd.DataFrame(scaled_features, columns=dropped_data.columns)

    x = scaled_data
    y = dropped_data

    # Построение тепловой карты
    data = raw_data.corr()
    sns.heatmap(data.corr(), annot=True)
    plt.show()

    # Разделение на обучающую и тестовую выборки
    x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size=0.3)

    # Создание и обучение модели KNN
    model_KNN = KNeighborsRegressor(n_neighbors=2)
    model_KNN.fit(x_training_data, y_training_data)
    predictions_KNN = model_KNN.predict(x_test_data)

    # Вывод результатов точности алгоритма (Knn)
    print('Средняя абсолютная ошибка (Knn): ', metrics.mean_absolute_error(y_test_data, predictions_KNN))
    print('Средняя квадратичная ошибка (Knn): ', metrics.mean_squared_error(y_test_data, predictions_KNN))

    # Создание и обучение модели (случайный лес)
    model_RF = RandomForestRegressor(n_estimators=100)
    model_RF.fit(x_training_data, y_training_data)
    predictions_RF = model_RF.predict(x_test_data)

    # Вывод результатов точности алгоритма (случайный лес)
    print('Средняя абсолютная ошибка (случайный лес):', metrics.mean_absolute_error(y_test_data, predictions_RF))
    print('Средняя квадратичная ошибка (случайный лес):', metrics.mean_squared_error(y_test_data, predictions_RF))
