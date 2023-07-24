import os
import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QPushButton, QFileDialog, QLabel
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from PyQt5.QtWidgets import QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import QRect
from matplotlib.dates import DayLocator, DateFormatter
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QMessageBox

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Создание выпадающего списка для выбора модели
        self.model_combo_box = QComboBox(self)
        self.model_combo_box.setGeometry(QRect(20, 20, 250, 37))
        font = self.model_combo_box.font()
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.model_combo_box.setFont(font)
        self.model_combo_box.activated.connect(self.load_model)

        # Создание кнопки для выбора файла CSV
        self.csv_button = QPushButton("CSV", self)
        self.csv_button.setGeometry(QRect(315, 20, 100, 37))
        font = self.csv_button.font()
        font.setPointSizeF(font.pointSizeF() * 1.25)
        self.csv_button.setFont(font)
        self.csv_button.clicked.connect(self.load_csv)

        # Создание метки для отображения выбранного файла CSV
        self.csv_label = QLabel(self)
        self.csv_label.setGeometry(QRect(20, 70, 375, 27))
        font = self.csv_label.font()
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.csv_label.setFont(font)

        # Создание кнопки для запуска прогнозирования
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setGeometry(QRect(20, 120, 100, 37))
        font = self.predict_button.font()
        font.setPointSizeF(font.pointSizeF() * 1.25)
        self.predict_button.setFont(font)
        self.predict_button.clicked.connect(self.predict)

        # Создание кнопки для прогноза на следующий день
        self.forecast_button = QPushButton("Forecast Next Day", self)
        self.forecast_button.setGeometry(QRect(140, 120, 187, 37))
        font = self.forecast_button.font()
        font.setPointSizeF(font.pointSizeF() * 1.25)
        self.forecast_button.setFont(font)
        self.forecast_button.clicked.connect(self.forecast_next_day)

        # Создание метки для отображения значения MSE
        self.mse_label = QLabel(self)
        self.mse_label.setGeometry(QRect(333, 120, 75, 37))
        font = self.mse_label.font()
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.mse_label.setFont(font)

        # Создание кнопки для сохранения прогнозов в файл CSV
        self.save_button = QPushButton("Save", self)
        self.save_button.setGeometry(QRect(20, 165, 100, 37))
        font = self.save_button.font()
        font.setPointSizeF(font.pointSizeF() * 1.25)
        self.save_button.setFont(font)
        self.save_button.clicked.connect(self.save_csv)

        # Создание метки для отображения пути к файлу, в который будут сохранены прогнозы
        self.save_label = QLabel(self)
        self.save_label.setGeometry(QRect(125, 165, 290, 37))
        font = self.save_label.font()
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.save_label.setFont(font)

        # Получение списка файлов .h5 в текущей директории
        model_files = [f for f in os.listdir() if f.endswith(".h5")]

        # Добавление имен файлов в выпадающий список
        for model_file in model_files:
            self.model_combo_box.addItem(model_file)

        # Установка размера окна
        self.setGeometry(100, 100, 450, 220)
        self.setWindowTitle("Quote Prediction Application")

        # ...existing code...
       # ...existing functions...
       
    def forecast_next_day(self):
        # Создание последовательности для прогноза на следующий день
        seq_length = 50
        X, y = self.create_sequences(seq_length=seq_length)
        last_sequence = X[-1]

        # Загрузка сохраненной модели
        model_file = self.model_combo_box.currentText()
        model = load_model(model_file)

        # Получение прогноза на следующий день
        y_pred = model.predict(last_sequence.reshape(1, seq_length, 1))

        # Обратное масштабирование прогноза
        scaler = MinMaxScaler()
        scaler.fit_transform(self.data[["Open"]])
        prediction = scaler.inverse_transform(y_pred)[0][0]

        # Получение последнего месяца данных
        last_month_data = self.data.set_index('Date')
        last_month_data.index = pd.to_datetime(last_month_data.index)

        # Интерполяция данных для заполнения пропущенных значений
        last_month_data = last_month_data.resample('D').last().interpolate()

        # Выбор последних 30 дней данных
        last_month_data = last_month_data.tail(30)

        # Добавление даты прогноза в индекс последней строки
        last_date = pd.to_datetime(last_month_data.index[-1])
        next_day = last_date + pd.Timedelta(days=1)

        # Создание нового DataFrame с одной строкой для прогноза на следующий день
        forecast_data = pd.DataFrame(
            {
                "Open": [prediction]
            },
            index=[next_day]
        )

        # Объединение последнего месяца данных и прогноза на следующий день
        last_month_data = pd.concat([last_month_data, forecast_data])

        # Преобразование индекса в объекты datetime
        last_month_data.index = pd.to_datetime(last_month_data.index)


        # Создание нового окна для вывода прогноза на следующий день
        self.forecast_window = QWidget()
        self.forecast_window.setGeometry(100, 100, 800, 600)
        self.forecast_window.setWindowTitle("Forecast for Next Day")

        # Создание графика для отображения последнего месяца и прогноза на следующий день
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self.forecast_window)

        # Добавление графика на окно прогноза
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.forecast_window.setLayout(layout)

        # Создание нового объекта Axes для графика прогноза
        ax2 = self.figure.add_subplot(111)

        # Построение графика последнего месяца и прогноза на следующий день
        ax2.plot(last_month_data.index, last_month_data["Open"], label="Last Month's Data", linestyle='-')
        ax2.plot(next_day, prediction, 'ro', label="Forecast for Next Day")
        ax2.legend()

        # Установка интервала между датами
        day_locator = DayLocator(interval=7)
        ax2.xaxis.set_major_locator(day_locator)
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

        # Добавление сетки на график
        ax2.grid(True)

        # Поворот дат на оси X
        plt.xticks(rotation=20)

        # Добавление текстового блока с прогнозом на следующий день
        ax2.text(next_day - pd.Timedelta(days=2), prediction - 10, f"{prediction:.2f}", fontsize=16, ha='center', va='center')

        # Отображение окна прогноза на экране
        self.forecast_window.show()

    def load_model(self):
        # Загрузка выбранной модели
        model_file = self.model_combo_box.currentText()
        self.model = load_model(model_file)

    def load_csv(self):
        # Открытие диалога выбора файла CSV
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("CSV files (*.csv)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            # Получение пути к выбранному файлу CSV
            csv_file = file_dialog.selectedFiles()[0]

            # Отображение выбранного файла CSV на метке
            self.csv_label.setText(csv_file)

            # Загрузка данных из файла CSV
            self.data = pd.read_csv(csv_file)

    def create_sequences(self, seq_length):
        # Нормализация входных данных
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(self.data[["Open"]])

        # Создание последовательностей
        X = []
        y = []
        for i in range(seq_length, len(data_scaled)):
            X.append(data_scaled[i-seq_length:i])
            y.append(data_scaled[i])
        X = np.array(X).reshape(-1, seq_length, 1)
        y = np.array(y)
        return X, y

    def predict(self):
        if os.path.exists('file.csv'):
            # загрузка файла и выполнение предсказания

            # Создание последовательностей
            X_test, y_test = self.create_sequences(seq_length=50)

            # Загрузка сохраненной модели
            model_file = self.model_combo_box.currentText()
            model = load_model(model_file)

            # Получение прогнозов на основе выбранной модели
            y_pred = model.predict(X_test)

            # Обратное масштабирование прогнозов
            scaler = MinMaxScaler()
            scaler.fit_transform(self.data[["Open"]])
            predictions = scaler.inverse_transform(y_pred)

            # Сохранение прогнозов для дальнейшего использования
            self.predictions = pd.DataFrame(predictions, columns=["Open"])

            # Рассчет среднеквадратичной ошибки (MSE) и среднеквадратичного отклонения (RMSE)
            mse = mean_squared_error(self.data["Open"].iloc[-len(predictions):], predictions)
            rmse = np.sqrt(mse)

            # Создание нового окна для вывода графика прогнозов
            fig = plt.figure(figsize=(8, 6))
            fig.canvas.setWindowTitle("Predictions")

            # Отображение графика прогнозов
            ax = fig.add_subplot(111)
            ax.plot(self.data["Date"].iloc[-200:], self.data["Open"].iloc[-200:], label="Actual")
            ax.plot(self.data["Date"].iloc[-200:], self.predictions[-200:], label="Predicted")
            ax.legend()

            # Добавление сетки на график
            ax.grid(True)

            # Установка интервала между датами
            day_locator = DayLocator(interval=30)
            ax.xaxis.set_major_locator(day_locator)
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

            # Поворот дат на оси X
            plt.xticks(rotation=20)

            # Отображение значения MSE на экране
            self.mse_label.setText(f"MSE: {mse:.2f}")

            # Отображение окна с графиком прогнозов на экране
            plt.show()
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("Please load file")
            msg_box.exec_()

    def save_csv(self):
        # Открытие диалога выбора файла для сохранения прогнозов
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("CSV files (*.csv)")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        if file_dialog.exec_():
            # Получение пути к файлу для сохранения прогнозов
            csv_file = file_dialog.selectedFiles()[0]

            # Сохранение прогнозов в файл CSV
            self.predictions.to_csv(csv_file, index=False)

            # Отображение пути к файлу на метке
            self.save_label.setText(csv_file)

         
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Создание экземпляра основного окна
    window = MainWindow()

    # Установка тёмной темы оформления
    app.setStyle("Fusion")
    palette = app.palette()

    # Установка цветовой схемы для приложения
    palette.setColor(palette.Window, QColor(30, 30, 30))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(20, 20, 20))
    palette.setColor(palette.AlternateBase, QColor(50, 50, 50))
    palette.setColor(palette.ToolTipBase, QColor(30, 30, 30))
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(50, 50, 50))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Highlight, QColor(80, 80, 80))
    palette.setColor(palette.HighlightedText, Qt.white)

    # Установка заголовка для основного окна
    window.setWindowTitle("Quote Prediction Application")

    app.setPalette(palette)

    # Отображение основного окна
    window.show()
    sys.exit(app.exec_())