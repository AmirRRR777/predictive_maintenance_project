import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # 1) Загрузка датасета через интерфейс
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # ----- Предобработка -----
        data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], inplace=True)
        le = LabelEncoder()
        data['Type'] = le.fit_transform(data['Type'])

        # Разделение на X и y
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']

        # Масштабирование (опционально)
        num_cols = ['Air temperature [K]',
                    'Process temperature [K]',
                    'Rotational speed [rpm]',
                    'Torque [Nm]',
                    'Tool wear [min]']
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ----- Обучение модели (например, Logistic Regression) -----
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # ----- Оценка -----
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.subheader("Результаты обучения модели")
        st.write(f"**Accuracy**: {acc:.3f}")

        # Confusion Matrix heatmap
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.text("Classification Report:")
        st.text(report)

        # ----- Предсказание на новых данных -----
        st.subheader("Предсказание на новых данных")
        with st.form("prediction_form"):
            # Примеры полей
            input_type = st.selectbox("Тип продукта (L=0, M=1, H=2)", [0, 1, 2])
            input_air_temp = st.number_input("Air temperature [K]", value=300.0)
            input_process_temp = st.number_input("Process temperature [K]", value=310.0)
            input_speed = st.number_input("Rotational speed [rpm]", value=1500)
            input_torque = st.number_input("Torque [Nm]", value=40.0)
            input_wear = st.number_input("Tool wear [min]", value=0)

            submit_button = st.form_submit_button("Предсказать")

            if submit_button:
                # Создаем DataFrame из введенных данных
                input_data = pd.DataFrame({
                    'Type': [input_type],
                    'Air temperature [K]': [input_air_temp],
                    'Process temperature [K]': [input_process_temp],
                    'Rotational speed [rpm]': [input_speed],
                    'Torque [Nm]': [input_torque],
                    'Tool wear [min]': [input_wear]
                })

                # Не забываем масштабировать так же, как обучающие данные
                input_data[num_cols] = scaler.transform(input_data[num_cols])

                # Получаем предсказание
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[:, 1]

                st.write(f"**Предсказание (0=Нет отказа, 1=Отказ)**: {prediction[0]}")
                st.write(f"**Вероятность отказа**: {prediction_proba[0]:.3f}")
