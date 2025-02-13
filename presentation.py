import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    # Пример содержимого с Markdown-разделителями (---)
    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    **Цель проекта:** Разработка модели для предсказания отказов оборудования  
    **Задача:** Определить, произойдет ли отказ (Target = 1) или нет (Target = 0)
    ---
    ## Датасет
    - **Название:** AI4I 2020 Predictive Maintenance Dataset
    - **Характеристики:** 10 000 записей, 14 признаков
    - **Описание:** Данные о температуре, скорости, крутящем моменте и износе инструмента
    ---
    ## Предобработка данных
    - Удаление лишних столбцов (например, уникальных идентификаторов)
    - Преобразование категориальных признаков в числовые
    - Масштабирование числовых данных для улучшения сходимости модели
    ---
    ## Обучение моделей
    - **Logistic Regression**
    - **Random Forest**
    - **XGBoost**
    - **SVM**
    ---
    ## Результаты
    - **Accuracy:** 97.4%
    - **Classification Report для отказов:** 
      - Precision: 0.67, Recall: 0.26, F1-score: 0.38  
      *(Указывают на дисбаланс классов)*
    ---
    ## Решения по улучшению
    - Применение методов балансировки (например, SMOTE)
    - Настройка порогов классификации и cost-sensitive learning
    ---
    ## Заключение
    - Модель успешно определяет отсутствие отказов
    - Требуется улучшение обнаружения отказов
    ---
    ## Благодарности
    Спасибо за внимание!
    """

    # Параметры презентации (выбор темы, т.д.)
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league",
                                      "beige", "sky", "night",
                                      "serif", "simple", "solarized"])
        transition = st.selectbox("Переход слайдов",
                                  ["slide", "convex", "concave", "zoom", "none"])

    rs.slides(
        presentation_markdown,
        height=600,
        theme=theme,
        config={"transition": transition},
        markdown_props={"data-separator-vertical": "^--$"},
    )