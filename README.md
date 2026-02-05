# Projects — Data Analytics / ML Portfolio

Начинающий аналитик данных / ML-аналитик.  
Работаю с Python и SQL: предобработка данных, EDA, проверка гипотез, A/B, построение признаков, обучение и оценка моделей (классификация / регрессия).  
Интересуюсь задачами табличного ML, NLP, временных рядов и CV (PyTorch / TensorFlow).

---

## Tech stack
- **Python:** pandas, numpy, scikit-learn, matplotlib  
- **ML:** classification / regression / clustering, метрики (Accuracy, F1, ROC-AUC, RMSE, MAE), работа с дисбалансом  
- **NLP:** TF-IDF, BERT/Transformers, embeddings, prompt-based анализ (LLM)  
- **CV:** PyTorch / torchvision (transfer learning), TensorFlow  
- **Statistics / A/B:** проверка гипотез, p-value, Bootstrap  
- **SQL / DB:** SQL, MySQL  
- **BI:** Power BI, Excel  
- **Tools:** Git, Jupyter Notebook / Google Colab  

---

## Links
- **GitHub profile:** https://github.com/belkovskyy  
- **Certificates:** https://disk.yandex.ru/d/41_6sZ9RMLaq-A  
- **Courses & documents:** https://disk.yandex.ru/d/7wBf0Hh009S0lQ  

---

## Repository structure

Проекты сгруппированы по темам (внутри каждой папки — свой README с описанием):

```text
ML_projects/
├─ 01_data_preprocessing/
├─ 02_eda/
├─ 03_stats_ab/
├─ 04_ml_classic/
├─ 05_ml_business/
├─ 06_nlp/
└─ 07_cv/
```

---

## Projects

### 01_data_preprocessing
- **Исследование надежности заемщиков** — очистка данных, пропуски, типы, категориальные признаки (`ML_projects/01_data_preprocessing/Исследование_надежности_заемщиков.ipynb`)

---

### 02_eda
- **Анализ продаж игр и прогноз трендов на 2017** — EDA, портреты пользователей по регионам, гипотезы (`ML_projects/02_eda/Анализ продаж игр и прогноз трендов на 2017.ipynb`)
- **Исследование объявлений о продаже квартир** — EDA рынка недвижимости, аномалии, факторы цены (`ML_projects/02_eda/Исследование объявлений о продаже квартир.ipynb`)

---

### 03_stats_ab
- **Анализ тарифных планов компании «Мегалайн»** — статистический анализ, проверка гипотез, p-value (`ML_projects/03_stats_ab/Анализ тарифных планов компании «Мегалайн».ipynb`)

---

### 04_ml_classic
- **Рекомендательная система тарифов** — классификация, подбор модели и гиперпараметров (`ML_projects/04_ml_classic/Рекомендательная система тарифов мобильной связи на основе машинного обучения.ipynb`)
- **Прогнозирование оттока клиентов банка** — классификация, дисбаланс классов, F1 и ROC-AUC (`ML_projects/04_ml_classic/Прогнозирование оттока клиентов банка на основе машинного обучения.ipynb`)
- **Оценка цен на недвижимость (СПБ)** — регрессия, ансамбли, feature engineering (`ML_projects/04_ml_classic/Оценка цен на недвижимость (СПБ).ipynb`)
- **Оценка цен на недвижимость (Kaggle House Prices)** — регрессия, сравнение моделей, инженерия признаков (`ML_projects/04_ml_classic/Оценка цен на недвижимость (Kaggle House Prices).ipynb`)

---

### 05_ml_business
- **Выбор региона для бурения скважин (Bootstrap)** — оценка прибыли и рисков, доверительные интервалы (`ML_projects/05_ml_business/Выбор региона для бурения скважин, оценка прибыли и рисков (Bootstrap).ipynb`)
- **Прогноз истощения абсорбента NaOH (промышленная аналитика)** — регрессия + бизнес-интерпретация результата (`ML_projects/05_ml_business/Прогноз истощения абсорбента NaOH в абсорбере (промышленная аналитика).ipynb`)

---

### 06_nlp
- **Классификация токсичных комментариев (TF-IDF)** — baseline, подбор порога, тестовые метрики (`ML_projects/06_nlp/Классификация токсичных комментариев (TF-IDF).ipynb`)
- **Классификация токсичных комментариев (BERT)** — улучшение baseline, тестовые метрики (`ML_projects/06_nlp/Классификация токсичных комментариев (BERT).ipynb`)
- **NLP-аналитика отзывов** — тональность/аспекты/summary на LLM (Qwen), прикладной пайплайн (`ML_projects/06_nlp/NLP-аналитика отзывов - тональность, аспекты и резюме (LLM, Qwen).ipynb`)
- **Кластеризация товаров продавца** — embeddings + UMAP + HDBSCAN/DBSCAN, метрики кластеризации (`ML_projects/06_nlp/Кластеризация товаров продавца.ipynb`)

---

### 07_cv
- **Определение возраста покупателей по фото** — CV-регрессия, transfer learning, метрика MAE (`ML_projects/07_cv/Определение возраста покупателей.ipynb`)

---

## Roadmap
Следующие проекты реализованы и находятся в процессе дооформления/переноса в репозиторий:
- **Time series:** прогноз заказов такси (forecasting, RMSE)
- **Tabular ML (advanced):** телеком / скоринг (CatBoost/LightGBM, feature engineering)
- **Hackathons:** скоринг / RAG / рекомендации (оформление репо: README + baseline + итог)
- **BI:** отдельный репозиторий с дашбордами Power BI (скриншоты + описание метрик)

---

## Notes
Если в проекте нельзя публиковать исходные данные (условия курса/хакатона),  
я оставляю код, описание решения и инструкции по воспроизведению.

