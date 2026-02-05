# 06_nlp — NLP / LLM Projects

Проекты по обработке естественного языка: от классического ML на **TF-IDF** до **BERT** и прикладных пайплайнов на **LLM** (анализ отзывов, кластеризация карточек).

---

## Projects

| Проект | Файл | Коротко | Метрики (TEST) |
|---|---|---|---|
| Toxic comments (TF-IDF + LogReg) | `Классификация токсичных комментариев (TF-IDF).ipynb` | Baseline для детекции токсичности | **F1 0.8139**, **ROC-AUC 0.9791**, **PR-AUC 0.8970** |
| Toxic comments (BERT) | `Классификация токсичных комментариев (BERT).ipynb` | Улучшение baseline за счёт BERT | **F1 0.8423**, **ROC-AUC 0.9872**, **PR-AUC 0.9269** |
| Reviews NLP analytics (LLM/Qwen) | `NLP-аналитика отзывов - тональность, аспекты и резюме (LLM, Qwen).ipynb` | Тональность/аспекты + краткие резюме отзывов | — |
| Product clustering | `Кластеризация товаров продавца (1).ipynb` | Семантическая кластеризация карточек | — |

---

## 1) Toxic comments — TF-IDF + Logistic Regression (baseline)
**Файл:** `Классификация токсичных комментариев (TF-IDF).ipynb`

**Что сделано:** предобработка текста → TF-IDF → Logistic Regression → подбор порога под F1 → оценка на test.

**TEST метрики:** **F1 0.8139**, **ROC-AUC 0.9791**, **PR-AUC 0.8970**

---

## 2) Toxic comments — BERT
**Файл:** `Классификация токсичных комментариев (BERT).ipynb`

**Что сделано:** пайплайн на BERT/Transformers → вероятности → подбор порога под F1 → оценка на test.

**TEST метрики:** **F1 0.8423**, **ROC-AUC 0.9872**, **PR-AUC 0.9269**

**Сравнение (BERT − TF-IDF+LogReg) на TEST:**  
- **F1:** +0.0284  
- **ROC-AUC:** +0.0081  
- **PR-AUC:** +0.0299  

---

## 3) NLP-аналитика отзывов — тональность, аспекты и резюме (LLM/Qwen)
**Файл:** `NLP-аналитика отзывов - тональность, аспекты и резюме (LLM, Qwen).ipynb`

Пайплайн для прикладного анализа отзывов с помощью LLM:
- разметка **sentiment** (positive/neutral/negative)
- разметка **aspect** (например: quality/price/delivery/service/other)
- генерация короткого **summary** для каждого отзыва  
+ сводные отчёты и примеры разметки.

---

## 4) Кластеризация товаров продавца
**Файл:** `Кластеризация товаров продавца (1).ipynb`

Семантическая кластеризация карточек товаров:
- подготовка текстов карточек
- эмбеддинги (Sentence-Transformers)
- снижение размерности (UMAP/t-SNE для визуализации)
- кластеризация (DBSCAN/HDBSCAN)
- интерпретация кластеров и примеры групп.

---

## Stack (short)
- **Python**, pandas, numpy, scikit-learn, matplotlib  
- **NLP:** TF-IDF, Transformers/BERT  
- **LLM:** Qwen (prompt-pipeline)  
- **Clustering:** embeddings, UMAP, DBSCAN/HDBSCAN


## Как запускать

- Для воспроизведения обычно достаточно Python-стека: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.
- Для отдельных ноутбуков дополнительно:
  - `transformers`, `torch` (BERT)
  - `sentence-transformers`, `umap-learn`, `hdbscan` (кластеризация)
  - LLM-части требуют доступной модели/окружения (локально/в облаке).


