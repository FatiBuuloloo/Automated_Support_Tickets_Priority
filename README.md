# Automated Support Ticket Triage & Intelligent Priority Scoring

## Project Overview
Customer support teams often face thousands of daily tickets with varying degrees of urgency. This project implements an **automated triage system** that classifies incoming support tickets into specific departments (Queue) and determines their priority level. 

The system goes beyond simple classification by calculating a **Final Priority Score** that combines predicted urgency with real-time sentiment analysis. This allows businesses to automatically route low-stakes tickets to AI Chatbots while prioritizing high-urgency, negative-sentiment cases for human intervention.

**Business Objective:** Improve Customer Satisfaction Score (CSAT) and optimize operational costs by reducing human agent workload through automated decision-making.

---

## Technical Workflow

### 1. Data Preprocessing & Multilingual Support
- **Translation:** Leveraged `deep-translator` to handle German-language tickets, converting them to English to standardize the input for the NLP pipeline.
- **Sentiment Labeling:** Utilized the `cardiffnlp/twitter-roberta-base-sentiment-latest` model to extract emotional context (Positive, Neutral, Negative) from customer queries.

### 2. Feature Engineering (NLP)
- **Text Embedding:** Transformed ticket bodies into 768-dimensional dense vectors using the `SentenceTransformer ('all-mpnet-base-v2')`. This model captures the semantic meaning of the tickets far better than traditional TF-IDF approaches.

### 3. Cascaded Classification Model
The project uses a two-stage classification approach:
- **Stage 1 (Queue Classification):** Classifies the ticket into one of 10 departments (e.g., Technical, Billing, etc.).
- **Stage 2 (Priority Classification):** Predicts the urgency level (Low, Medium, High). The model for this stage uses the predicted output of Stage 1 as an additional feature, capturing the correlation between department and urgency.

---

## Model Performance (KNN Benchmark)

The system evaluated multiple algorithms (Random Forest, XGBoost, CatBoost, etc.), with **K-Nearest Neighbors (KNN)** showing robust performance on the high-dimensional embeddings.
**Confusion Matrix Classifying Queue**
![Confusion Matrix Classifying Queue](cm_queue-knn.png)

**Confusion Matrix Classifying Priority**
![Confusion Matrix Classifying Priority](cm_priority_knn.png)

### Queue Classification Report (10 Classes)
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **81%** |
| **Macro Avg F1-Score** | **80%** |
| **Weighted Avg F1-Score** | **81%** |

### Priority Classification Report (3 Classes)
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **87%** |
| **Macro Avg F1-Score** | **86%** |
| **Weighted Avg F1-Score** | **87%** |

---

## Intelligent Decision Logic (Simulation)

The core value of this project lies in the **Final Priority Score** formula, which balances technical urgency and customer emotion:

$$Final\ Score = (Weight_{Priority} \times f_{Priority}) + (Weight_{Sentiment} \times f_{Sentiment})$$

- **Threshold > 0.5:** Ticket is flagged for **Human Agent** intervention (High Risk/Urgent).
- **Threshold ≤ 0.5:** Ticket is handled by **AI Chatbot** (Routine/Low Risk).

**Example Output:**
> **ACTION:** Prioritize to Human Agent  
> **Queue:** Technical Support  
> **Priority Level:** High  
> **Sentiment:** Negative (0.98)  
> **Final Priority Score:** 0.85

---

## Tech Stack
- **Language:** Python
- **NLP:** HuggingFace Transformers (RoBERTa), Sentence-Transformers (MPNet)
- **Machine Learning:** Scikit-learn (KNN, Random Forest, MLP), XGBoost, CatBoost, LightGBM
- **Translation:** Deep-translator, Langdetect
- **Visualization:** Seaborn, Matplotlib

## Project Structure
- `translating_and_labeling.ipynb`: translation, and sentiment extraction.
- `classifying_queue_and_priority.ipynb`: Feature embedding, model benchmarking, and the final decision simulation.
