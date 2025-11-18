# Evaluation Types and Metrics

The system compares the performance of student models with and without customization using F1-scores.

## Evaluation Types

The system supports two evaluation types:

### Base Evaluation (`base-eval`)

**Zero-shot F1-score baseline of student model before customization**

Base evaluation tests the student model on a held-out evaluation dataset sampled from production logs, without any fine-tuning. This establishes the baseline performance of the out-of-the-box model.

- **Dataset:** Held-out evaluation set from production data
- **Model:** Student model (no fine-tuning)
- **Metric:** F1-score
- **Purpose:** Establishes baseline performance before customization

> **Note:** The model receives only the `request.messages` as input and generates its own response, which is then compared against the ground truth for F1-score calculation.

### Customized Evaluation (`customized-eval`)

**F1-score evaluation of customized model**

Customized evaluation tests the fine-tuned version of the student model on the same evaluation dataset. This measures the improvement from customization.

- **Dataset:** Same held-out evaluation set as base evaluation
- **Model:** Fine-tuned student model (via LoRA)
- **Metric:** F1-score
- **Purpose:** Quantifies improvement from customization and enables direct comparison to the base model

> **Note:** The customized model receives the same inputs as the base evaluation, allowing for direct F1-score comparison.

## Metrics

The primary metric used for evaluation is the **F1-score**, which measures the harmonic mean of precision and recall when comparing model outputs against ground truth responses.

### F1-Score

The F1-score balances precision (accuracy of positive predictions) and recall (coverage of actual positives), providing a single metric to assess model performance. Scores range from 0 to 1, where 1 represents perfect performance.

## Evaluation Results Format

Evaluation results are returned in a consistent structure. Each result includes metadata (such as evaluation type, timestamps, and progress) and a `scores` dictionary containing the F1-score metric.


