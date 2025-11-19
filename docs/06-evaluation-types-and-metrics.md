# Evaluation Types and Metrics

The system compares the performance of student models with and without customization using F1-scores.

## Key Terminology

- **Student Model**: The smaller, more efficient model being trained/evaluated (e.g., Llama 3.2 1B, Llama 3.2 3B)
- **Teacher Model**: The larger, more capable model that generated the production responses used as ground truth
- **Base Evaluation**: Testing the student model before any customization (zero-shot)
- **Customized Evaluation**: Testing the student model after fine-tuning with LoRA
- **F1-Score**: The primary metric measuring how well the student model's responses match the teacher model's responses

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

The F1-score balances precision (accuracy of positive predictions) and recall (coverage of actual positives), providing a single metric to assess model performance.

**In this system**, the F1-score compares the student model's generated responses against the teacher model's responses (ground truth) by:
- **Precision**: Measuring what percentage of tokens/concepts in the student's response are correct
- **Recall**: Measuring what percentage of the teacher's response tokens/concepts the student captured

Scores range from 0 to 1, where:
- **1.0** = Perfect match with teacher model's response
- **0.0** = No overlap with teacher model's response
- **Higher scores** = Better alignment between student and teacher outputs

## Workload Types and Metrics

The system supports two workload types, each with different evaluation metrics:

### Classification Workloads

Classification workloads use **F1-score** as the primary evaluation metric. This is the default for the financial services variant.

**Metrics:**
- **F1-Score**: Measures how well the student model's classification responses match the teacher model's responses

**Use Cases:**
- News classification (e.g., financial news sentiment analysis)
- Text categorization tasks
- Any task where responses are categorical labels

### Tool-Calling Workloads

Tool-calling workloads use multiple metrics to evaluate function call accuracy and correctness.

**Metrics:**
- **Function Name Accuracy**: Percentage of correct function names in tool calls
- **Function Name and Arguments Accuracy**: Percentage of tool calls with both correct function name and arguments
- **Tool Calling Correctness** (optional): LLM-as-judge rating for overall correctness (requires `tool_eval_type: "tool-calling-judge"`)

**Use Cases:**
- Agent tool routing
- Function calling workflows
- Multi-step agent tasks

**Configuration:**
To use tool-calling evaluation, set `workload_type: "tool_calling"` in your configuration file or use auto-detection (the system detects tool calls in your data).

## Evaluation Results Format

Evaluation results are returned in a consistent structure. Each result includes metadata (such as evaluation type, timestamps, and progress) and a `scores` dictionary containing the relevant metrics:

**For Classification Workloads:**
```json
{
  "scores": {
    "f1_score": 0.85
  }
}
```

**For Tool-Calling Workloads:**
```json
{
  "scores": {
    "function_name": 0.92,
    "function_name_and_args_accuracy": 0.88,
    "tool_calling_correctness": 0.90
  }
}
```


