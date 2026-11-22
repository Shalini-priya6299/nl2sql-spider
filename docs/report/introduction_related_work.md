## 1. Introduction

Relational databases hold much of the world's structured data, but querying them requires SQL, which most
people do not know. *Text-to-SQL* systems close this gap: given an English question and a database, they
write the SQL query that answers it. The hard part is generalisation. A useful system must work on a database
it has never seen, with table and column names it could not have memorised, so it has to read the schema and
link the words in the question to the right tables and columns.

Large proprietary models such as GPT-4 do well on this task, but they are paid, closed, and often impractical
when data is private. In this project we ask how far **small open models** (about 1–7 billion parameters, free
to download, and able to run on a single free GPU) can go. We use Spider [Yu et al., 2018], a cross-domain
benchmark in which the test databases are different from the training databases, and the code model
Qwen2.5-Coder-1.5B-Instruct [Hui et al., 2024].

We compare three ways of improving a small model and measure each with execution accuracy (EX), split by
query difficulty (Easy, Medium, Hard, Extra hard):

1. **Prompting:** zero-shot (E1) and few-shot with similar solved examples (E2).
2. **Self-correction:** run the generated SQL and, if it fails, give the error back to the model and let it
   try again (E3, E5), following the idea in DIN-SQL [Pourreza & Rafiei, 2023].
3. **Fine-tuning:** train the model on Spider's training set with LoRA [Hu et al., 2022] (E4, and E5 together
   with self-correction).

Our central question is which of these gives the most benefit for its cost. Self-correction costs extra model
calls for every question at test time, while fine-tuning costs GPU hours once, at training time. We also ask
whether the answer changes between easy and extra-hard queries.

**Contributions.** (i) A reproducible pipeline from the same prompt format and evaluator for all five
experiments; (ii) a LoRA adapter for a 1.5B code model trained on Spider; (iii) a comparison of prompting,
self-correction and fine-tuning by difficulty and by cost; (iv) an error analysis of each system.

*(Part 2 may adjust the wording of the last paragraphs once all results are in.)*

---

## 2. Related Work

**Text-to-SQL benchmarks.** Early datasets such as WikiSQL used a single table per question. Spider [Yu et al.,
2018] raised the difficulty by providing about 200 multi-table databases from 138 domains, with human-written
questions and SQL, and by splitting train and test by *database*, so that models must handle unseen schemas.
Its four difficulty levels, based on the SQL components used, let us see where methods fail. The standard
metrics are exact set match and execution accuracy; because exact match can reject correct queries that are
written differently, execution-based comparison, and the test-suite refinement of it [Zhong et al., 2020],
is now widely used. We report EX.

**Prompting large language models.** Rajkumar et al. [2022] showed that a language model prompted with the
`CREATE TABLE` schema (optionally with sample rows) can write SQL without task-specific training, and that
adding a few examples helps. Our schema format follows this work. **[TODO: Part 2 may add retrieval of similar
examples, if cited in their section.]**

**Decomposition and self-correction.** DIN-SQL [Pourreza & Rafiei, 2023] splits the task into schema linking,
query classification, SQL generation and a self-correction step, and reached state-of-the-art accuracy on Spider
using a large proprietary model. Its error categories (schema linking, JOIN, GROUP BY, nesting, and so on)
are the ones we use in our error analysis. Our self-correction experiments (E3, E5) use a simpler loop with
execution feedback, because we use a much smaller model.

**Parameter-efficient fine-tuning.** Updating every weight of a language model needs a lot of GPU memory and
produces a full copy of the model for each task. LoRA [Hu et al., 2022] freezes the pretrained weights and
adds, next to selected weight matrices, a pair of thin trainable matrices whose product is a low-rank update.
Only these small matrices are trained, so the number of trainable parameters and the memory for optimiser
state both drop sharply, and the result is a small "adapter" file that is loaded on top of the base model.
QLoRA [Dettmers et al., 2023] goes further: the frozen base model is stored in 4-bit precision (NF4, with
double quantisation) while the LoRA weights are trained in higher precision, which lets larger models be
fine-tuned on a single GPU of 16 GB with little loss of quality. We use QLoRA because our only hardware is the
free 16 GB T4 GPU on Google Colab.

**Code language models.** Qwen2.5-Coder [Hui et al., 2024] is a family of open code models (0.5B to 32B)
pretrained on a large amount of code. Code models are a natural choice for SQL because SQL is code and the
schema is given in the form of `CREATE TABLE` statements.
