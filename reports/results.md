**Results**
We evaluated four approaches for prioritizing critical patients at ED intake: a rule-based triage baseline, a context-aware logistic model, a graph-based model with patient-attribute embeddings, and a pairwise learning-to-rank model. To prioritize critical patients, we used a validation-tuned threshold that maximized F2 score (favoring recall), with the label defined as acuity ≤ 2.

Across all models, Recall@k and Precision@k were identical (Recall@5 = 0.819, Recall@10 = 0.963, Precision@k = 1.0), indicating that each model consistently ranked the most critical patients near the top. Differentiation between models appeared in classification metrics. The context-aware model achieved the strongest performance, with F2 = 0.903 and F1 = 0.788 at a tuned threshold of 0.15, and accuracy of 0.654. This outperformed the baseline (F1 = 0.282, accuracy = 0.422) and the graph-based model (F1 = 0.737, F2 = 0.814, accuracy = 0.597). The pairwise learning-to-rank model failed to learn meaningful separation (F1/F2 = 0.0), likely due to insufficient positive-negative pairs within hourly groups.

Overall, the context-aware model provided the best balance for safety-critical prioritization while remaining interpretable and deployable.

**Discussion**
The identical Recall@k and Precision@k across models suggest that the top-ranked critical cases are strongly determined by basic clinical signals (vitals and acuity), making the ranking task relatively easy for this dataset. However, once we applied a class-balanced training scheme and threshold optimization for recall-heavy objectives, the context-aware model showed clear gains in F1/F2, indicating better classification of borderline cases beyond the top-k.

The graph-based model underperformed relative to the context model, likely because the available graph features (chief complaint, transport, race) were too coarse to add meaningful structure. The pairwise ranker also did not learn effectively, likely because small group sizes (hourly buckets) limited the number of valid training pairs.

These results support the project’s hypothesis that triage is better framed as a ranking and prioritization problem, but they also highlight the need for richer operational signals (real bed availability, staffing, queue length, or time-to-physician) to fully realize the advantages of context-aware or graph-based recommendation systems.
