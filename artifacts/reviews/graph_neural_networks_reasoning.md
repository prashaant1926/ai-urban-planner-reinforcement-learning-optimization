# Review: Graph Neural Networks for Logical Reasoning

## Paper Summary

**Title**: "Learning to Reason with Graph Neural Networks for Knowledge Base Question Answering"
**Authors**: Anonymous (under review)
**Venue**: ICML 2024 (submitted)

This paper proposes a graph neural network architecture for multi-hop reasoning over knowledge bases. The authors claim their approach outperforms existing methods on complex reasoning benchmarks while maintaining interpretability through attention mechanisms.

## Technical Contributions

### 1. Architecture Design
The paper introduces a **Reasoning Graph Neural Network (ReasonGNN)** with three key components:

1. **Knowledge Graph Encoder**: Encodes entities and relations using GraphSAGE
2. **Multi-hop Reasoning Module**: Iterative message passing with attention
3. **Answer Prediction Layer**: Classification over candidate entities

### 2. Key Technical Innovation
The main novelty is the **"reasoning paths attention"** mechanism that:
- Tracks reasoning chains across multiple hops
- Provides interpretable explanation for predictions
- Uses soft attention over all possible paths (not just highest-scoring)

### 3. Training Strategy
- Pre-training on synthetic logical reasoning tasks
- Fine-tuning on downstream QA datasets
- Curriculum learning from simple to complex questions

## Experimental Evaluation

### Datasets
- **MetaQA**: 1-hop, 2-hop, and 3-hop questions
- **ComplexWebQuestions**: Real-world complex questions
- **WebQuestionsSP**: Multi-hop questions with Freebase

### Results Summary
| Dataset | Previous SOTA | ReasonGNN | Improvement |
|---------|---------------|-----------|-------------|
| MetaQA-3hop | 76.4% | **84.2%** | +7.8% |
| ComplexWebQA | 45.9% | **52.3%** | +6.4% |
| WebQuestionsSP | 74.7% | **78.1%** | +3.4% |

### Ablation Studies
The authors provide ablation studies showing:
- Reasoning paths attention: +5.2% improvement
- Curriculum learning: +3.1% improvement
- Pre-training: +4.7% improvement

## Strengths

### 1. Technical Soundness
- **Well-motivated architecture**: The reasoning paths attention addresses a clear limitation in existing GNN approaches
- **Comprehensive evaluation**: Tests on multiple datasets with proper baselines
- **Good ablation analysis**: Clearly shows contribution of each component

### 2. Practical Impact
- **Significant improvements**: Consistent gains across all benchmarks
- **Interpretability**: Attention weights provide reasoning explanations
- **Scalability**: Can handle knowledge bases with millions of entities

### 3. Experimental Rigor
- **Fair comparisons**: Uses same train/test splits as prior work
- **Statistical significance**: Reports confidence intervals and p-values
- **Error analysis**: Analyzes failure cases and limitations

## Weaknesses

### 1. Limited Novelty
- **Incremental contribution**: The reasoning paths attention is a relatively straightforward extension of standard attention
- **Missing baselines**: Doesn't compare against recent transformer-based approaches (e.g., T5, BART)
- **Architecture choices**: Some design decisions lack sufficient justification

### 2. Evaluation Concerns
- **Dataset limitations**: Mostly evaluates on relatively simple, structured datasets
- **Missing analysis**: No computational complexity analysis or runtime comparisons
- **Generalization questions**: How well does this work on truly open-domain questions?

### 3. Technical Issues
- **Scalability claims**: Authors claim scalability but don't provide evidence for very large KBs
- **Training stability**: No discussion of training stability or hyperparameter sensitivity
- **Implementation details**: Some important implementation details are missing

## Detailed Comments

### Section 3: Method
The method section is generally well-written but has several issues:

1. **Equation 3** appears to have a typo - the summation should be over neighbors, not all nodes
2. The reasoning paths attention mechanism needs clearer mathematical formulation
3. Missing discussion of computational complexity

### Section 4: Experiments
The experimental section is comprehensive but could be improved:

1. Need to compare against more recent transformer baselines
2. Runtime analysis is missing - how does inference time scale with KB size?
3. The curriculum learning setup could be described more clearly

### Section 5: Analysis
The analysis section provides good insights:

1. **Attention visualizations** are helpful for understanding the model
2. **Error analysis** reveals that the model struggles with numerical reasoning
3. **Case studies** effectively demonstrate the model's capabilities

## Minor Issues

1. **Figure 2**: The attention visualization is hard to read - consider using different colors
2. **Table 3**: Some numbers don't match those reported in the main text
3. **Related work**: Missing discussion of recent neuro-symbolic reasoning approaches
4. **Writing**: Several grammatical errors and unclear sentences throughout

## Questions for Authors

1. **Scalability**: How does performance degrade as knowledge base size increases beyond the tested datasets?

2. **Comparison with LLMs**: How does ReasonGNN compare against large language models like GPT-3 that can do few-shot reasoning?

3. **Multi-modal reasoning**: Can this approach be extended to incorporate textual context alongside structured knowledge?

4. **Training efficiency**: How many training examples are needed to achieve good performance? Is the pre-training step always necessary?

5. **Failure analysis**: What are the main categories of questions where the model fails? Are these fundamental limitations?

## Recommendation

**Weak Accept**

This paper makes a solid incremental contribution to neural reasoning over knowledge graphs. The reasoning paths attention mechanism is technically sound and provides interpretable explanations for predictions. The experimental evaluation is comprehensive and shows consistent improvements across multiple datasets.

However, the technical novelty is somewhat limited - the core contribution is a relatively straightforward extension of existing attention mechanisms. The paper would be stronger with comparisons against more recent baselines, deeper analysis of computational complexity, and evaluation on more challenging datasets.

The work is technically correct and provides useful insights for the community, but falls short of being a significant breakthrough. I recommend acceptance with revisions addressing the evaluation and analysis concerns.

## Suggestions for Revision

1. **Add transformer baselines**: Include comparisons with T5/BART fine-tuned on QA tasks
2. **Computational analysis**: Provide complexity analysis and runtime measurements
3. **Stronger datasets**: Evaluate on more challenging, open-domain datasets
4. **Implementation details**: Provide more complete implementation details for reproducibility
5. **Writing improvements**: Fix grammatical errors and clarify technical exposition

## Minor Corrections

- Page 3, line 15: "reasonig" should be "reasoning"
- Page 6, Table 2: Check accuracy numbers against main text
- Page 8, Figure 3: Improve visualization clarity
- References: Fix incomplete citations (e.g., [23] is missing page numbers)