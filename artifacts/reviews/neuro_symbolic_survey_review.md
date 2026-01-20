# Review: Neuro-Symbolic Methods for Reasoning - A Survey

## Summary

This paper provides a comprehensive survey of neuro-symbolic approaches that combine neural networks with symbolic reasoning systems. The authors categorize existing methods and identify key challenges in the field.

## Strengths

1. **Comprehensive Coverage**: The survey covers a broad range of neuro-symbolic approaches from differentiable programming to neural module networks.

2. **Clear Taxonomy**: The proposed categorization into three main paradigms (symbolic grounding, neural-symbolic integration, and symbolic reasoning) is well-motivated and helpful.

3. **Identification of Gaps**: The authors clearly identify open challenges including scalability, interpretability, and the symbol grounding problem.

## Weaknesses

1. **Limited Technical Depth**: While comprehensive, the survey lacks sufficient technical detail for many of the covered approaches. More formal definitions and algorithmic descriptions would strengthen the work.

2. **Evaluation Metrics**: The paper doesn't adequately address how to evaluate neuro-symbolic systems, which is a critical gap in the field.

3. **Reproducibility Concerns**: Many cited works lack available code or clear implementation details, but this isn't adequately discussed.

## Detailed Comments

### Section 2: Background
The background section effectively motivates the need for neuro-symbolic approaches but could benefit from a clearer discussion of the fundamental representational differences between neural and symbolic systems.

### Section 4: Applications
The applications section would benefit from more quantitative analysis of performance gains achieved through neuro-symbolic approaches compared to purely neural or symbolic baselines.

## Minor Issues

- Figure 2 caption is too brief and doesn't explain the notation used
- Some recent work in differentiable theorem proving is missing
- Typo on page 7: "approches" should be "approaches"

## Recommendation

**Accept with Minor Revisions**

This is a solid survey that will be valuable to the community. The identified weaknesses can be addressed with revisions focusing on technical depth and evaluation discussion.

## Questions for Authors

1. How do you see the field evolving in terms of standardized benchmarks?
2. What role do you think large language models will play in neuro-symbolic reasoning?
3. Could you elaborate on the computational complexity trade-offs discussed in Section 5?