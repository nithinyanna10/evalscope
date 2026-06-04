# Handout B — Why This Matters and How to Use It

**Audience:** Sales engineers, deployment leads, product managers, customer teams

---

## What Changes for the Customer Conversation

Today, answering "is this model good enough?" requires running 315 coding problems and 100 long-context questions — hours of compute per candidate model. These pruners cut that to ~30 coding problems and ~20 long-context questions while preserving the same pass/fail answer. A sales engineer can now run a benchmark during a customer call, not after it.

## How to Run It Tomorrow

Three commands. No setup beyond the evalscope install.

```bash
# 1. Run the pruned coding benchmark (~10× faster than full)
evalscope eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy": "stratified", "prune_ratio": 0.1}' \
    --output ./results/

# 2. Run the pruned long-context benchmark
evalscope eval --model <model> --datasets aa_lcr_pruned \
    --dataset-args '{"pruning_strategy": "stratified", "prune_ratio": 0.2}' \
    --output ./results/

# 3. Get the go/no-go answer
python -m evalscope_ext.tools.compare_runs --full ./baseline/ --pruned ./results/
```

The output is a table showing whether the model's pruned scores match the full-benchmark baseline. Green check = the model ranks the same. That's your go/no-go.

## What About Multimodal?

If the customer's roadmap includes vision capabilities next quarter, the `mmmu_pruned` probe gives an early signal. Unlike random sampling — which mostly picks text-answerable questions where the image is decorative — this probe specifically selects questions that require reading diagrams, interpreting charts, and understanding spatial detail in images. An image encoder that's degrading will fail these questions first, surfacing problems that random sampling would miss until you've run thousands of samples.

## Why Should a PM Care?

Two reasons. First, faster benchmarking means faster customer conversations — the turnaround from "can you test this model?" to "here are the results" drops from hours to minutes. Second, the pruners are model-agnostic. When a new model drops next month, the same pruned benchmark works without any reconfiguration. The evaluation infrastructure scales with the model catalog, not against it.
