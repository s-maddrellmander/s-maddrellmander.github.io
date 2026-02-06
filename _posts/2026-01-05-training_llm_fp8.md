
# What happens when you train an LLM in fp8?

## Motivation

- FLOPS limited, especially in large models. Lower precision means bigger models trained faster.
- Update two weights for the cost of one.
- How stable is it? Low precision work is beset with numerical instabilities. How hard on latest GPUs to get this working?

## Approach

- Inspired by Karparthy’s nanoChat / nonGPT we will train a GPT2 variant with computation in FP8.
- Yes it’s not bringing all the instabilities for full FP8, but we get most of the speed ups.
- Swap out all linear layers (QKV projections, and MLPs in the SwiGLU) with FP8 variants.
    - Do this with torchao - handles the backend kernels and applies some stability tricks as well.
    - Notes on these in the end notes.
- These use the dedicated kernels on the H100 and above series Nvidia GPUs.

## Methodology

- Investigate the scaling laws here.
- Small models to calibrate (125M - 350M) and then see how the 1.5B model agrees.
- Use the Chinchilla rule of thumb, and say 20x tokens per parameter is “compute optimal”.
    - This means given a fixed compute budget what’s the best performing model you can get trading off between size and training tokens.
- Build our own dataset mixture for interest:
    - Mostly FineWeb EDU but mixed in with limited python and C programming, and arXiv papers.
    - This is a hunch - didn’t have resources to explore if this genuinely improves downstream tasks.
- First - compare BF16 with FP8 compute. We do this on the smallest models, showing the training dynamics follow really closely. They see the same data in the same order, so the only differences should be due to the computational precision.
    - We can see the training curves following really closely.
- Then we run some scaling experiments to see if we can replicate the classic Kaplan scaling plots, but in FP8 compute.
    - We run the 125M, 250M, and 350M models to 20x tokens and plot the loss curves.
    - So far so good.
    - Then we set the final 1.5B run off and track the loss to see if we can get the final point on our plot.
- And all things considered given that we only have four points - it’s not bad! Some margin for error due to seeding / fluctuations we expect in particularly small models. But the trend is good.
- And the most important part, the run maintains stability (almost) for the full 30B tokens, while gaining a real throughput improvement of 40%.

## Results

- Quantify the results. First of all we can plot the characteristic panel scaling plot for fun. We see the loss trending down. And the linear relationship falling out in model size and dataset size.
- The optimal frontier in the first panel shows the step from one model size to the next with the frontier more or less tracking along the diagonal as expected.
- (Again only 4 data points, but nice to see)
- The other type of quantification is the classic benchmarks.
- We will use:
    - Perplexity measures, MQAs, boolean, and physical scene understanding.
- NOTE: We don’t expect the model to perform well at all here. Very much not the point, but good to see where we stack up against comparable models.
- TABLE OF THE RESULTS
- We compute all of the metrics on our own models using the lm_eval harness.

### Results of Note

- Base Llama 3 models are not far ahead here.
- Our base model nearly scraping above random tbh - not surprising at this stage, but reassuring to see.
- Qwen models perform really well, I suspect there’s a lot of post training here, but impressive from such a small model.
- Obviously the big models swap everything.
- But other locally run LLMs are in the same ballpark, and we haven’t even done the mid / post training yet.
