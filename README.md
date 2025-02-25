# Step 1: Set Up Environment

## 1. Create a Virtual Environment
```bash
conda create -n ADL_ENV python=3.10
conda activate ADL_ENV
```
## 2. Install Required Packages
```bash
pip install transformers torch datasets evaluate
pip install bitsandbytes tqdm packaging pandas
pip install dataclasses peft matplotlib
```
# Step 2: Train the Model
Run the training script `qlora.py`with the training data`train.jsonl`and save the model. To use the Llama-3-Taiwan model for training, change the `--model_name_or_path` argument to `yentinglin/Llama-3-Taiwan-8B-Instruct`
```bash
python qlora.py --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo --dataset  hw3/data/train.json --do_train --do_eval --train_on_source False --output_dir ./output --max_train_samples 5000 --max_steps 300 --save_steps 30  --bf16
```

# Step 3: Evaluate Model Performance
Use`ppl.py`to calculate the perplexity. Specify the path to the trained model, or download a pre-trained model first to obtain the Mean Perplexity.

```bash
bash ./download.sh 
```

```bash
python ppl.py --base_model_path kyara_finetune_checkpoint  --peft_path adapter_checkpoint --test_data_path hw3/data/public_test.json
```

# Step 4: Generate Model Predictions
Use`run.sh`with`private_test.json`to generate the final prediction output`prediction.json`

```bash
bash ./run.sh kyara_finetune_checkpoint adapter_checkpoint hw3/data/private_test.json prediction.json
```

# Step 5: Post-process Data
Perform post-processing on the predictions in `prediction.json`to produce a simplified output file,`simplified_prediction.json`

```bash
python simplified_output.py
```

# Step 6: Plot Learning Curves
Gather the perplexity values from each checkpoint of the trained model, and use `plot.py` to generate a learning curve plot.

```bash
python plot.py
```

