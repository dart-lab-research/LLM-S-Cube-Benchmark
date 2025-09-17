# 🔬 FAS Simulation Instructions (RECS Example)

Follow this step-by-step guide to run FAS simulations using RECS as an example:

---

## 1. 🚀 Run Simulation
Navigate to the target directory and execute:
```bash  
cd RECS  
python Fsimulation.py  
```

### Output:
- Generates all simulated pairs
- Stores results in `.txt` file

### Sample Output Format:
```json  
[  
    ["12521", "1955.06", "2656.89"],  
    ["5243", "713.27", "975"],  
    ["2387", "334.51", "522.65"],  
    ["9275", "1424.86", "2061.77"],  
    ["5869", "1087", "1463.04"],  
    // ... additional entries ...  
]  
```

---

## 2. 🧹 Filter Invalid Files
**Manual verification required**：
- Inspect all generated `.txt` files
- Remove any files that:
  - Are not valid JSON arrays
  - Contain malformed entries
  - Have inconsistent column counts

> ⚠️ **Critical Step**:  
> This manual filtering ensures downstream processing compatibility

---

## 3. 🤝 Merge Simulated & Real Data
Execute merge script:
```bash  
python deal.py  
```

### Configuration:
```python  
# Example path configuration in deal.py  
SIMULATED_PATH = "path/to/filtered_simulated.txt"  # ← Update this  
REAL_DATA_PATH = "path/to/real_data.csv"          # ← Update this  
OUTPUT_PATH = "path/to/merged_results.csv"         # ← Update this  
```

> 📌 **Remember**:  
> Modify these paths in `deal.py` to match your actual directory structure

---

## 4. 📊 Calculate Distribution Difference
Run KL divergence calculation:
```bash  
python metrics.py  
```

### Output Interpretation:
- Generates KL divergence metrics in `kl_results.txt`
- **Final metric calculation**: Based on the KL divergence values, we apply the normalization formula to convert them to [0,1] scores:
  
  $$S_j = \frac{ \ln\left(1 + \frac{1}{D_j}\right) }{ 1 + \ln\left(1 + \frac{1}{D_j}\right) }$$
  
  Where $D_j$ is the KL divergence for each target attribute, and higher $S_j$ indicates better model performance.
- The normalized scores represent our final evaluation metric

---

> 💌 For additional assistance or troubleshooting, contact the project author.
