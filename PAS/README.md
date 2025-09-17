# 🗂️ Project Description

This project provides scripts to generate simulated data for various attributes across multiple datasets. Please follow these instructions to run the corresponding scripts:

---

## 📊 1. ACS
**Simply run**:
```bash  
cd ACS  
python prompt_test2wrk/income.py  
```

---

## 🗳️ 2. ANES2020
**Selection based on candidate**:
- Simulate **biden** support rate:
  ```bash  
  cd Anes2020  
  python biden_xxx.py  
  ```
- Filename conventions:
  - `biden_round2few.py`：Few-shot method
  - `biden_round2zero.py`：Zero-shot method
- Other candidates:
  ```bash  
  python trump_xxx.py  
  python obama_xxx.py  
  ```

---

## 📈 3. BIS
**Generate simulated data**：
```bash  
cd BIS  
python prompt_numerical.py  
```

---

## 📋 4. EmpS
**Run corresponding script**：
```bash  
cd EmpS
python rpla_emp.py  
```

---

## 🧾 5. GSS
**Select appropriate script**：
```bash  
cd GSS  
python work_prompt.py   # Primary option  
# or  
python work_prompt2.py  # Alternative option  
```
> 💡 **Configuration Tip**：  
> Control few-shot examples in prompt generation at `line 178`

---

## 🔋 6. RECS
**Attribute-specific scripts**：
```bash  
cd RECS  
python prompt_test1.py   # Simulate UGASHERE  
python prompt_test2.py   # Simulate USEEL (paper exclusion)  
python prompt_test3.py   # Simulate USELP  
```

## 📺 7. Media
**Attribute-specific scripts**：
```bash  
cd Media
python num2_conview.py # Simulate numerical results
python num2_weekdays.py # Simulate accuracy
```

## 🎥 8. MxMH
**Run corresponding script**：
```bash  
cd MxMH
python rpla_musicMental.py  
```

## 🏠 9. NHTS
**Run corresponding script**：
```bash  
cd NHTS 
python rpla_NHTS.py  
```

## 🧑‍🎤 10. YPS
**Run corresponding script**：
```bash  
cd YPS  
python rpla_youth.py  
```

## 💭 11. MHD
**Run corresponding script**：
```bash  
cd MHD
python rpla_MentalHealth.py  
```
---

## ⚙️ Model Configuration
When switching question-answering models (e.g. askgpt/llama):

**1. Locate model selection block**：
```python  
try:  
    response1 = extract_float_string(ol3(conversation_history))    # llama3  
    response2 = extract_float_string(ol31(conversation_history))   # llama3.1  
    response1 = extract_float_string(ask_gpt3(client, conversation_history))  # GPT-3.5-Turbo  
    response2 = extract_float_string(ask_gpt4(client, conversation_history))  # GPT-4-Turbo  
except (ValueError, TypeError) as e:  
    ...  
```

**2. API Key Requirement**：  
Replace `apikey` in `utils.py` with your OpenAI API key when using GPT models

---

> 💬 For additional assistance, please contact the project author.
