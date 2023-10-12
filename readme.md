
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center">VHAC - Vietnamese Custom Knowledge Chatbot</h1>
  <h2 align="center">Team: KPPTDLL</h2>
</div>

# HOW TO RUN THIS CODE?
## INFERENCE
Please follow instructions in "VHAC_Chatbot.ipynb".

Note: The inference process can be run on Google Colab (GC) with a single GPU T4. Also, we only tested on GC. Please contact us if you have any troubles.

## TRAIN
### TRAIN PASSAGE RETRIEVAL MODELS
Please follow instructions in "VHAC_Chatbot.ipynb". In particular, please run from Step 1 to Step 3 (do not run Step 4) and follow instructions from section "HOW TO TRAIN RETRIEVAL MODELS".


### FINETUNE ANSWER GENERATION MODEL
Please follow instructions in "finetune_viT5.ipynb".

# DATA AND MODEL USAGE
* [Q&A Dataset](https://huggingface.co/datasets/trientp/wiki_chatbot): Generated by "gpt-3.5-turbo".
* [Base Model xlm-roberta-base](https://huggingface.co/xlm-roberta-base): For training passage retrieval models.
* [Passage Retrieval Models](https://drive.google.com/drive/folders/11cmhjBNk5e5zYmKgcbCTbdg1WIOpBQc8?usp=drive_link): Candidate generation model, candidate ranking model and candidate encode model.
* [Base Model VietAI/vit5-base](https://huggingface.co/VietAI/vit5-base): For training answer generation models.
* [Answer Generation Model](https://huggingface.co/trientp/vit5_base_qa): Trained model.

#CONTACT
<p align="center">
Phat-Trien Thai
</p>
<p align="center">
trientp@viettel.com.vn
</p>
<p align="center">
phattrienthai99@gamil.com
</p>
<p align="center">
0979675072
</p>

