# Keyword_Extraction_Economics
---

Implementation and pre-trained models of the paper *Keyword Extraction in Economics Literatures using Natural Language Processing*:

[🔗 Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9528546)

If you encounter any problems, feel free to contact me or submit a GitHub issue.

Using Natural Language Process (NLP) as an efficient way to research paper is important when user feedback is sparse or unavailable.
The task of text mining research paper is challenging, mainly due to the problem of unique characteristics such as jargon.
Nowadays, there exist many language models that learn deep semantic representations by being trained on huge corpora.
In this paper, we specify the NLP pre-processing process with Economics journal paper and apply it to a deep learning model to extract keywords.
Here, we focus on the strength of NLP when applied to an unknown field.
The analysis result shows the possibility and potential usefulness of the relationship research between keywords in research papers.

### Work flow of keyword extraction from Economics journal paper using BERT
![fig4](https://user-images.githubusercontent.com/48557539/193765866-86792ecd-420d-426d-996b-627aa0584efc.jpg)

---


### Prepare Data

---

**Economics Journal Paper Data**

For keyword extraction, we apply the economics journal data records extracted from Web of Science. A total of 20 2019 SSCI edition journals ranked in the Impact Factor (IF) and Journal Citation Reports (JCR), which indices the influence of the paper, were extracted. This information is available on the site provided by Clarivate Analytics, a well-known developer of the paper's index management program, End Note. 

The total number of the journal data records is 46,014.
Specifically, we use only three data, which are document type, abstract, and author keywords.
If the data doesn’t have at least one of the three elements, it is eliminated, and the number
of the remains is 36,345 in total.

> Postprocess.py
> 

> [Preprocess.py](http://Preprocess.py)
> 

preprocessing and postprocessing for journal paper data. 

After loading the data files, Word-piece tokenizer does the tokenizing encode process, which is a sub-word tokenization algorithm used for BERT. It converts ‘abstract’ and ‘author keywords’ strings into a sequence of tokens. It split in words for word-based vocabulary or sub-words for sub-word-based vocabularies.

> BERT.py
> 

After preprocessing the data, we import the pre-trained module implemented with Torch package and Tensor Dataset. 

In order to train a deep bidirectional representation, we predict keywords by tagging tasks. The tokens split by the tokenizer are linked with tag of words. 

In fine-tuning, we add a classification layer to the last layer of the model in order to tag the keyword of tokens. We train the model with labeling and tagging with cross-entropy loss with batch size of 12.

### Experiment Results

---
![camera_ready_result](https://user-images.githubusercontent.com/48557539/193765896-4a0252df-1b1b-4e65-92df-1f1d474623f9.PNG)


### How to Cite

---

If you are using our code, please cite [our paper](https://ieeexplore.ieee.org/document/9528546):

```jsx
@INPROCEEDINGS{9528546,
  author={Kim, Soojeong and Choi, Sunho and Seok, Junhee},  
	booktitle={2021 Twelfth International Conference on Ubiquitous and Future Networks (ICUFN)},   
	title={Keyword Extraction in Economics Literatures using Natural Language Processing},   
	year={2021},  
	pages={75-77},  
	doi={10.1109/ICUFN49451.2021.9528546}}
```
