Welcome to the repository of my bachelor thesis. 

In **"all_json_csv_files"** you can find all the annotated data that is used for my research.

In **"final_version_thesis"** you can find the pdf with the thesis, and all the other fills to complile the latex into a pdf.

In **"sub_question_1_3"** you can find the notebooks that include the code to calculate the results of sub question 1 and 3, and the gpt_suggestions vs the gold_standard. At the top of the notebook, the needed fills and how to run the notebook are stated.

In **sub_question_2_training_models** you can find all the python scripts to train the different models. Also all the results can be found inside the 'sub_question_2_training_models/results' folder.

In **"sub_question_4"** you can find the documents with the results of the forms and the notebook to calculate the results of sub question 4. At the top of the notebook, the needed fills and how to run the notebook are stated.

**Abstract:**
Although Large Language Models (LLMs) are highly efficient at processing structured formal data, their ability to interpret informal human communication remains a major challenge. Understanding this limitation is crucial if we want to use LLMs to speed up the analysis of online discussions. In current research, there are two big gaps. First, most research that tests models on framing tasks uses formal text as datasets, making it unclear how well LLMs handle finding frames in informal and persuasive discussions, like those on the ChangeMyView (CMV) subreddit. Second, while studies show that AI advice can bias human annotators on factual datasets, they miss what happens when humans get LLM suggestions for subjective tasks. This is crucial to see whether humans become overly dependent on LLM suggestions, which affects maintaining the quality and reliability of the dataset.

This research fills these gaps by testing both stand-alone models and hybrid human-LLM teams on the CMV dataset. To do this, a dataset of 495 unique comments was sampled from CMV. A human gold standard was created using \citeauthor{card2015media}'s \citeyearpar{card2015media} 15 framing dimensions. Different models were tested, to see how well they can perform the task. Finally, a user study looked at how humans and LLMs work together. It compared how humans annotate on their own against how they perform when they get LLM suggestions, allowing us to measure automation bias.

Our results show that while humans remain highly effective at interpreting implicit meaning, LLMs consistently struggle with these nuances. Furthermore, the hybrid annotation workflow failed to improve performance. Instead, it led to automation bias, which caused f1-scores to drop whenever participants accepted wrong LLM suggestions. This study concludes that without critical human oversight, current human-LLM collaboration can actually lower the quality instead of improving it. Because of this, human annotators are still necessary for framing tasks, and hybrid workflows need to be set up in a way that encourages annotators to stay critical of the LLM suggestions.


**Note:** annotator 1 = Floris, annotator 2 = Julius, annotator 4 = Anya
  
