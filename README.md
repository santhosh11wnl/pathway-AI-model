nlp.py is the file , where I have performed topic modeling using lda and ner in summarizing a pdf.
Main.py file contains gpt-4 api, where extracted and identified pathway signaling images are taken and gpt is guided by a prompt , using that gpt -4 generates gene names along with relations.
Output_k.json file contains lda otpic modeling of a pathway signanling pdf and ner results summary of clinical data.
Output.json contains all the extracted gene pair wise relations.
Numbered text json files contains all the text data which is primarly focused on gene data from the pdf.
These json files can be used in biological gene analysis and as well as text classification, with this user can easily understand pathway papers and importance of it.
Three outputs can be generated from the model, one contains gene pairwise relations from images , next one contains all the text information which focuses on gene classification as well as thier relations and finally last one contains topic modelling of the data through lda and a summary is created based on ner results.
