Steps to execute this project:
1) First run Output_code_1.py, save the output  .csv files in a dataset.
2) Then run Output_code_2.py, save the output file in another dataset.
3) For running model.py, read 
				-> train1 from the output made in Output_code_1.py
				-> train2 from the output made in Output_code_2.py
				-> test_data from the output made in Output_code_1.py
   After that, the model training will start in model.py
	
Reason for creating seperate datasets was RAM overflow when we perform feature extraction on entire dataset on one go.
So we divided the traindata into 2 halves, cleaned and feature extracted output was stored in seperate datasets and then merged in model.py .
