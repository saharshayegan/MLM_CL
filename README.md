# MLM_CL
Implementation of MLM+CL training on ParsBERT\
This project was done for the Natural Language Understanding with Deep Learning course (COMP 545) at McGill university.\
\
You need to create the following directory with the following files to run this code:
```
runs/ 
data/ 
├── tweets/
│   ├── tweets_user_{ID}.json
│   ├── ...
│   └── tweets_user_{ID}.json
├── df_monarchy_and_gov_top_25_rsrc_tweets.csv
└── file_names.txt
```

## Instruction to run the code:
the order of running the codes is:
1. create_dataset.py
2. no_trainer.py
3. cluster_users.ipynb
###  create_dataset.py
This file will preprocess the raw tweet files and converts them to datasets that are understandable for the training code. 
The default values are `NUM_FILES_TO_PICK=500` and `MAX_NUM_TRAIN_SAMPLES = 842952` which cover the big dataset. In order to make the data smaller you can change these numbers. Make sure to change these numbers in `no_trainer.py` too if you decided to change them.\
This code will create a dataset folder in the `data/` directory and it will be used in the next step.\
You should change the `` variable to the project path.
### no_trainer.py
This code runs the model training procedure. \
You need to take care of the project path in this code as well. There is also the variable `run_name` that picks the name of the model (e.g. `Final_NoCL_FullData`), and the `WithCL` variable that decides whether to use the contrastive loss on the training or not. Please make sure to change them to your desired value.
### cluster_users.ipynb
This code is designed as a notebook to give you the freedom of changing the settings of the plots and seeing the results.\
It will use the model that was trained (you need to specify the model name and the model path) and embeds the labeled tweets and visualizes then in a 2D space.
Please make sure to change the path you are willing to save the plots in `plot_2d` function.


#### Note: The dataset used for this project is considered as sensitive data and will only be shared with fellow-researchers under certain conditions.

---------

These HuggingFace tutorials helped alot through this project: \
https://huggingface.co/learn/nlp-course/en/chapter5/5 \
https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt


---------
If there were any problems with the code please do not hesitate to contact me at sahar [dot] omidishayegan [at] mail [dot] mcgill [dot] ca
