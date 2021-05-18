# Deeper-into-Biases-with-Personality-in-Recommender-Systems
### This repository is for my project in CS274 in Spring 2021:

I have used codes from the repository:

  - https://github.com/CPJKU/pers_bias
  - https://github.com/CRIPAC-DIG/SR-GNN

They come from the papers:

- "Personality Bias in Music Recommendation" by Alessandro B. Melchiorre, Eva Zangerle, Markus Schedl published at RecSys 2020.
- "Session-based Recommendation with Graph Neural Networks" by Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu, which is an AAAI 2019 Paper.

Part of their code is in this repository. I built off of that to combine the papers and use the SR-GNN to try to see if there were biases in personality with recommender systems along with the results form the "Personality Bias in Music Recommendation" paper. The files that I have changed are: 
- algorithms/GNN/pytorch\_code/main.py
- algorithms/GNN/pytorch\_code/preprocess.py

I have added datafiles:
- algorithms/GNN/pytorch\_code/twitter
- algorithms/GNN/pytorch\_code/shuffled.csv

To preprocess the data:
- run algorithms/GNN/pytorch\_code/preprocess.py

To run the SR-GNN model:
- run algorithms/GNN/pytorch\_code/main.py

The datafiles that I have generated for this project were not able to be uploaded onto github so please get it here and put it under algorithms/GNN/pytorch\_code/twitter
- Datafiles location: https://drive.google.com/file/d/1ta5UvW-FutAsHjOTeQMjS1CrYezwzssU/view?usp=sharing
