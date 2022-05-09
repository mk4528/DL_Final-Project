# Project Title

Uncertainty of Deep Learning for Time Series Data

# Team members

Masataka Koga and Yuki Ikeda

# Project Goal/Objective

In this project with my teammate in a practical deep learning class, we examined the nature of the uncertainty (robustness, stability or variance) of deep learning models for time series data in finance. Our main concerns are how is the uncertainty changing by different models, different network architectures, and different training process. Comprehension about these effect to the uncertainty helps make robust, reliable deep learning models. Especially, financial companies are of interest in the robust ones for time series data. We can estimate the uncertainty of deep learning by using Monte Carlo Dropout [1].

Specifically, we investigated the difference of model’s uncertainty between feed forward NN and LSTM, different dropout rates, different epochs and optimizers. In conclusion, we found the following characteristics about the uncertainty:

✓Though LSTM seems to have advanced structure for time series, it is not necessarily superior in uncertainty.

✓As expected, appropriate selection of optimizers and training epochs improves fitting accuracy and mitigates the uncertainty.

✓Generally, lower dropout rate leads to overfitting more easily but it reduce the uncertainty.

# About this Repository

This repository is presenting all the two relevant codes we used to develop this project, a base_nn.py and base_lstm.py. These files employ yahoo! API and the wikipedia SP500 ticker lists, collect the price data of SP500 index and the individual 500 stocks for past 10 years, and preprocesse the data to compute return rates and technical indices, eliminate missing values,  reduce feature dimensions, split train and test data, and construct input tensors. Then, the feed forward neural networks or LSTM models are trained under different settings of model and hyperparameters. As a result, they compute and visualize the uncertainty of the models and using MC Dropout. The codes for developing comparative analyses under different settings and the plots of loss and uncertainty are also included.

Also, there are image files for expalination in this README.md in images directory. There are no other files because this project uses only the data available online.

# Motivations

- Deriving a prediction interval or a Bayesian predictive distribution for a deep learning model is crucial for real applications to be aware of model uncertainty and ensure reproducibility, explainability and fairness, but unlike the conventional statistical predictive models, quantifying, comparing and clarifying the nature of uncertainty for different deep learning models and under different training settings (e.g. hyperparameters) is not well-researched.


- Although there is research that explores estimation of uncertainty of deep learning models, e.g., MC dropout [[1]](https://arxiv.org/pdf/1506.02142.pdf), deep ensemble [[2]](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf), MC dropout applied to LSTM for time series data [[3]](https://arxiv.org/pdf/1709.01907.pdf), and there are some empirical analyses (mainly in medical fields using medical images and CNNs [[4]](https://www.sciencedirect.com/science/article/pii/S0925231219301961) [[5]](https://www.nature.com/articles/s42256-018-0004-1)) that put them into practice, at the best of our knowledge, there is no survey that widely and cohesively computes, compares, and clarifies the nature of uncertainty of deep learning models for time series data which is quite familiar in finance. 


- Hence, we will conduct such a survey at the best of our skills and resources for S&P500 stock price data.

# Approach/Techniques

1. Define model uncertainty as predictive distribution $p(y^*|x^*)$ for new sample $(x^*, y^*)$


2. Decompose the predictive variance into model uncertainty (1st term) and inherent uncertainty (2nd term):
$$
~\\
{\rm Var}(y^*|x^*) = {\rm Var}(f(x^*)) + E[{\rm Var}(y^*|x^*)]. \\
$$
For trained deep learning model, compute model uncertainty ${\rm Var}(f(x^*))$ using Monte Carlo dropout [[1]](https://arxiv.org/pdf/1506.02142.pdf)
  
 
3. We conduct comparisons of uncertainty with different models and under different settings such as the following and clarify what factors affect the amount of uncertainty and how.
 - Model settings: 
     - Feed forward NN or LSTM
     - Dropout probabilities
 - Training settings
     - Train epochs
     - Optimization methods (SGD, Adagrad, Nadam, etc)

# Entire pipeline summary

![image.png](https://github.com/mk4528/DL_Final-Project/blob/main/images/pipeline%20summary.png)

# Data, Target and features

 - Original Data: daily price data (May-7-2012 to May-2-2022) of S&P 500 and IS: individual stocks  selected as one of S&P 500
 - Data Selection: exclude IS with less period data & IS highly (> 0.7) correlated with another IS
 - Data Transformation: 
   - SP500_rate, IS_rate: Rate of Change for S&P 500 and IS
   - STM: Short-Term Momentum of IS Price (Time Period = 5)
   - MTM: Medium-Term Momentum of IS Price (Time Period = 25)
   - LTM: Long-Term Momentum of IS Price (Time Period = 75)
   - RSI: Relative Strength Index of IS Price (Time Period = 14)
   - NATR: Normalized Average True Range of IS Price (Time Period = 14)
 - Missing Data Fill: drop NA for SP500_rate, forward fill for others
 - Features: (SP500_rate, IS_rate, STM, MTM, LTM, RSI, NATR)_{t, t-1, t-2, t-3}
 - Target: SP500_{t+2}

# Example commands to execute the code

- To execute the `py` file in the git repository, execute the codes like the ones below:


```python
python base_nn.py
python base_lstm.py
```

# Results and Observations

Utilizing MC Dropout, we examined all the comparison of uncertainty listed below:
- Optimization methods: SGD/Adagrad/Adadelta/RMSprop/Adam/Nadam
- Epochs: 10, 20, 30, 50, 75, 100, 200
- Dropout probabilities: 0.01, 0.05, 0.1, 0.2, 0.5
 - by changing Dropout probabilities, we investigate how the uncertainty due to model misspecification behaves for models with different regularization (dropout probability)
- Models: feed forward NN models and LSTM

## Optimization methods and Epochs

In the following result graph, loss and uncertainty is correlated, i.e. stopping at optimal epoch in terms of validation loss leads to lower uncertainty. Because of the correlation, superior optimization method (w.r.t. epoch or wall clock time) would lead to lower uncertainty, especially in the case with the smaller number of epochs.

![image_optimizer_epoch_relation.png](https://github.com/mk4528/DL_Final-Project/blob/main/images/image_optimizer_epoch_relation.png)

![image_optimizer_epoch-2.png](https://github.com/mk4528/DL_Final-Project/blob/main/images/image_optimizer_epoch.png)

## Dropout probability

Low dropout probability (say 1%) leads to low uncertainty, while relatively high probability (20%, 50%) leads to low MSE/MAE on the test data (might be coming from the variance p(1-p) of a Bernoulli random variable). There appears to be a trade-off of the model's uncertainty to loss between some models with different drop rates.

![image_dropout_relation.png](https://github.com/mk4528/DL_Final-Project/blob/main/images/image_dropout_relation.png)

![image_dropout_band.png](https://github.com/mk4528/DL_Final-Project/blob/main/images/image_dropout_band.png)

## LSTM Model

- We also tried LSTM using features on last 10 days with different architectures, it ends up almost the same test error with feed forward NN using only last 4 days.

  - **Though LSTM is considered to be more advanced model, the uncertainty of LSTM was not necessarily lower than feed forward NN.**

- Note that train data size in out experiment is N≒2000 which is small for using neural network in the first place. We should also do experiment for larger data.

![LSTM](https://github.com/mk4528/DL_Final-Project/blob/main/images/image_1.png)

# Conclusion

- We have investigated the uncertainty in deep learning models for time series data.


- Question: how is the uncertainty changing by
    - Different models? (feed forward NN, LSTM)
        - LSTM is not necessarily superior in uncertainty
        
    - Different network architectures (dropout probability)
        - There might be tradeoff between loss and uncertainty
        
    - Different training process (epochs/optimizers)
        - Loss and uncertainty was highly correlated

# References

[1] Yarin Gal, Zoubin Ghahramani (2016), [Dropout as a bayesian approximation: Representing model uncertainty in deep learning](https://arxiv.org/pdf/1506.02142.pdf), *Proceedings of Machine Learning Research* Vol 48, p.1050-1059

[2] Balaji Lakshminarayanan, Alexander Pritzel (2017), Charles Blundell, [Simple and scalable predictive uncertainty estimation using deep ensembles](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf), *Neural Information Processing Systems*

[3] Lingxue Zhu, Nikolay Laptev (2017), [Deep and Confident Prediction for Time Series at Uber](https://arxiv.org/pdf/1709.01907.pdf), *IEEE International Conference on Data Mining Workshops*

[4] Guotai Wang, Wenqi Li, Michael Aertsen, Jan Deprest, Sebastien Ourselin, Tom Vercauteren (2019), [Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S0925231219301961), *Neurocomputing* Vol 338, p.34-45

[5] Edmon Begoli, Tanmoy Bhattacharya, Dimitri Kusnezov (2019), [The need for uncertainty quantification in machine-assisted medical decision making](https://www.nature.com/articles/s42256-018-0004-1), *Nature Machine Intelligence* Vol 1, p.20–23
