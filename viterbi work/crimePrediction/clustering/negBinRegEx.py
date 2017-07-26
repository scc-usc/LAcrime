"""
Negative binomial regression example
"""


import statsmodels.api as sm

endog = [2, 2, 3]
exog = [[1,2],[2,4],[5,6]]

#endog1 = [1,2,3]
exog1 = [[1,2],[2,4]]

nbm = sm.NegativeBinomial(endog, sm.add_constant(exog),loglike_method='nb2')

res = nbm.fit()

#print(res.summary())

pred = list(res.predict(exog1))
print pred
