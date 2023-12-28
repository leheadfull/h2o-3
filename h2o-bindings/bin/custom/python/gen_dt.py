options = dict(
)

doc = dict(
    __class__="""
Builds a Decision Tree (DT) on a preprocessed dataset.
"""
)
examples = dict(
    algorithm_params="""
>>> import h2o
>>> from h2o.estimators.infogram import H2OInfogram
>>> h2o.init()
>>> f = "https://erin-data.s3.amazonaws.com/admissible/data/taiwan_credit_card_uci.csv"
>>> col_types = {'SEX': "enum", 'MARRIAGE': "enum", 'default_payment_next_month': "enum"}
>>> df = h2o.import_file(path=f, col_types=col_types)
>>> train, test = df.split_frame(seed=1)
>>> y = "default_payment_next_month"
>>> x = train.columns
>>> x.remove(y)
>>> pcols = ["SEX", "MARRIAGE", "AGE"]
>>> ig = H2OInfogram(protected_columns=pcols)
>>> ig.train(y=y, x=x, training_frame=train)
>>> ig.plot()
"""
    data_fraction="""
>>> import h2o
>>> from h2o.estimators.infogram import H2OInfogram
>>> h2o.init()
>>> f = "https://erin-data.s3.amazonaws.com/admissible/data/taiwan_credit_card_uci.csv"
>>> col_types = {'SEX': "enum", 'MARRIAGE': "enum", 'default_payment_next_month': "enum"}
>>> df = h2o.import_file(path=f, col_types=col_types)
>>> train, test = df.split_frame(seed=1)
>>> y = "default_payment_next_month"
>>> x = train.columns
>>> x.remove(y)
>>> pcols = ["SEX", "MARRIAGE", "AGE"]
>>> ig = H2OInfogram(protected_columns=pcols, data_fraction=0.7)
>>> ig.train(y=y, x=x, training_frame=train)
>>> ig.plot()
"""
    net_information_thresholdn="""
>>> import h2o
>>> from h2o.estimators.infogram import H2OInfogram
>>> h2o.init()
>>> f = "https://erin-data.s3.amazonaws.com/admissible/data/taiwan_credit_card_uci.csv"
>>> col_types = {'SEX': "enum", 'MARRIAGE': "enum", 'default_payment_next_month': "enum"}
>>> df = h2o.import_file(path=f, col_types=col_types)
>>> train, test = df.split_frame(seed=1)
>>> y = "default_payment_next_month"
>>> x = train.columns
>>> x.remove(y)
>>> pcols = ["SEX", "MARRIAGE", "AGE"]
>>> ig = H2OInfogram(protected_columns=pcols, net_information_threshold=-1.0)
>>> ig.train(y=y, x=x, training_frame=train)
>>> ig.plot()

