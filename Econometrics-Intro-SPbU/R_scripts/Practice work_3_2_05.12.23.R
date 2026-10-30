
##### #### - удобная фича, позволяет скрыть часть кода

# summary stats
st(default)
attach(default)
hist(INCOME)
# стандартное логнормальное распределение
# same here
hist(ADDRESS)
hist(INCOME)
hist(default$DEBT)
plot(INCOME, ADDRESS)

# Divide sample (разделяем выборку на 2); круто, но больше так делать не будем :) 
sample_size = floor (6/7 * nrow(default))
set.seed(101)
train_ind = sample(seq_len(nrow(default)), size = sample_size)
train = default[train_ind, ]
test = default[-train_ind, ]

## LPM

LPM1 = lm(DEFAULT ~ DTI, data = train)

LPM2 = lm(DEFAULT ~ DTI + log(DEBT) + log(INCOME), data = train)

LPM3 = lm(DEFAULT ~ DTI + log(DEBT) + log(INCOME) + AGE + EDUCATION, data = train)

LPM4 = lm(DEFAULT ~ DTI + log(DEBT) + log(INCOME) + AGE + EDUCATION + log(ADDRESS + 0.0001) + log(EMPLOY + 0.0001), 
          data = train)
# Посмотрим на summary
summary(LPM1)
summary(LPM2)
summary(LPM3)
summary(LPM4)
summary(LPM5)

########## Качество предсказаний ####
predict = predict(LPM4, newdata = test, type = 'response')

# Д.В.Кислицын СДАЛСЯ, код вышлет потом
treshold = 0.5
test = mutate(def.prob = predict(LMP4, newdata = test, type = 'response'))
#____________________________________________________________________________

install.packages("pROC")
library("pROC")
# Хотим представить ROC-кривую 
roc_score = (test, predict) 
#_________________________________

# Делаем все то же самое, только теперь пробит; probid and pseudo R-squared
probit1 = glm(DEFAULT ~ DTI,
              family = binomial(link = 'probit'),
              data = train)

probit2 = glm(DEFAULT ~ DTI + log(DEBT) + log(INCOME),
              family = binomial(link = 'probit'),
              data = train)

probit3 = glm(DEFAULT ~ DTI + log(DEBT) + log(INCOME) + AGE + EDUCATION,
              family = binomial(link = 'probit'),
              data = train)

probit4 = glm(DEFAULT ~ DTI + log(DEBT) + log(INCOME) + AGE + EDUCATION + log(ADDRESS + 0.0001) + log(EMPLOY + 0.0001),
              family = binomial(link = 'probit'),
              data = train)

probit_null = glm(formula = DEFAULT ~1,
                  family = binomial(link = 'probit'),
                                    data = train)

# Так как не можем посчитать обычный R-squared, считаем псевдо
1 - logLik(probit1)[1] / logLik(probit_null)[1]              
1 - logLik(probit2)[1] / logLik(probit_null)[1]  
1 - logLik(probit3)[1] / logLik(probit_null)[1]                                      
1 - logLik(probit4)[1] / logLik(probit_null)[1]                  

##### AIC ####
#AIC: Akaike's An Information Criterion
install.packages("AICmodavg")
AIC(LPM1)
AIC(LPM2)
AIC(LPM3)
AIC(LPM4)

AIC(probit1)
AIC(probit2)
AIC(probit3)
AIC(probit4)


