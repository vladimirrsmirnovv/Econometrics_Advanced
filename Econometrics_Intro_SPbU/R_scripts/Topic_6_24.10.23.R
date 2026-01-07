install.packages("AER")
install.packages("sandwich")
library(AER)
library(sandwich)
library(lmtest)
data("Journals")
#Единица наблюдения - один журнал;

# define and rename variables
Journals$PricePerCitation = Journals$price/Journals$citations
Journals$Age = 2000 - Journals$foundingyear
Journals$Characters = Journals$charpp * Journals$pages/10^6
Journals$Subscriptions = Journals$subs

#compute summary statistics for price per citation
summary(Journals$PricePerCitation)
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# 0.005223  0.464495  1.320513  2.548455  3.440171 24.459459 

# ln(subs) = b0 + b1*ln(price)) + u

# Estimate models (I) - (IV)
Journals_mod1 = lm(log(Subscriptions) ~ log(PricePerCitation), 
                   data = Journals)

coeftest(Journals_mod1, vcov. = vcovHC, type = "HC1")

#Интерпретация:
#ln(subs) = 4,77 - 0,53*ln(price) + u
#При увеличении цены на 1% спрос уменьшается на 0.5%

Journals_mod2 = lm(log(Subscriptions) ~ log(PricePerCitation) + Age + Characters, 
                   data = Journals)

Journals_mod3 = lm(log(Subscriptions) ~ log(PricePerCitation) + Age + Characters + factor(field), 
                   data = Journals)
#Оценить робастные ошибки и сопоставить оценки в моделях
#Подумать про мультиколлинеарность в модели 2?

