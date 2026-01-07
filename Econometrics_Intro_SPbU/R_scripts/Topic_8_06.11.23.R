install.packages("AER")
library(AER)
data("CigarettesSW")
summary(CigarettesSW)

# оцениваем эластичность спроса по цене, значит логарифмируем X и Y

CigarettesSW$rprice <- with(CigarettesSW, price/cpi) #with - панельные данные, сопоставляет год с годом (наверное???)
CigarettesSW$salestax <- with(CigarettesSW, (taxs - tax)/cpi)
cor(CigarettesSW$salestax, CigarettesSW$price)
# 0.6141228 - п

c1995 <- subset(CigarettesSW, year == "1995")

cig_s1 <- lm(log(rprice)~salestax, data = c1995)
coeftest(cig_s1, vcov. = vcovHC, type = "HC1")
#t test of coefficients:
  
#  Estimate Std. Error  t value  Pr(>|t|)    
#(Intercept) 4.6165463  0.0289177 159.6444 < 2.2e-16 ***
#  salestax    0.0307289  0.0048354   6.3549 8.489e-08 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# *** - есть релевантность (первое условие), т.к. стат. значимое
# экзогенность - налог влияет на прибыль, но не влияет на спрос (никак не можем проверить, думаем логически)

lcigp_pred <- cig_s1$fitted.values #вытащили предсказанные значения
cig_s2 <- lm(log(c1995$packs)~lcigp_pred)
coeftest(cig_s2, vcov=vcovHC)
# цена стат значимо влияет на спрос: при увеличении цены на 1% спрос уменьшится на 1,08359 


cig_s3 <- lm(log(c1995$packs)~log(c1995$price))
coeftest(cig_s3, vcov=vcovHC)
# посмотрели, как с лоховской моделью сработало бы - смещенная оценка отличается на 0,21


cig_ivreg <- ivreg(log(packs)~log(rprice) | salestax, data = c1995)
coeftest(cig_ivreg, vcov = vcovHC, type = "HC1")


CigarettesSW$rincome <- with(CigarettesSW, income/population/cpi)
c1995 <- subset(CigarettesSW, year == "1995")
cig_ivreg2 <- ivreg(log(packs) ~ log(rprice) + log(rincome) | log(rincome) + salestax, data = c1995)
coeftest(cig_ivreg2, vcov = vcovHC, type = "HC1")

CigarettesSW$sigtax <- with(CigarettesSW, tax/cpi)
c1995 <- subset(CigarettesSW, year == "1995")
cig_ivreg3 <- ivreg(log(packs) ~ log(rprice) + log(rincome) | log(rincome) + salestax + sigtax , data = c1995)
coeftest(cig_ivreg3, vcov = vcovHC, type = "HC1")

c1985 <- subset(CigarettesSW, year == "1985")
packsdiff <- log(c1995$packs) - log(c1985$packs)
pricediff <- log(c1995$price/c1995$cpi) - log(c1985$price/c1985$cpi)
incomediff <- log(c1995$income/c1995$population/c1995$cpi) - log(c1985$income/c1985$population/c1985$cpi)
salestaxdiff <- (c1995$taxs - c1995$tax)/c1995$cpi - (c1985$taxs - c1985$tax)/c1985$cpi
cigtaxdiff <- c1995$tax/c1995$cpi - c1985$tax/c1985$cpi

cig_ivreg_diff1 <- ivreg(packsdiff ~ pricediff + incomediff | incomediff + salestaxdiff)
cig_ivreg_diff2 <- ivreg(packsdiff ~ pricediff + incomediff | incomediff + cigtaxdiff)
cig_ivreg_diff3 <- ivreg(packsdiff ~ pricediff + incomediff | incomediff + cigtaxdiff + salestaxdiff)

coeftest(cig_ivreg_diff1, vcov. = vcovHC, type = "HC1")
coeftest(cig_ivreg_diff2, vcov. = vcovHC, type = "HC1")
coeftest(cig_ivreg_diff3, vcov. = vcovHC, type = "HC1")

rob_se <- list(sqrt(diag(vcovHC(cig_ivreg_diff1, type = "HC1"))),
               sqrt(diag(vcovHC(cig_ivreg_diff2, type = "HC1"))),
               sqrt(diag(vcovHC(cig_ivreg_diff3, type = "HC1"))))

install.packages("stargazer")
library(stargazer)
library(sandwich)
stargazer(cig_ivreg_diff1, cig_ivreg_diff2,cig_ivreg_diff3,type = "text", se = rob_se)

# ===========================================================
#                                  Dependent variable:     
#                               -----------------------------
#                                        packsdiff          
#                                 (1)       (2)       (3)   
# -----------------------------------------------------------
#   pricediff                     -0.938*** -1.343*** -1.202***
#                                 (0.208)   (0.229)   (0.197) 
# 
#  incomediff                      0.526     0.428     0.462  
#                                 (0.339)   (0.299)   (0.309) 
# 
#  Constant                       -0.118*   -0.017    -0.052  
#                                 (0.068)   (0.067)   (0.062) 
# 
# -----------------------------------------------------------
#   Observations                     48        48        48    
#   R2                              0.550     0.520     0.547  
#   Adjusted R2                     0.530     0.498     0.526  
#   Residual Std. Error (df = 45)   0.091     0.094     0.091  
# ===========================================================
#   Note:                           *p<0.1; **p<0.05; ***p<0.01


# начнем с OLS - МНК (ordinary list squares)
data(AJR2001)
attach(AJR2001)
ols <- lm(loggdp ~ risk)
coeftest(ols,vcov. = vcovHC, type = "HC1")

# введем инструмент - смертность поселенцев
ivreg <- ivreg (loggdp ~ risk | logmort0)

#Coefficients:
#  (Intercept)         risk  
#1.9943       0.9295 

coeftest(ivreg, vcov. = vcovHC, type="HC1")

# t test of coefficients:
#   
#              Estimate Std. Error t value  Pr(>|t|)    
# (Intercept)  1.99430    1.15315  1.7294   0.08871 .  
# risk         0.92949    0.17281  5.3787 1.211e-06 ***

step1 <- lm(risk ~ logmort0)
summary(step1) # из всей вариации риска инструмент объясняет 0.27 (R squared)
coeftest(step1, vcov. = vcovHC, type="HC1")
# в чем смысл: смотрим, как нынешние институты предсказываются смертностью поселенцев 
# выполнено первое условие: инструмент можно применять (не ссылаемся на здравый смысл)



