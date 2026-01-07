library(AER)
library(stargazer)
data("HMDA")
attach(HMDA)

###### 1. линейная вероятностная модель (LPM) #####

# обычная модель лин. регрессии, но Y интерпретируем не как условное 
# матожидание по регрессорам, а как условную вероятность

summary(HMDA)

HMDA$deny <- as.numeric(HMDA$deny)-2 # превращаем да/нет в 1 и 0
denymod1 <- lm(deny ~ pirat, data = HMDA)
coeftest(denymod1, vcov. = vcovHC, type="HC1")

# plot the data
plot(x = HMDA$pirat, 
     y = HMDA$deny,
     main = "Scatterplot Mortgage Application Denial and 
                                    the Payment-to-Income Ratio",
     xlab = "P/I ratio",
     ylab = "Deny",
     pch = 20,
     ylim = c(-0.4, 1.4),
     cex.main = 0.8)

# add horizontal dashed lines and text
abline(h = 1, lty = 2, col = "darkred")
abline(h = 0, lty = 2, col = "darkred")
text(2.5, 0.9, cex = 0.8, "Mortgage denied")
text(2.5, -0.1, cex= 0.8, "Mortgage approved")

# add the estimated regression line
abline(denymod1, 
       lwd = 1.8, 
       col = "steelblue")
# видим, что модель "неуважительно относится к природе вероятности" - предсказывает значения вероятности >1 и <0

coeftest(denymod1, vcov. = vcovHC, type = "HC1")
#             Estimate Std. Error   t value  Pr(>|t|)    
# (Intercept) -0.079910   0.031967 -2.4998   0.01249 *  
#   pirat     0.603535    0.098483  6.1283    1.036e-09 ***
# Интерпретация: увеличение pirat на 1 процентный пункт приводит к увеличению вероятности отказа в кредите на 0,06%


# rename the variable 'afam' for consistency
colnames(HMDA)[colnames(HMDA) == "afam"] <- "black"

# estimate the model
denymod2 <- lm(deny ~ pirat + black, data = HMDA)
coeftest(denymod2, vcov. = vcovHC)

#   Estimate     Std. Error t value   Pr(>|t|)    
#   (Intercept) -0.090514   0.033430 -2.7076  0.006826 ** 
#   pirat        0.559195   0.103671  5.3939 7.575e-08 ***
#   blackyes     0.177428   0.025055  7.0815 1.871e-12 ***

# Даже при постоянном pirat, black увеличивает вероятность отказа в выдаче ипотечного кредита примерно на 17,7%




###### 2. probit-модель #####

# кумулятивная стандартная функция нормального распределения Φ(⋅)
# используется для моделирования функции регрессии, когда зависимая переменная является бинарной

denyprobit <- glm(deny ~ pirat, 
                  family = binomial(link = "probit"), 
                  data = HMDA)

coeftest(denyprobit, vcov. = vcovHC, type = "HC1")
#   Estimate     Std. Error z value   Pr(>|z|)    
#   (Intercept) -2.19415    0.18901   -11.6087 < 2.2e-16 ***
#   pirat        2.96787    0.53698   5.5269 3.259e-08 ***

# значимость pirat - есть
# знак - положительный (увеличение pirat приводит к росту вероятности отказа)
# интерпретация эффекта - научимся потом

# plot data
plot(x = HMDA$pirat, 
     y = HMDA$deny,
     main = "Probit Model of the Probability of Denial, Given P/I Ratio",
     xlab = "P/I ratio",
     ylab = "Deny",
     pch = 20,
     ylim = c(-0.4, 1.4),
     cex.main = 0.85)

# add horizontal dashed lines and text
abline(h = 1, lty = 2, col = "darkred")
abline(h = 0, lty = 2, col = "darkred")
text(2.5, 0.9, cex = 0.8, "Mortgage denied")
text(2.5, -0.1, cex= 0.8, "Mortgage approved")

# add estimated regression line
x <- seq(0, 3, 0.01)
y <- predict(denyprobit, list(pirat = x), type = "response")

lines(x, y, lwd = 1.5, col = "steelblue")

# видим, что большая часть кривой - это линейная зависимость, поэтому LPM тоже релевантен
# отличия кроются в хвостах

# для интерпретации используем predict
# 1. compute predictions for P/I ratio = 0.3, 0.4
predictions <- predict(denyprobit, 
                       newdata = data.frame("pirat" = c(0.3, 0.4)),
                       type = "response")

# 2. Compute difference in probabilities
diff(predictions)
# 2 
# 0.06081433 
# увеличение доли платежа с 30% до 40% увеличивает вероятность отказа на 6%

predictions2 <- predict(denyprobit, 
                        newdata = data.frame("pirat" = c(0.8, 0.9)),
                        type = "response")

diff(predictions2)
# 2 
# 0.1118139 
# увеличение доли платежа с 80% до 90% увеличивает вероятность отказа на 11%

denyprobit2 <- glm(deny ~ pirat + black, 
                   family = binomial(link = "probit"), 
                   data = HMDA)

coeftest(denyprobit2, vcov. = vcovHC, type = "HC1")
#   Estimate    Std. Error  z value   Pr(>|z|)    
#   (Intercept) -2.258787   0.176608 -12.7898 < 2.2e-16 ***
#   pirat        2.741779   0.497673   5.5092 3.605e-08 ***
#   blackyes     0.708155   0.083091   8.5227 < 2.2e-16 ***
# вывод: чернокожим чаще отказывают, насколько - сейчас посчитаем

# 1. compute predictions for P/I ratio = 0.3
predictions <- predict(denyprobit2, 
                       newdata = data.frame("black" = c("no", "yes"), 
                                            "pirat" = c(0.3, 0.3)),
                       type = "response")

# 2. compute difference in probabilities
diff(predictions)
# 2 
# 0.1578117 
# черным отказывают на 15.7% чаще
###### 3. logit-модель #####
denylogit <- glm(deny ~ pirat, 
                 family = binomial(link = "logit"), 
                 data = HMDA)

coeftest(denylogit, vcov. = vcovHC, type = "HC1")
#   Estimate Std. Error     z value  Pr(>|z|)    
#   (Intercept) -4.02843    0.35898 -11.2218 < 2.2e-16 ***
#   pirat        5.88450    1.00015   5.8836 4.014e-09 ***

# plot data
plot(x = HMDA$pirat, 
     y = HMDA$deny,
     main = "Probit and Logit Models of the Probability of Denial, Given P/I Ratio",
     xlab = "P/I ratio",
     ylab = "Deny",
     pch = 20,
     ylim = c(-0.4, 1.4),
     cex.main = 0.9)

# add horizontal dashed lines and text
abline(h = 1, lty = 2, col = "darkred")
abline(h = 0, lty = 2, col = "darkred")
text(2.5, 0.9, cex = 0.8, "Mortgage denied")
text(2.5, -0.1, cex= 0.8, "Mortgage approved")

# add estimated regression line of Probit and Logit models
x <- seq(0, 3, 0.01)
y_probit <- predict(denyprobit, list(pirat = x), type = "response")
y_logit <- predict(denylogit, list(pirat = x), type = "response")

lines(x, y_probit, lwd = 1.5, col = "steelblue")
lines(x, y_logit, lwd = 1.5, col = "black", lty = 2)

# add a legend
legend("topleft",
       horiz = TRUE,
       legend = c("Probit", "Logit"),
       col = c("steelblue", "black"), 
       lty = c(1, 2))

# добавим дискриминацию

denylogit2 <- glm(deny ~ pirat + black, 
                  family = binomial(link = "logit"), 
                  data = HMDA)

coeftest(denylogit2, vcov. = vcovHC, type = "HC1")
#   Estimate    Std. Error  z value  Pr(>|z|)    
#   (Intercept) -4.12556    0.34597 -11.9245 < 2.2e-16 ***
#   pirat        5.37036    0.96376   5.5723 2.514e-08 ***
#   blackyes     1.27278    0.14616   8.7081 < 2.2e-16 ***
# и снова: эффект значим, он положительный, проинтерпретировать нельзя

predictions <- predict(denylogit2, 
                       newdata = data.frame("black" = c("no", "yes"), 
                                            "pirat" = c(0.3, 0.3)),
                       type = "response")

predictions
# 1          2 
# 0.07485143 0.22414592 
# вероятность, что откажут белому - 7%; черному - 22%


diff(predictions)
# 2 
# 0.1492945 
# черным отказывают на 14.9% чаще

# Экономисты чаще используют пробит, а медики и эпидемиологи - логит (ТИС - так исторически сложилось)

###### 

# я буду тестировать LPM

# смотрим зависимость отказов от расы
lpm1 <- lm(deny ~ black, data = HMDA)
coeftest(lpm1, vcov. = vcovHC, type = "HC1")

# добавим экономические факторы
lpm2 <- lm(deny ~ black + pirat + hirat + lvrat, data = HMDA)
coeftest(lpm2, vcov. = vcovHC, type = "HC1")

# добавим кредитную историю
lpm3 <- lm(deny ~ black + pirat + hirat + lvrat + chist + mhist + phist, data = HMDA)
coeftest(lpm3, vcov. = vcovHC, type = "HC1")

# добавим соцдем
lpm4 <- lm(deny ~ black + pirat + hirat + lvrat + chist + mhist + phist + unemp + selfemp + single + hschool, data = HMDA)
coeftest(lpm4, vcov. = vcovHC, type = "HC1")

# добавим все остальное
lpm5 <- lm(deny ~ black + pirat + hirat + lvrat + chist + mhist + phist + unemp + selfemp + single + hschool + condomin + insurance, data = HMDA)
coeftest(lpm5, vcov. = vcovHC, type = "HC1")

rob_se <- list(sqrt(diag(vcovHC(lpm1, type="HC1"))),
               sqrt(diag(vcovHC(lpm2, type="HC1"))),
               sqrt(diag(vcovHC(lpm3, type="HC1"))),
               sqrt(diag(vcovHC(lpm4, type="HC1"))),
               sqrt(diag(vcovHC(lpm5, type="HC1"))))

stargazer(lpm1,lpm2,lpm3,lpm4,lpm5,
          digits=2,
          type="text",
          header = FALSE,
          se = rob_se,
          model.numbers = FALSE,
          column.labels = c("1",'2','3','4','5'))

# 2. смотрим logit
logit_onlyblack <- glm(deny ~ black, 
                       family = binomial(link = "logit"), 
                       data = HMDA)
install.packages("mfx")
library(mfx)

# оцениваем предельный эффект переменной black в моделях логистических регрессий
logitmfx(formula = deny ~ black, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)
logitmfx(formula = deny ~ black + pirat + hirat + lvrat, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)
logitmfx(formula = deny ~ black + pirat + hirat + lvrat + chist + mhist + phist, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)
logitmfx(formula = deny ~ black + pirat + hirat + lvrat + chist + mhist + phist + unemp + selfemp + single + hschool, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)
logitmfx(formula = deny ~ black + pirat + hirat + lvrat + chist + mhist + phist + unemp + selfemp + single + hschool + condomin + insurance, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)

# то же самое для пробита
probitmfx(formula = deny ~ black, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)
probitmfx(formula = deny ~ black + pirat + hirat + lvrat, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)
probitmfx(formula = deny ~ black + pirat + hirat + lvrat + chist + mhist + phist, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)
probitmfx(formula = deny ~ black + pirat + hirat + lvrat + chist + mhist + phist + unemp + selfemp + single + hschool, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)
probitmfx(formula = deny ~ black + pirat + hirat + lvrat + chist + mhist + phist + unemp + selfemp + single + hschool + condomin + insurance, data = HMDA, atmean = TRUE, robust = TRUE, clustervar1 = NULL, clustervar2 = NULL, start = NULL)

