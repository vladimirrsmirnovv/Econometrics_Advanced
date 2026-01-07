install.packages('zoo')
install.packages("car")
install.packages("margins")
library(margins)
library(car)
library(AER)
library(sandwich)
library(lmtest)
data("CASchools")

CASchools$STR <- CASchools$students/CASchools$teachers
CASchools$score <- (CASchools$read + CASchools$math)/2
model = lm(score ~ STR + english, data = CASchools)
model
summary(model)

coeftest(model, vcov. = vcovHC, type = "HC1")
confint(model)
# (-1.8487969  -0.3537944) - отвергаем гипотезу H0, STR не входит, так как 0 не входит
# Ошибка - неробастный доверительный интервал
confint(model, level = 0,9)

#compute robust standard errors
rob_se = diag(vcovHC(model, type = "HC1"))^0.5

#compute robust 95% confidence intervals
rbind("lower" = coef(model) - qnorm(0.975) * rob_se,
      "upper" = coef(model) + qnorm(0.975) * rob_se)
#(Intercept)        STR    english
#lower    668.9252 -1.9496606 -0.7105980
#upper    703.1393 -0.2529307 -0.5889557

#compute robust 90% confidence intervals
rbind("lower" = coef(model) - qnorm(0.95) * rob_se,
      "upper" = coef(model) + qnorm(0.95) * rob_se)


model1 = lm(score ~ STR + english + expenditure, data = CASchools)
model1
coeftest(model1, vcov. = vcovHC, type = "HC1")

#t test of coefficients:
  
#  Estimate  Std. Error  t value Pr(>|t|)    
#(Intercept) 649.5779466  15.4583437  42.0212  < 2e-16 ***
#  STR          -0.2863991   0.4820728  -0.5941  0.55277    
#english      -0.6560227   0.0317844 -20.6398  < 2e-16 ***
#  expenditure   0.0038679   0.0015807   2.4469  0.01482 *  
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Расходы на ученика связаны с более высокими результатами на тесте, так как H0 отвергли
# если p-value меньше 0.05, отвергаем нулевую гипотезу о незначимости переменной
# чтобы увеличить успеваемость, надо перестать кормить детей, что тупо


model_3 = lm(score ~ STR + english + expenditure + lunch + income + calworks, data = CASchools)
coeftest(model_3, vcov. = vcovHC, type = "HC1")


#execute the function on the model object and privde both linear restrictions
#to be tested as strings
#install.packages("car")
#library(car)

#model1 = lm(score ~ STR + english + expenditure, data = CASchools)
linearHypothesis(model1, c("STR=0", "expenditure=0"))
#Linear hypothesis test

#Hypothesis:
#  STR = 0
#expenditure = 0

#Model 1: restricted model
#Model 2: score ~ STR + english + expenditure

#Res.Df   RSS Df Sum of Sq      F   Pr(>F)    
#1    418 89000                                 
#2    416 85700  2    3300.3 8.0101 0.000386 ***
#  ---
 # Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 

#гипотеза 0: оба коэффициента не влияют, отвергаем

#Интервал без робастных ошибок
confidenceEllipse(model1,
                  fill = T,
                  lwd = 5,
                  which.coef = c("STR", "expenditure"), # содержатальная строчка
                  main = "95% Confidence Set",
                  ylab = "Coefficients of Expenditure",
                  xlab = "Coefficients of STR")
#посмотрели по эллипсу: ноль не входит, гипотезу отвергаем
#построим то же с робастными ошибками                                 

confidenceEllipse(model1,
                  fill = T,
                  lwd = 3,
                  which.coef = c("STR", "expenditure"), # содержатальная строчка
                  main = "95% Confidence Set",
                  vcov. = vcovHC(model1, type = "HC1"),
                  col = "red",
                  ylab = "Coefficients of Expenditure",
                  xlab = "Coefficients of STR")

# Наложим робастный на неробастный
confidenceEllipse(model1,
                  fill = T,
                  lwd = 1,
                  which.coef = c("STR", "expenditure"),
                  add = T)

attach(CASchools)
hist(STR)
#english = доля детей, для кот-х англ. не является родным
hist(english)
CASchools$logenglish = log(CASchools$english + 0.0001)
hist(logenglish)
plot(CASchools$score, CASchools$english)
#посмотрели влияние логарифма инглиша 
model_log = lm(score ~ STR + logenglish + expenditure + lunch + income + calworks, data = CASchools)
coeftest(model_log, vcov. = vcovHC, type = "HC1")
# Это модель log-level
# При увеличении процента english на 1 % пункт результаты по тестам снижаются на 
# 0.05
CASchools$englishq = CASchools$english^2
hist(CASchools$englishq)

model_q = lm(score ~ STR + english + englishq + expenditure + lunch + income + calworks, data = CASchools)
coeftest(model_q, vcov. = vcovHC, type = "HC1")
#посмотрели модель с квадратом английского - он не значим, можно использовать линейную модель
#проведем тест на совместную значимость
linearHypothesis(model_q, c("english=0", "englishq=0"))
#Linear hypothesis test

#Hypothesis:
#  english = 0
#englishq = 0

#Model 1: restricted model
#Model 2: score ~ STR + english + englishq + expenditure + lunch + income + 
#  calworks

#Res.Df   RSS Df Sum of Sq      F   Pr(>F)    
#1    414 32034                                 
#2    412 29094  2    2940.1 20.817 2.44e-09 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#гипотеза: при прочих равных доля детей, для которых английский не является родным,
# не влияет на успеваемость - гипотеза отвергается на высоком уровне значимости. 
# таким образом, английский имеет значение


hist(computer)
hist(CASchools$compsperone)
hist(expenditure)
hist(lunch)
hist(calworks)
hist(income)
hist(score)

CASchools$compsperone = computer/students

#calworks - доля детей, имеющих право на мат. помощь
model_4 = lm(I(log(score)) ~ STR + compsperone + I(log(english + 0.001)) + I(log(expenditure)) + 
  I(log(lunch + 0.001)) + I(log(income)) + I(log(calworks + 0.001)), data = CASchools)
coeftest(model_4, vcov. = vcovHC, type = "HC1")

#посмотрели модель с квадратом английского - он не значим, можно использовать линейную модель
# первое замечение: когда добавляются контрольные переменные про доход, STR теряет значимость
# calworks - доля детей, имеющих право на денежную помощь
# у нас log-log модель, то есть интерпретация %-% (ничего не домножаем)
# напомню: нулевая гипотеза, что переменная НЕ влияет


CASchools$CSR <- CASchools$computer/CASchools$students
hist(CASchools$CSR) #расчет удельного показателя компьютеров можно рассматривать как альтернативу лог-трансформации


# I - трансформация переменных, чтобы добавить нелинейные эффекты в модель
# Так как иначе R не поймет нашу идею с логарифмом?

linearHypothesis(model_4, c("STR=0", "compsperone=0", "expenditure=0"))
#При прочих равных, не влияют на успеваемость учеников - это отвергаем,
#Как минимум что-то одно из этого влияет


# УЧИМСЯ СЧИТАТЬ ПРЕДЕЛЬНЫЕ ЭФФЕКТЫ (ОЧЕНЬ ВАЖНО!!!)
#Какую модель используем?
# install.packages("margins")
# library(margins)
m = lm(I(log(score)) ~ STR + compsperone + + english + I(english^2) + I(log(expenditure)) + 
         I(log(lunch + 0.001)) + I(log(income)) + I(log(calworks + 0.001)), data = CASchools)
mfx = margins(model_4, variables = "english")
summary(mfx)

# summary(mfx)
# factor     AME     SE       z      p   lower   upper
# english -0.0012 0.0001 -9.5554 0.0000 -0.0014 -0.0009
 
# 0 не входит в границы доверительного интервала => 
# гипотезу 0 о его незначемости отвергаем (т.е. английский связан с score)
# AME - средний предельный эффект; он и логарифм - два лучших способа измерения влияния
# при увеличении инглиша на 1 п.п. результаты тестов меняются на -0,08% процента 
# (модель log (скор в логе) - lean (инглиш линейный))
# для немонотонной зависимости нет смысла оценивать предельный эффект, 
# т.к. + и - значения пердельного эффекта все усредняют


mfx2 = margins(model_4, variables = "english",
               at = list(income = c(10, 20, 30, 40, 50)))
summary(mfx2)

plot(income, score)
# граф интерпретация: чем больше доля, тем меньше наклон касательной

linearHypothesis(model_4, c("STR=0", "compsperone=0", "I(log(expenditure))=0"))







