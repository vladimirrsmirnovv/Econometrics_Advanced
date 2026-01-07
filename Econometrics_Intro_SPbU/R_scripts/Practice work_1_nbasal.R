install.packages("wooldridge")
install.packages("stargazer")
install.packages("lmtest")
install.packages("sandwich")
library("stargazer")
library("lmtest")
library(sandwich)

library("wooldridge")
data(nbasal)
#marr - наша переменная интереса

attach(nbasal) #сделать пометку о данной функции
#Переменная, которая меряет производительность (эфф) игрока
#Делаем переменную EFF как сумму игровых показателей для каждого игрока

EFF = (points + rebounds + assists)/minutes

simple = lm(EFF ~ marr)
simple
summary(simple)
#В среднем, если человек женат, то его эффективность хуже (wha?? типа, p value же большой man)

#Просто посчитаем по приколу
simple2 = lm(wage ~ marr)
summary(simple2)
#Правда, если баскетболист женат, то он зарабатывает больше на 317.69$

#проверим, зарабатывают ли мужчины старше больше
age_model <- lm(formula = wage~age, data= nbasal)
summary(age_model)

#гетероскетастичность - непостоянность дисперсии случайной ошибки модели (например, с увеличением значения независимой переменной растет и ошибка)

#Пункт про пропущенные переменные - например, известность баскетболиста в целом
#Что за переменные мы можем добавить?
#Гомоскедастичность и гетероскедастичность
#Гетероскедастичность не влияет на смещенность и несмещенность оценок
#Стандартная ошибка - именно то, что отвечает за разброс оценок, их распределение (?)
#Гетероскедастичность приводит (оценки по-прежнему несмещенные, распределение нормальное,
#только более широкое.
#робастные стандартные ошибки???

#С этого момента и робастных ошибок надо разобраться.....

#модель по первой группе - соц, демограф характеристики
model1 <- lm(formula = EFF~age + exper + black, data = nbasal)

# модель по второй группе - положение на поле
model2 <- lm(formula = EFF~guard + forward, data = nbasal)

# модель по третьей группе - смотрим, как складывалась карьера
model3 <- lm(formula = EFF~draft+coll, data = nbasal)

#______________________________________________________________________________

simple = lm(EFF ~ marr)
coeftest(simple, vcov. = vcovHC, type = "HC0")
cov0 = vcovHC(simple, type = "HC0")
se0 = sqrt(diag(cov0))


#Мы хотим экзогенные переменные - те, на которые точно не влияет наши переменные интереса
#1) age, exper, black 2) guard, forward, center 3) draft, college

#Модель
socdem = lm(EFF ~ marr + black + age + exper)
cov1 = vcovHC(socdem, type = "HC0")
se1 = sqrt(diag(cov1))


position = lm(EFF ~ marr + black + age + exper + guard + forward)
cov2 = vcovHC(position, type = "HC0")
se2 = sqrt(diag(cov2))

career = lm(EFF ~ marr + black + age + exper + guard + forward + draft + coll)
cov3 = vcovHC(career, type = "HC0")
se3 = sqrt(diag(cov3))

stargazer(simple, socdem, position, career,
          se = list(se0, se1, se2, se3),
          type = "text",
          title = "Результаты оценивания")
#Производительность защитников ниже на 0.005

#__________________________________________________________________________
#############оцениваем нелинейные эффекты в лин регрессии#######################

# например, убывающая отдача в микроэкономике
squaredexper <- lm(EFF ~ marr+black+age+exper+expersq+guard+forward+draft+coll, data = nbasal)
coeftest(squaredexper, vcov. = vcovHC, type="HC0")


squaredexper <- lm(EFF ~ marr+black+age+exper+I(exper^2)+guard+forward+draft+coll, data = nbasal)
coeftest(squaredexper, vcov. = vcovHC, type="HC0")

squaredexper <- lm(lwage ~ marr+black+age+exper+I(exper^2)+guard+forward+draft+coll, data = nbasal)
coeftest(squaredexper, vcov. = vcovHC, type="HC0")


# доказали, что с ростом стажа предельный рост доходов уменьшается (5 лет - прирост 140 долларов, 10 лет - прирост 40 долларов)


model = lm(score ~ STR + english, data = CASchools)
coeftest(model, vcov. = vcovHC, type = "HC1")
