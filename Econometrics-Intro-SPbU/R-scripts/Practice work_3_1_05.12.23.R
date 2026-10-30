# Исследовательский вопрос: правда ли там, где существует запрет на курение на рабочем месте, 
# люди меньше курят?
# Гипотеза: там где существует запрет на курение на рабочем месте, люди курят меньше.
# Механизм: микроэкономика

install.packages("vtable")
install.packages("mfx")
library(vtable)
library(AER)
library(mfx) # пакет для логита
data(SmokeBan)
attach(SmokeBan)
st(SmokeBan)
# Стандартный способ представления описательных статистик для категориальных переменных:
# таблица частотности (frequency table)
# 1. Высшее 2. Техникум 3. ПТУ 4. Среднее ; Закодировано как 1, 2, 3, 4 - бред же среднее смотреть
# А у нас все круто - есть frequency

# Describe data
# 1) Statistics summary
st(SmokeBan)
# 2) The scatter plot
plot(age, smoker)
# 3) Таблица сопряженности: описание категориальных переменных
tab = table(ban, smoker)
tab
#     smoker
# ban   no  yes
#   no  2772 1130
#   yes 4805 1293

# 4) Преобразование переменных : превращаем да/нет в 1 и 0
SmokeBan$smoker = as.numeric(SmokeBan$smoker)-1

## LPM
LPM1 = lm(smoker ~ ban, data = SmokeBan)
coeftest(LPM1, vcov. = vcovHC, type = "HC1")
#         Estimate   
#(Intercept)  0.2895951  
#  banyes      -0.0775583
# Вывод: бан на курение на рабочем месте влияет отрицательно на то, курит ли человек;
# вероятность того, что человек курит, меньше на 8% (-0.0775583)

# Введем контрольные переменные

LPM2 = lm(smoker ~ ban + as.factor(gender), data = SmokeBan)
coeftest(LPM2, vcov. = vcovHC, type = "HC1")
# слегка падает эффект

LPM3 = lm(smoker ~ ban + as.factor(gender) + age, data = SmokeBan)
coeftest(LPM3, vcov. = vcovHC, type = "HC1")


LPM4 = lm(smoker ~ ban + as.factor(gender) + age + as.factor(education), data = SmokeBan)
coeftest(LPM4, vcov. = vcovHC, type = "HC1")

# Добавим расы afam и hispanic
LPM5 = lm(smoker ~ ban + as.factor(gender) + age + as.factor(education) + afam + hispanic, 
          data = SmokeBan)
coeftest(LPM5, vcov. = vcovHC, type = "HC1")

## Logit
# Оставим пакет, чтобы посчитать предельные издержки
# Считаем предельные эффекты, чтобы оценить истинную значимость и знак
# Если нам необходимо оценить именно его величину
library(mfx)

logit1 = glm(smoker ~ ban, 
             family = 'binomial' (link = 'logit'), 
             data = SmokeBan)

logitmfx(formula = smoker ~ ban, data = SmokeBan, atmean = TRUE, robust = TRUE)

logitmfx(formula = smoker ~ ban + as.factor(gender), data = SmokeBan, atmean = TRUE, robust = TRUE)

logitmfx(formula = smoker ~ ban + as.factor(gender) + age, 
         data = SmokeBan, atmean = TRUE, robust = TRUE)


logitmfx(formula = smoker ~ ban + as.factor(gender) + age + as.factor(education), 
         data = SmokeBan, atmean = TRUE, robust = TRUE)

logitmfx(formula = smoker ~ ban + as.factor(gender) + age + as.factor(education) + afam + hispanic, 
         data = SmokeBan, atmean = TRUE, robust = TRUE)

# поговорим про герогенность эффекта
# на разные социальные группы эффект влияет по-разному
# Мы хотим выяснить, что запрет на курение действует по разному на мужчин и женщин

logitmfx(formula = smoker ~ ban + as.factor(gender) * ban + age + as.factor(education) + afam + hispanic, 
         data = SmokeBan, atmean = TRUE, robust = TRUE)
###недодеделано
logitmfx(formula = smoker ~ ban + as.factor(gender) * ban + age + as.factor(education) + ban * afam + hispanic, 
         data = SmokeBan, atmean = TRUE, robust = TRUE)
###недоделано
logitmfx(formula = smoker ~ ban + as.factor(gender) * ban + age + as.factor(education) + ban * afam + hispanic, 
         data = SmokeBan, atmean = TRUE, robust = TRUE)









