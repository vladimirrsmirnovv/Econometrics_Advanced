################################### 28.11

# исследовательский вопрос: правда ли, что развитые институты влияют на экономическое развитие?
# исследовательская гипотеза: страны с более низким риском экспроприации имеют больший подушный ВВП (нулевая гипотеза - "связи нет", мы должны ее опровергнуть)

# модель парной линейной регрессии: log_gdp = b0 + b1_качество_институтов + u
# проблема эндогенности (нарушение первой предпосылки о равенстве условного матожидания ошибки нулю). последствие: смещение оценок
# причина нарушения эндогенности 1: потенциально пропущенные переменные: тип экономики, менталитет, полит. устройство
# причина нарушения эндогенности 2: одновременная причинность (расходы на полицию и преступность, например)
# причина нарушения эндогенности 3: нарушение валидности

# самое сложное и творческое, что есть, - найти хороший инструмент
# в статье инструментом является смертность ранних поселенцев

############ проведем разведывательный анализ данных #########
# 1. строим таблицу описательных статистик
install.packages("vtable")
library(vtable)
st(AJR2001)
# 2. строим графики, описывающие изучаемую взаимосвязь
scatterplot(risk, loggdp)
# оцениваем качество инструмента (видим, что инструмент релевантен)
scatterplot(logmort0, risk)

############ регресионный анализ ###########
# 1. строим наивную модель
ols <- lm(loggdp ~ risk)
coeftest(ols,vcov. = vcovHC, type = "HC1")
# 2. строим двухшаговую модель
ivreg <- ivreg (loggdp ~ risk | logmort0)
coeftest(ivreg, vcov. = vcovHC, type="HC1")
# вывод: оценка в наивном методе занижена

# добавим в двухшадовый МНК контрольную переменную - широту
ivreg2 <- ivreg(loggdp ~ risk + latitude | logmort0 + latitude)
coeftest(ivreg2, vcov. = vcovHC, type="HC1")

# добавим 3 дамми с континентами
ivreg3 <- ivreg(loggdp ~ risk + latitude + asia + africa + other| logmort0 + latitude + asia + africa + other)
coeftest(ivreg3, vcov. = vcovHC, type="HC1")



############################### часть 2: ЗП и образование ############################
# исследовательсикй вопрос: правда ли, что дополнительный год обучения приводит к тому, что ЗП растет?
# осозанно используем слово "приводит", т.к. используем метод инструментальных переменных, кот. помогает увидеть причинность
# гипотеза: при увеличении уровня образования растет ЗП

# наивная модель: ЗП ~ образование
data(Card1993)
attach(Card1993)
install.packages("vtable")
library(vtable)
st(Card1993)

ols <- lm(lwage76 ~ ed76)
coeftest(ols, vcov. = vcovHC, type="HC1")
plot(lwage76, ed76)

# добавляем образование родителей в качестве контроля
ols2 <- lm(lwage76 ~ ed76 + daded + momed)
coeftest(ols2, vcov. = vcovHC, type="HC1")
# когда добавляем образование родителей, значимость образования человека падает

# добавляем образование родителей в качестве контроля + soc-dem + region
ols3 <- lm(lwage76 ~ ed76 + daded + momed + black + marsta76 + age76 + sinmom14 + south66)
coeftest(ols3, vcov. = vcovHC, type="HC1")
# log-lin модель: 1 год образования дает рост дохода на 3.9%

step1 <- lm(ed76 ~ nearc2 + nearc4)
coeftest(step1, vcov. = vcovHC, type="HC1")

# оцениваем модели с инструментами
iv2 <- ivreg(lwage76 ~ ed76 | nearc2)
coeftest(iv2, vcov. = vcovHC, type="HC1")

iv4 <- ivreg(lwage76 ~ ed76 | nearc4)
coeftest(iv4, vcov. = vcovHC, type="HC1")

# добавим второй инструмент
iv12 <- ivreg (lwage76 ~ ed76 | nearc2 + nearc4)
coeftest(iv12, vcov. = vcovHC, type = "HC1")

### IV + база соцдем

iv3 <- ivreg (lwage76 ~ ed76 + black + marsta76 + age76 + south66 | nearc2 + nearc4 
              + black + marsta76 + age76 + south66)
coeftest(iv3, vcov. = vcovHC, type = "HC1")

### IV + база соцдем + качества семьи

iv4 <- ivreg (lwage76 ~ ed76 + black + marsta76 + age76 + south66 + momed + daded + sinmom14 | nearc2 + nearc4 + black + marsta76 + age76 + south66 + momed + daded + sinmom14)
coeftest(iv4, vcov. = vcovHC, type = "HC1")
