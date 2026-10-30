library(readxl)

# data_model <- read_excel("C:/Users/user/Downloads/data_model.xlsx")

data_model <- read.csv("C:/Users/smirn/OneDrive/Рабочий стол/Panel Data Econometrics/data_model.csv")


install.packages("plm")
install.packages("hdm")
install.packages("ivreg")
library(plm)
library(dplyr)
library(ggplot2)
library(lmtest)
library(sandwich)
library(modelsummary)
library(AER)
#library(ivreg)
library(tidyr)
library(dplyr)
library(hdm)

#options(timeout = 6000)
#Sys.which("make")

#install.packages(c("glmnet","Matrix","stats","utils"))

#options(download.file.method = "libcurl")

#install.packages("hdm", repos = "https://cran.r-project.org/")

#дообработка данных
data <- filter(data_model, direction == 'export')
data$year <- as.numeric(as.character(data$year))
data$ln_gdp_partner <- as.numeric(as.character(data$ln_gdp_partner))
data$ln_dist <- as.numeric(as.character(data$ln_dist))
data$ln_value <- as.numeric(as.character(data$ln_value))
data$ln_price <- as.numeric(as.character(data$ln_price))
data$ln_quantity <- as.numeric(as.character(data$ln_quantity))
data$quantity <- as.numeric(as.character(data$quantity))
data$iip_partner <- as.numeric(as.character(data$iip_partner))
data$cpi_partner <- as.numeric(as.character(data$cpi_partner))


data_full <- data %>%
  complete(partner, year)
data <- data[order(data$partner, data$year), ]

pdata <- pdata.frame(data, index = c("importer", "year"))

#панельные графики
ggplot(data, aes(year, quantity)) +
  geom_point() +
  facet_wrap(~ partner) +
  scale_y_discrete(labels = NULL) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8))


ggplot(data, aes(year, ln_quantity)) +
  geom_point() +
  facet_wrap(~ partner) +
  scale_y_discrete(labels = NULL) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8))

ggplot(data, aes(year, price)) +
  geom_point() +
  facet_wrap(~ partner) +
  scale_y_discrete(labels = NULL) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8))

ggplot(data, aes(year, ln_price)) +
  geom_point() +
  facet_wrap(~ partner) +
  scale_y_discrete(labels = NULL) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8))

#делаем вывод, что в логарифмах лучше


cor(pdata$ln_gdp_partner, pdata$iip_partner, use = "complete.obs")
#тест на гетерогенность по времени
pool_model <- plm(ln_quantity ~ ln_price + ln_gdp_partner + ln_dist + iip_partner,
                  data = pdata, model = "pooling")

# FE модель по годам
FE_time <- plm(ln_quantity ~ ln_price + ln_gdp_partner + ln_dist + iip_partner,
               data = pdata,
               model = "within",
               effect = "time")

# F-тест: FE по годам vs Pool
f_time_test <- pFtest(FE_time, pool_model)
f_time_test

#Pool-модель адекватна



# выбор из Pool, FE, RE (по объектам) без фиксированных временных эффектов

# 1) Pool-модель (без учета индивидуальных эффектов стран)
pool_model <- plm(ln_quantity ~ ln_price + ln_gdp_partner + ln_dist + iip_partner,
                  data = pdata,
                  model = "pooling")

# 2) FE-модель по странам (fixed effects)
FE_model <- plm(ln_quantity ~ ln_price + ln_gdp_partner + ln_dist + iip_partner,
                data = pdata,
                model = "within",
                effect = "individual")

# 3) RE-модель по странам (random effects)
RE_model <- plm(ln_quantity ~ ln_price + ln_gdp_partner + ln_dist + iip_partner,
                data = pdata,
                model = "random",
                effect = "individual")

#Тест на наличие индивидуальных эффектов: F-тест FE vs Pool
f_test <- pFtest(FE_model, pool_model)
f_test

#Тест Хаусмана
h_test <- phtest(FE_model, RE_model)
h_test
#тут выбираем FE на страны



#ОБЩАЯ ТАБЛИЦА
library(broom)

# Функция для извлечения оценок
get_coefs <- function(model){
  coefs <- summary(model)$coefficients
  if(is.null(coefs)){ coefs <- summary(model)$coefficients } # на всякий случай
  data.frame(
    Variable = rownames(coefs),
    Estimate = coefs[,1],
    StdError = coefs[,2],
    row.names = NULL
  )
}

pool_coef <- get_coefs(pool_model)
FE_coef <- get_coefs(FE_model)
RE_coef <- get_coefs(RE_model)

# Объединяем в одну таблицу
library(dplyr)
coef_table <- full_join(pool_coef, FE_coef, by="Variable", suffix = c("_Pool","_FE")) %>%
  full_join(RE_coef, by="Variable") %>%
  rename(Estimate_RE = Estimate, StdError_RE = StdError)

coef_table
#по результатам тестов оптимальная модель - FE




# выбор из Pool, FE, RE (по объектам) с учетом фиксированных временных эффектов

# Pool-модель с фиксированными эффектами по годам
pool_time <- plm(ln_quantity ~ ln_price + ln_gdp_partner + ln_dist + iip_partner + factor(year),
                 data = pdata,
                 model = "pooling")

# FE по странам + фиксированные эффекты по годам
FE_time <- plm(ln_quantity ~ ln_price + ln_gdp_partner + ln_dist + iip_partner + factor(year),
               data = pdata,
               model = "within",
               effect = "individual")

# RE по странам + фиксированные эффекты по годам
RE_time <- plm(ln_quantity ~ ln_price + ln_gdp_partner + ln_dist + iip_partner + factor(year),
               data = pdata,
               model = "random",
               effect = "individual")

#аналогично F-тест и тест Хаусмана

f_test_time <- pFtest(FE_time, pool_time)
f_test_time
#FE лучше, чем Pool

h_test_time <- phtest(FE_time, RE_time)
h_test_time
#FE лучше, чем RE
#Итого FE

pool_coef <- get_coefs(pool_time)
FE_coef <- get_coefs(FE_time)
RE_coef <- get_coefs(RE_time)

# Объединяем в одну таблицу
coef_table <- full_join(pool_coef, FE_coef, by="Variable", suffix = c("_Pool","_FE")) %>%
  full_join(RE_coef, by="Variable") %>%
  rename(Estimate_RE = Estimate, StdError_RE = StdError)

coef_table

#На данном этапе адекватна модель с FE на страны и без учета годов 
#(потому что самый первый тест показал, что Pool-модель адекватна)

#нарушения предпосылок КЛРМ
#1. Гетероскедастичность
#Тест Бройша-Пагана для панельных данных

# FE модель по странам (без фиксированных эффектов по годам)
FE_model <- plm(ln_quantity ~ ln_price + ln_gdp_partner  + iip_partner,
                data = pdata,
                model = "within",
                effect = "individual")

# Тест на гетероскедастичность Бройша-Пагана для панельных данных
bptest(FE_model, studentize = TRUE) 

#Есть гетероскедастичность

#2. Автокорреляция временная (тест Бройша-Годфри/Вулдриджа)
pbgtest(FE_model)

#На уровнях значимости 0.05 и 0.1 автокорреляция присутствует


#2. Автокорреляция пространственная
#Тест Песарана

pcd <- pcdtest(FE_model, test = "cd")
pcd
# Пространственная автокорреляция есть


library(lmtest)
se_standard <- sqrt(diag(vcov(FE_model)))

# Гетероскедастичность + AR(1) по объектам
se_robust_group <- sqrt(diag(vcovHC(FE_model, type="HC3", cluster="group")))

# Пространственная корреляция (кластер по годам)
se_robust_time <- sqrt(diag(vcovHC(FE_model, type="HC3", cluster="time")))


library(dplyr)
library(xtable)

# Коэффициенты
coef_FE <- coef(FE_model)

# Таблица
coef_table <- data.frame(
  Variable = names(coef_FE),
  Estimate = round(coef_FE, 3),
  StdError = round(se_standard, 3),
  StdError_Robust_Group = round(se_robust_group, 3),
  StdError_Robust_Time = round(se_robust_time, 3)
)

coef_table

#2

#Предполагаем, что есть эндогенность цены по внешнему шоку - стоимость поставок,
#особенно выросшая в ходе энергетического кризиса 2022-2023 + СВО - это можно схватить 
#через цену на нефть
#С другой стороны, можно схватить шок 2022-2023 гг. через относительный валютный курс

#цены на нефть - https://ourworldindata.org/grapher/crude-oil-prices
oil_prices <- c(
  "2017" = 341,
  "2018" = 449,
  "2019" = 404,
  "2020" = 263,
  "2021" = 446,
  "2022" = 637,
  "2023" = 520
)

add_oil_price <- function(df, year_col = "year") {
  df$log_oil_price <- log(oil_prices[as.character(df[[year_col]])])
  return(df)
}

pdata <- add_oil_price(pdata)

# релевантность
first_stage1 <- plm(
  ln_price ~ log_oil_price + ln_gdp_partner + iip_partner,
  data = pdata,
  model = "within",
  effect = "individual"
)
summary(first_stage1)

# F-тест по инструментам (для релевантности)
coefs_fs <- summary(first_stage1)$coefficients
f_stat <- (coefs_fs["log_oil_price","Estimate"]/coefs_fs["log_oil_price","Std. Error"])^2
cat("F-statistic для инструмента log_oil_price:", f_stat, "\n")

#F сильно меньше 10 (0.26) - инструмент не релевантен

iv_model1 <- plm(
  ln_quantity ~ ln_price + ln_gdp_partner + iip_partner |
    ln_gdp_partner + iip_partner + ln_gdp_partner + log_oil_price,
  data = pdata,
  model = "within"
)

phtest(FE_model, iv_model1)
#Тест Хаусмана не выявляет эндогенности

summary(iv_model1, diagnostics = TRUE)

#Получается плохая модель (отрицательный модифицированный R^2, например) + изначально тест Хаусмана 
#имел p-value >> 0.05


#Пробуем с относительным курсом
pdata$ln_bilateral_er = as.numeric(as.character(data$ln_bilateral_er))
# релевантность
first_stage2 <- plm(
  ln_price ~ ln_bilateral_er + ln_gdp_partner + iip_partner,
  data = pdata,
  model = "within",
  effect = "individual"
)
summary(first_stage2)

# F-тест по инструментам (для релевантности)
coefs_fs <- summary(first_stage2)$coefficients
f_stat <- (coefs_fs["ln_bilateral_er","Estimate"]/coefs_fs["ln_bilateral_er","Std. Error"])^2
cat("F-statistic для инструмента ln_bilateral_er:", f_stat, "\n")

#F сильно меньше 10 (0.12) - инструмент не релевантен

iv_model2 <- plm(
  ln_quantity ~ ln_price + ln_gdp_partner + iip_partner |
    ln_gdp_partner + iip_partner + ln_bilateral_er,
  data = pdata,
  model = "within"
)

phtest(FE_model, iv_model2)
#Тест Хаусмана так же не свидетельствует в пользу эндогенности 
summary(iv_model2, diagnostics = TRUE)
#модель слабая


#Пробуем с "прямым" курсом (доллара к рублю)
pdata$ex_rate_russia = as.numeric(as.character(data$ex_rate_russia))
pdata$log_er_rur = log(pdata$ex_rate_russia)

# релевантность
first_stage3 <- plm(
  ln_price ~ log_er_rur + ln_gdp_partner + iip_partner,
  data = pdata,
  model = "within",
  effect = "individual"
)
summary(first_stage3)

# F-тест по инструментам (для релевантности)
coefs_fs <- summary(first_stage3)$coefficients
f_stat <- (coefs_fs["log_er_rur","Estimate"]/coefs_fs["log_er_rur","Std. Error"])^2
cat("F-statistic для инструмента log_er_rur:", f_stat, "\n")

#F меньше 10 (5.59) - инструмент не релевантен, но уже сильно лучше


#Пробуем с курсом
iv_model3 <- plm(
  ln_quantity ~ ln_price + ln_gdp_partner + iip_partner |
    ln_gdp_partner + iip_partner + log_er_rur,
  data = pdata,
  model = "within"
)

phtest(FE_model, iv_model3)
#Тест Хаусмана свидетельствует в пользу возможной эндогенности цены по внешним шокам и смещения оценок 
summary(iv_model3, diagnostics = TRUE)






# https://ru.tradingeconomics.com/commodity/containerized-freight-index

freight_index <- c(
  "2017" = 990,
  "2018" = 961,
  "2019" = 1023,
  "2020" = 2642,
  "2021" = 5047,
  "2022" = 5110,
  "2023" = 1061
)

# Добавляем переменную в pdata
pdata$freight_index <- freight_index[as.character(pdata$year)]

# Логарифмируем для модели
pdata$log_freight_index <- log(pdata$freight_index)



# релевантность
first_stage4 <- plm(
  ln_price ~ log_freight_index + ln_gdp_partner + iip_partner,
  data = pdata,
  model = "within",
  effect = "individual"
)
summary(first_stage4)

# F-тест по инструментам (для релевантности)
coefs_fs <- summary(first_stage4)$coefficients
f_stat <- (coefs_fs["log_freight_index","Estimate"]/coefs_fs["log_freight_index","Std. Error"])^2
cat("F-statistic для инструмента log_freight_index:", f_stat, "\n")

#F меньше 10 (6.5) - инструмент не релевантен, но не так безнадежно

iv_model4 <- plm(
  ln_quantity ~ ln_price + ln_gdp_partner + iip_partner |
    ln_gdp_partner + iip_partner + ln_gdp_partner + log_freight_index,
  data = pdata,
  model = "within"
)

phtest(FE_model, iv_model4)
#Тест Хаусмана показывает пограничную ситуацию - p-value = 0.06

summary(iv_model4, diagnostics = TRUE)



# Объединяем в одну таблицу
iv_model1_coef <- get_coefs(iv_model1)
iv_model2_coef <- get_coefs(iv_model2)
iv_model3_coef <- get_coefs(iv_model3)
iv_model4_coef <- get_coefs(iv_model4)

coef_table <- full_join(iv_model1_coef, iv_model2_coef, by="Variable", suffix = c("_IV1","_IV2")) %>%
  full_join(iv_model3_coef, by="Variable") %>%
  rename(Estimate_IV3 = Estimate, StdError_IV3 = StdError) %>%
  full_join(iv_model4_coef, by="Variable") %>%
  rename(Estimate_IV4 = Estimate, StdError_IV4 = StdError)

coef_table

#RE с ln_dist 
re_model <- plm(
  ln_quantity ~ ln_price + ln_gdp_partner + iip_partner + ln_dist,
  data = pdata,
  model = "random"
)

summary(re_model)
# ln_dist значима со знаком минус


#RE с ln_dist и contig (общая граница)
re_model <- plm(
  ln_quantity ~ ln_price + ln_gdp_partner + iip_partner + ln_dist + contig,
  data = pdata,
  model = "random"
)

summary(re_model)

# ln_dist теперь не значима (p-value 0.11) - общая граница обладает в данном случае
# намного большей объясняющей способностью

# оставим ln_dist как артефакт гравитационной модели

# FE + BE
fe_model <- plm(
  ln_quantity ~ ln_price + ln_gdp_partner + iip_partner + ln_dist + contig,
  data = pdata,
  model = "within"
)

summary(fe_model)


be_model <- plm(
  ln_quantity ~ ln_price + ln_gdp_partner + iip_partner + ln_dist + contig,
  data = pdata,
  model = "between"
)

summary(be_model)

# интересно, что здесь значим логарифм ВВП, а не индекс промышленного производства

phtest(fe_model, re_model)

# Тест Хаусмана показывает, что RE несостоятелен. Нужно брать Хаусмана-Тейлора (HT)

# ! HT
# Для HT делим переменные на 4 группы:
# X1 - экзогенная меняющаяся во времени
# X2 - эндогенная меняющаяся во времени
# Z1 - экзогенная неизменная во времени
# Z2 - эндогенная неизменная во времени

# В нашем случае:
# X1 - ln_gdp_partner, iip_partner
# X2 - ln_price
# Z1 - contig, ln_dist
# Z2 - -

#мы предполагаем contig экзогенной переменной, поскольку так часто делается + для идентификации HT.

ht_model <- plm(
  ln_quantity ~ ln_price + ln_gdp_partner + iip_partner + ln_dist + contig |
    ln_gdp_partner + iip_partner + contig + ln_dist, 
    data = pdata, 
    model = "ht")
summary(ht_model)


# Так как мы не предполагаем наличие ээндогеннфх неизменных во времени переменных,
# то нет необходимости проводить дополнительные тесты на инструменты

# Сводная таблица коэффициентов

re_coef = get_coefs(re_model)
be_coef = get_coefs(be_model)
ht_coef = get_coefs(ht_model)

coef_table <- full_join(re_coef, be_coef, by="Variable", suffix = c("_RE","_BE")) %>%
  full_join(ht_coef, by="Variable") %>%
  rename(Estimate_HT = Estimate, StdError_HT = StdError) 

coef_table

# Лаги
pdata$ln_quantity_lag1 <- lag(pdata$ln_quantity, 1)
pdata$ln_quantity_lag2 <- lag(pdata$ln_quantity, 2)

dyn_FE1 <- plm(
  ln_quantity ~ ln_quantity_lag1 + ln_price + ln_gdp_partner + iip_partner + ln_dist + contig,
  data = pdata,
  model = "within",
  effect = "individual"
)

summary(dyn_FE1)

pbgtest(dyn_FE1)
# p-value = 0.051. На грани значимости нет автокорреляции

coeftest(dyn_FE1, vcov=vcovHC(dyn_FE1, method="arellano", type="HC3", cluster="group"))
# лаг незначим

dyn_FE2 <- plm(
  ln_quantity ~ ln_quantity_lag1 + ln_quantity_lag2 + ln_price + ln_gdp_partner + iip_partner + ln_dist + contig,
  data = pdata,
  model = "within",
  effect = "individual"
)

summary(dyn_FE2)

pbgtest(dyn_FE2)
# автокорреляции нет

coeftest(dyn_FE2, vcov=vcovHC(dyn_FE2, method="arellano", type="HC3", cluster="group"))
# лаги значимы

# Гетероскедастичность
bptest(dyn_FE2)
# Есть гетероскедастичность

coeftest(dyn_FE2, vcov = vcovHC(dyn_FE2, type="HC3"))
# значимость не меняется

# Тест Песарана
pcdtest(dyn_FE2, test = "cd")

# Есть зависимость между странами

dyn_pool <- plm(
  ln_quantity ~ ln_quantity_lag1 + ln_quantity_lag2 + ln_price + ln_gdp_partner + iip_partner + ln_dist + contig,
  data = pdata,
  model = "pool"
)

summary(dyn_pool)


pFtest(dyn_FE2, dyn_pool)
# Индивидуальные эффекты статистически значимы

dyn_RE2 <- plm(
  ln_quantity ~ ln_quantity_lag1 + ln_quantity_lag2 + ln_price + ln_gdp_partner + iip_partner + ln_dist + contig,
  data = pdata,
  model = "random",
  effect = "individual"
)

summary(dyn_RE2)

phtest(dyn_FE2, dyn_RE2)
# Эффекты коррелируют с объясняющими переменными, поэтому нужна модель FE


