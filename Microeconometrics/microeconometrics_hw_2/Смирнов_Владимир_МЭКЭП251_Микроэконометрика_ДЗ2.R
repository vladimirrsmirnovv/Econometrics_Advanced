setwd("C:/Users/smirn/OneDrive/Рабочий стол/Microeconometrics")

# Загрузка данных
# data <- readRDS("Mroz87.xslx")

# Часть 1. Теория и гипотезы 
# 1.1. Выберите независимые переменные для уравнения зарплаты и уравнения 
# занятости. Кратко теоретически обоснуйте выбор каждой из них: не обязательно со 
# ссылками на литературу, достаточно здравого смысла. Укажите и кратко обоснуйте 
# предполагаемые направления эффектов. Уравнение занятости должно включать по крайней 
# мере одну переменную, которой не было в уравнении зарплаты, и одну переменную, 
# которая есть в уравнении зарплаты. Желательно, чтобы общая для двух уравнений 
# переменная была непрерывной, например, возраст, а также не использовать более трех 
# переменных в каждом из уравнений. 

#---------------------------------------------------
# Загрузка данных Mroz87
#---------------------------------------------------

#---------------------------------------------------
# Загрузка данных Mroz87
#---------------------------------------------------

options(scipen = 999)

setwd("C:/Users/smirn/OneDrive/Рабочий стол/Microeconometrics")

#---------------------------------------------------
# Вариант 1. Загрузка из Excel-файла
#---------------------------------------------------

library(readxl)

data <- read_excel("Mroz87.xlsx")
data <- as.data.frame(data)

head(data)
str(data)
summary(data)
dim(data)
names(data)
colSums(is.na(data))

#---------------------------------------------------
# Описание встроенного датасета Mroz87
#---------------------------------------------------

install.packages("sampleSelection")
library(sampleSelection)

# Загрузим встроенный датасет
data("Mroz87", package = "sampleSelection")

# Открыть встроенную справку R / RStudio
help("Mroz87", package = "sampleSelection")

# То же самое, коротко:
?Mroz87

# Названия переменных
names(Mroz87)

# Структура данных
str(Mroz87)

#---------------------------------------------------
# Таблица с расшифровкой переменных
#---------------------------------------------------

var_desc <- data.frame(
  variable = c(
    "lfp",
    "hours",
    "kids5",
    "kids618",
    "age",
    "educ",
    "wage",
    "repwage",
    "hushrs",
    "husage",
    "huseduc",
    "huswage",
    "faminc",
    "mtr",
    "motheduc",
    "fatheduc",
    "unem",
    "city",
    "exper",
    "nwifeinc",
    "wifecoll",
    "huscoll"
  ),
  description = c(
    "участие женщины в рабочей силе: 1 — работает / участвует, 0 — не работает",
    "количество отработанных часов",
    "число детей младше 5 лет",
    "число детей в возрасте от 6 до 18 лет",
    "возраст женщины",
    "число лет образования женщины",
    "часовая заработная плата женщины",
    "зарплата женщины, сообщенная в интервью",
    "количество отработанных часов мужа",
    "возраст мужа",
    "число лет образования мужа",
    "часовая заработная плата мужа",
    "семейный доход",
    "предельная налоговая ставка",
    "число лет образования матери женщины",
    "число лет образования отца женщины",
    "уровень безработицы в регионе проживания",
    "проживание в городе: 1 — город, 0 — не город",
    "опыт работы женщины",
    "семейный доход за вычетом дохода женщины",
    "наличие колледжа у женщины: 1 — да, 0 — нет",
    "наличие колледжа у мужа: 1 — да, 0 — нет"
  )
)

var_desc

# Удобно открыть таблицей в RStudio
View(var_desc)


# Уравнение зарплаты
wage_formula <- wage ~ educ + exper + age

# Уравнение занятости / отбора
work_formula <- lfp ~ age + kids5 + nwifeinc


#---------------------------------------------------
# Часть 2. Модель Тобина
# 2.1. Оценивание Тобит-модели
#---------------------------------------------------

options(scipen = 999)

# Пакеты
# install.packages("crch")
# install.packages("lmtest")

library(crch)
library(lmtest)

# Точка левого цензурирования
tr_left <- 0

# Оцениваем Тобит-модель
# wage = 0 для неработающих, wage > 0 для работающих
model_tobit <- crch(
  wage ~ educ + exper + age,
  data = data,
  left = tr_left,
  truncated = FALSE,
  dist = "gaussian"
)

# Результат модели
summary(model_tobit)

# Логарифм правдоподобия, AIC, BIC
logLik(model_tobit)
AIC(model_tobit)
BIC(model_tobit)

#---------------------------------------------------
# Сохраним коэффициенты и sigma
#---------------------------------------------------

est_tobit <- coef(model_tobit)

coef_tobit <- est_tobit[-length(est_tobit)]
sigma_tobit <- exp(est_tobit[length(est_tobit)])

coef_tobit
sigma_tobit

#---------------------------------------------------
# Таблица результатов
#---------------------------------------------------

model_summary <- summary(model_tobit)

# В crch таблица коэффициентов обычно лежит здесь
tobit_table <- as.data.frame(model_summary$coefficients)

tobit_table

# Удобный вариант таблицы для отчета
tobit_table_out <- data.frame(
  variable = rownames(tobit_table),
  estimate = round(tobit_table[, 1], 4),
  std_error = round(tobit_table[, 2], 4),
  z_value = round(tobit_table[, 3], 4),
  p_value = round(tobit_table[, 4], 4)
)

rownames(tobit_table_out) <- NULL
tobit_table_out

#---------------------------------------------------
# Ручная функция логарифма правдоподобия Тобит-модели
#---------------------------------------------------

lnL_tobit <- function(par, y, X, tr_left = 0)
{
  par_n <- length(par)
  
  beta <- matrix(par[-par_n], ncol = 1)
  sigma <- par[par_n]
  
  if (sigma <= 0) {
    return(-1e+100)
  }
  
  X <- cbind(1, X)
  XB <- X %*% beta
  
  L <- rep(NA, length(y))
  
  censored <- y == tr_left
  uncensored <- y > tr_left
  
  L[censored] <- pnorm(tr_left - XB[censored, ], sd = sigma)
  
  L[uncensored] <- dnorm(
    y[uncensored] - XB[uncensored, ],
    sd = sigma
  )
  
  lnL_value <- sum(log(L))
  
  return(lnL_value)
}

# Матрица регрессоров без константы
X_tobit <- as.matrix(data[, c("educ", "exper", "age")])
y_tobit <- data$wage

# Начальные значения можно взять из обычного МНК по положительным wage
model_lm_start <- lm(wage ~ educ + exper + age, data = data[data$wage > 0, ])

x0 <- c(
  coef(model_lm_start),
  sigma(model_lm_start)
)

# Оптимизация
opt_tobit <- optim(
  par = x0,
  fn = lnL_tobit,
  y = y_tobit,
  X = X_tobit,
  tr_left = 0,
  method = "Nelder-Mead",
  control = list(
    fnscale = -1,
    maxit = 100000,
    reltol = 1e-10
  )
)

opt_tobit$par
opt_tobit$value

#---------------------------------------------------
# 2.4. Расчет E(y*), E(y), P(y > 0)
#---------------------------------------------------

Boris <- data.frame(
  educ = 12,
  exper = 10,
  age = 40
)

# Оценки из Тобит-модели
beta_hat <- coef_tobit
sigma_hat <- as.numeric(sigma_tobit)

# A) E(y*) = x beta
mu_hat <- as.numeric(
  beta_hat["(Intercept)"] +
    beta_hat["educ"] * Boris$educ +
    beta_hat["exper"] * Boris$exper +
    beta_hat["age"] * Boris$age
)

# Стандартизированный индекс
z_hat <- mu_hat / sigma_hat

# Вероятность положительной зарплаты
prob_work <- pnorm(z_hat)

# B) E(y) для Тобит-модели с левым цензурированием в 0
Ey <- prob_work * mu_hat + sigma_hat * dnorm(z_hat)

# Соберем результаты
tobit_individual_results <- data.frame(
  indicator = c(
    "E(wage*)",
    "E(wage)",
    "P(wage > 0)"
  ),
  value = round(c(
    mu_hat,
    Ey,
    prob_work
  ), 4)
)

tobit_individual_results

#---------------------------------------------------
# 2.5. Предельные эффекты exper в Тобит-модели
#---------------------------------------------------

# Характеристики индивида
Boris <- data.frame(
  educ = 12,
  exper = 10,
  age = 40
)

# Оценки модели
beta_hat <- coef_tobit
sigma_hat <- as.numeric(sigma_tobit)

# Линейный индекс
mu_hat <- as.numeric(
  beta_hat["(Intercept)"] +
    beta_hat["educ"] * Boris$educ +
    beta_hat["exper"] * Boris$exper +
    beta_hat["age"] * Boris$age
)

# Стандартизированный индекс
z_hat <- mu_hat / sigma_hat

# Phi и phi
Phi_hat <- pnorm(z_hat)
phi_hat <- dnorm(z_hat)

# Коэффициент при exper
beta_exper <- as.numeric(beta_hat["exper"])

# Предельные эффекты
ME_Eystar <- beta_exper
ME_Ey <- Phi_hat * beta_exper
ME_prob <- phi_hat * beta_exper / sigma_hat

tobit_me_results <- data.frame(
  indicator = c(
    "dE(wage*)/d exper",
    "dE(wage)/d exper",
    "dP(wage > 0)/d exper"
  ),
  value = round(c(
    ME_Eystar,
    ME_Ey,
    ME_prob
  ), 4)
)

tobit_me_results

#---------------------------------------------------
# 2.6. Тобит-модель с нелинейным эффектом exper
#---------------------------------------------------

options(scipen = 999)

library(crch)

tr_left <- 0

# Добавляем квадрат опыта
model_tobit_nl <- crch(
  wage ~ educ + exper + I(exper^2) + age,
  data = data,
  left = tr_left,
  truncated = FALSE,
  dist = "gaussian"
)

summary(model_tobit_nl)

logLik(model_tobit_nl)
AIC(model_tobit_nl)
BIC(model_tobit_nl)

#---------------------------------------------------
# Сохраняем оценки коэффициентов и sigma
#---------------------------------------------------

est_tobit_nl <- coef(model_tobit_nl)

coef_tobit_nl <- est_tobit_nl[-length(est_tobit_nl)]
sigma_tobit_nl <- exp(est_tobit_nl[length(est_tobit_nl)])

coef_tobit_nl
sigma_tobit_nl

#---------------------------------------------------
# Таблица коэффициентов для отчета
#---------------------------------------------------

model_summary_nl <- summary(model_tobit_nl)

tobit_nl_table <- as.data.frame(model_summary_nl$coefficients)

tobit_nl_location_table <- data.frame(
  variable = rownames(tobit_nl_table),
  estimate = round(tobit_nl_table[, 1], 4),
  std_error = round(tobit_nl_table[, 2], 4),
  z_value = round(tobit_nl_table[, 3], 4),
  p_value = round(tobit_nl_table[, 4], 4)
)

rownames(tobit_nl_location_table) <- NULL
tobit_nl_location_table

tobit_nl_scale_table <- data.frame(
  variable = "log(sigma)",
  estimate = round(tobit_nl_table[1, 5], 4),
  std_error = round(tobit_nl_table[1, 6], 4),
  z_value = round(tobit_nl_table[1, 7], 4),
  p_value = round(tobit_nl_table[1, 8], 4)
)

tobit_nl_scale_table

#---------------------------------------------------
# 2.7. LR-тест на гомоскедастичность в Тобит-модели
#---------------------------------------------------

options(scipen = 999)

library(crch)

tr_left <- 0

#---------------------------------------------------
# 1. Гомоскедастичная Тобит-модель
# H0: sigma_i = sigma
#---------------------------------------------------

model_tobit_homo <- crch(
  wage ~ educ + exper + age,
  data = data,
  left = tr_left,
  truncated = FALSE,
  dist = "gaussian"
)

summary(model_tobit_homo)
logLik(model_tobit_homo)


#---------------------------------------------------
# 2. Гетероскедастичная Тобит-модель
# log(sigma_i) = delta_0 + delta_1 kids5_i + delta_2 nwifeinc_i
#---------------------------------------------------

model_tobit_het <- crch(
  wage ~ educ + exper + age | kids5 + nwifeinc,
  data = data,
  left = tr_left,
  truncated = FALSE,
  dist = "gaussian"
)

summary(model_tobit_het)
logLik(model_tobit_het)


#---------------------------------------------------
# 3. LR-тест
# H0: delta_1 = delta_2 = 0
# H1: хотя бы один коэффициент в уравнении дисперсии не равен 0
#---------------------------------------------------

ll_homo <- as.numeric(logLik(model_tobit_homo))
ll_het  <- as.numeric(logLik(model_tobit_het))

LR_stat <- 2 * (ll_het - ll_homo)

df_lr <- attr(logLik(model_tobit_het), "df") -
  attr(logLik(model_tobit_homo), "df")

p_value_lr <- pchisq(LR_stat, df = df_lr, lower.tail = FALSE)

lr_test_tobit <- data.frame(
  test = "LR test for homoskedasticity",
  logLik_homo = round(ll_homo, 4),
  logLik_het = round(ll_het, 4),
  LR_statistic = round(LR_stat, 4),
  df = df_lr,
  p_value = signif(p_value_lr, 4)
)

lr_test_tobit





#---------------------------------------------------
# 2.8. Гетероскедастичная Тобит-модель
#      переменная age входит и в основное уравнение,
#      и в уравнение дисперсии
#---------------------------------------------------

options(scipen = 999)

library(crch)

tr_left <- 0

model_tobit_het_common <- crch(
  wage ~ educ + exper + age | age + kids5 + nwifeinc,
  data = data,
  left = tr_left,
  truncated = FALSE,
  dist = "gaussian"
)

summary(model_tobit_het_common)
logLik(model_tobit_het_common)
AIC(model_tobit_het_common)
BIC(model_tobit_het_common)

#---------------------------------------------------
# Извлекаем коэффициенты
#---------------------------------------------------

beta_hat_het <- coef(model_tobit_het_common, model = "location")
gamma_hat_het <- coef(model_tobit_het_common, model = "scale")

beta_hat_het
gamma_hat_het

#---------------------------------------------------
# Индивид с произвольными характеристиками
#---------------------------------------------------

Boris_het <- data.frame(
  educ = 12,
  exper = 10,
  age = 40,
  kids5 = 1,
  nwifeinc = 20
)

#---------------------------------------------------
# Линейный индекс основного уравнения:
# mu_i = x_i beta
#---------------------------------------------------

mu_hat_het <- as.numeric(
  beta_hat_het["(Intercept)"] +
    beta_hat_het["educ"] * Boris_het$educ +
    beta_hat_het["exper"] * Boris_het$exper +
    beta_hat_het["age"] * Boris_het$age
)

#---------------------------------------------------
# Уравнение масштаба:
# log(sigma_i) = z_i gamma
# sigma_i = exp(z_i gamma)
#---------------------------------------------------

log_sigma_hat_het <- as.numeric(
  gamma_hat_het["(Intercept)"] +
    gamma_hat_het["age"] * Boris_het$age +
    gamma_hat_het["kids5"] * Boris_het$kids5 +
    gamma_hat_het["nwifeinc"] * Boris_het$nwifeinc
)

sigma_hat_het <- exp(log_sigma_hat_het)

#---------------------------------------------------
# Стандартизированный индекс
#---------------------------------------------------

a_hat_het <- mu_hat_het / sigma_hat_het

Phi_hat_het <- pnorm(a_hat_het)
phi_hat_het <- dnorm(a_hat_het)

#---------------------------------------------------
# Переменная, входящая в оба уравнения: age
#---------------------------------------------------

beta_age <- as.numeric(beta_hat_het["age"])
gamma_age <- as.numeric(gamma_hat_het["age"])

# Производные:
# d mu / d age = beta_age
# d sigma / d age = sigma_i * gamma_age

dmu_dage <- beta_age
dsigma_dage <- sigma_hat_het * gamma_age

#---------------------------------------------------
# Предельные эффекты
#---------------------------------------------------

# A) dE(y*) / d age
ME_Eystar_age <- dmu_dage

# Б) dE(y) / d age
# E(y|x,z) = Phi(a) * mu + sigma * phi(a)
# dE(y)/d age = Phi(a) * dmu/dage + phi(a) * dsigma/dage
ME_Ey_age <- Phi_hat_het * dmu_dage + phi_hat_het * dsigma_dage

# В) dP(y > 0) / d age
# P(y > 0) = Phi(mu / sigma)
# dP/dage = phi(a) * (beta_age / sigma_i - a * gamma_age)
ME_prob_age <- phi_hat_het * (beta_age / sigma_hat_het - a_hat_het * gamma_age)

#---------------------------------------------------
# Итоговая таблица
#---------------------------------------------------

tobit_het_me_results <- data.frame(
  indicator = c(
    "dE(wage*)/d age",
    "dE(wage)/d age",
    "dP(wage > 0)/d age"
  ),
  value = round(c(
    ME_Eystar_age,
    ME_Ey_age,
    ME_prob_age
  ), 4)
)

tobit_het_me_results

#---------------------------------------------------
# Дополнительные расчетные величины
#---------------------------------------------------

tobit_het_calc_values <- data.frame(
  indicator = c(
    "mu_hat",
    "log_sigma_hat",
    "sigma_hat",
    "mu_hat / sigma_hat",
    "Phi(mu_hat / sigma_hat)",
    "phi(mu_hat / sigma_hat)",
    "dmu / d age",
    "dsigma / d age"
  ),
  value = round(c(
    mu_hat_het,
    log_sigma_hat_het,
    sigma_hat_het,
    a_hat_het,
    Phi_hat_het,
    phi_hat_het,
    dmu_dage,
    dsigma_dage
  ), 4)
)

tobit_het_calc_values



#---------------------------------------------------
# Часть 3. Модель Хекмана
# 3.1. Оценивание модели Хекмана методом ММП
#---------------------------------------------------

options(scipen = 999)

# Пакеты
# install.packages("switchSelection")
library(switchSelection)

#---------------------------------------------------
# Подготовка зависимой переменной для уравнения зарплаты
#---------------------------------------------------

# В модели Хекмана зарплата наблюдается только для работающих женщин.
# Поэтому для lfp = 0 ставим NA, а не 0.
data$wage_heckman <- ifelse(data$lfp == 1, data$wage, NA)

# Проверка
table(data$lfp, is.na(data$wage_heckman))

#---------------------------------------------------
# Спецификация модели
#---------------------------------------------------

# Уравнение занятости / отбора:
# lfp = 1, если женщина работает / участвует в рабочей силе
selection_formula <- lfp ~ age + kids5 + nwifeinc

# Уравнение зарплаты:
# wage_heckman наблюдается только для работающих
wage_formula_heckman <- wage_heckman ~ educ + exper + age

#---------------------------------------------------
# Оценивание модели Хекмана методом максимального правдоподобия
#---------------------------------------------------

model_heckman_mle <- msel(
  formula   = selection_formula,
  formula2  = wage_formula_heckman,
  data      = data,
  estimator = "ml"
)

summary(model_heckman_mle)

#---------------------------------------------------
# Извлечение коэффициентов и параметров модели
#---------------------------------------------------

# Коэффициенты уравнения отбора
coef_selection_mle <- coef(
  model_heckman_mle,
  type = "coef",
  eq = 1
)

# Коэффициенты уравнения зарплаты
coef_wage_mle <- coef(
  model_heckman_mle,
  type = "coef2",
  eq2 = 1
)

# Стандартное отклонение ошибки уравнения зарплаты
sigma_mle <- sigma(model_heckman_mle)

# Ковариация между ошибками уравнения отбора и уравнения зарплаты
cov_mle <- coef(
  model_heckman_mle,
  type = "cov12",
  eq = 1,
  regime = 0
)

# Корреляция ошибок
rho_mle <- cov_mle / sigma_mle

coef_selection_mle
coef_wage_mle
sigma_mle
cov_mle
rho_mle

# Log-likelihood, AIC, BIC
logLik(model_heckman_mle)
AIC(model_heckman_mle)
BIC(model_heckman_mle)

#---------------------------------------------------
# Количество наблюдений
#---------------------------------------------------

n_total <- nrow(data)
n_work <- sum(data$lfp == 1)
n_not_work <- sum(data$lfp == 0)

data.frame(
  indicator = c("Всего наблюдений", "Работающие", "Неработающие"),
  value = c(n_total, n_work, n_not_work)
)



#---------------------------------------------------
# 3.3. Двухшаговый метод Хекмана и сравнение с MLE
#---------------------------------------------------

options(scipen = 999)

library(switchSelection)

#---------------------------------------------------
# 1. Подготовка зависимой переменной
#---------------------------------------------------

# В модели Хекмана зарплата наблюдается только для работающих
data$wage_heckman <- ifelse(data$lfp == 1, data$wage, NA)

table(data$lfp, is.na(data$wage_heckman))

#---------------------------------------------------
# 2. Спецификации
#---------------------------------------------------

selection_formula <- lfp ~ age + kids5 + nwifeinc

wage_formula_heckman <- wage_heckman ~ educ + exper + age

#---------------------------------------------------
# 3. Модель Хекмана методом максимального правдоподобия
#    Если объект model_heckman_mle уже создан, этот блок можно не запускать повторно
#---------------------------------------------------

model_heckman_mle <- msel(
  formula   = selection_formula,
  formula2  = wage_formula_heckman,
  data      = data,
  estimator = "ml"
)

summary(model_heckman_mle)

#---------------------------------------------------
# 4. Модель Хекмана двухшаговым методом
#---------------------------------------------------

model_heckman_2step <- msel(
  formula   = selection_formula,
  formula2  = wage_formula_heckman,
  data      = data,
  estimator = "2step"
)

summary(model_heckman_2step)

#---------------------------------------------------
# 5. Удобная функция для извлечения коэффициентов
#---------------------------------------------------

make_named_vector <- function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    v <- as.numeric(x[1, ])
    names(v) <- colnames(x)
  } else {
    v <- as.numeric(x)
    names(v) <- names(x)
  }
  return(v)
}

#---------------------------------------------------
# 6. Коэффициенты уравнения зарплаты
#---------------------------------------------------

coef_wage_mle <- coef(
  model_heckman_mle,
  type = "coef2",
  eq2 = 1
)

coef_wage_2step <- coef(
  model_heckman_2step,
  type = "coef2",
  eq2 = 1
)

coef_wage_mle_vec <- make_named_vector(coef_wage_mle)
coef_wage_2step_vec <- make_named_vector(coef_wage_2step)

comparison_wage <- data.frame(
  variable = names(coef_wage_mle_vec),
  Heckman_MLE = round(coef_wage_mle_vec, 4),
  Heckman_2step = round(coef_wage_2step_vec[names(coef_wage_mle_vec)], 4),
  difference_2step_minus_MLE = round(
    coef_wage_2step_vec[names(coef_wage_mle_vec)] - coef_wage_mle_vec,
    4
  )
)

comparison_wage

#---------------------------------------------------
# 7. Коэффициенты уравнения занятости / отбора
#---------------------------------------------------

coef_selection_mle <- coef(
  model_heckman_mle,
  type = "coef",
  eq = 1
)

coef_selection_2step <- coef(
  model_heckman_2step,
  type = "coef",
  eq = 1
)

coef_selection_mle_vec <- make_named_vector(coef_selection_mle)
coef_selection_2step_vec <- make_named_vector(coef_selection_2step)

comparison_selection <- data.frame(
  variable = names(coef_selection_mle_vec),
  Heckman_MLE = round(coef_selection_mle_vec, 4),
  Heckman_2step = round(coef_selection_2step_vec[names(coef_selection_mle_vec)], 4),
  difference_2step_minus_MLE = round(
    coef_selection_2step_vec[names(coef_selection_mle_vec)] - coef_selection_mle_vec,
    4
  )
)

comparison_selection

#---------------------------------------------------
# 8. Ручная двухшаговая процедура Хекмана
#---------------------------------------------------

# Первый шаг: пробит-модель занятости
model_selection_probit <- glm(
  lfp ~ age + kids5 + nwifeinc,
  data = data,
  family = binomial(link = "probit")
)

summary(model_selection_probit)

# Линейный индекс первого шага
data$selection_index <- predict(model_selection_probit, type = "link")

# Обратное отношение Миллса для работающих:
# lambda_i = phi(z_i gamma) / Phi(z_i gamma)
data$lambda_heckman <- NA
data$lambda_heckman[data$lfp == 1] <- 
  dnorm(data$selection_index[data$lfp == 1]) /
  pnorm(data$selection_index[data$lfp == 1])

# Второй шаг: уравнение зарплаты по работающим с lambda
model_heckman_2step_manual <- lm(
  wage_heckman ~ educ + exper + age + lambda_heckman,
  data = data
)

summary(model_heckman_2step_manual)

# Коэффициент при lambda_heckman оценивает rho * sigma
lambda_coef_manual <- coef(model_heckman_2step_manual)["lambda_heckman"]
lambda_coef_manual
  

#---------------------------------------------------
# 3.5. Условные ожидания и предельные эффекты
#      в MLE-модели Хекмана
#---------------------------------------------------

# Индивид с произвольными характеристиками
Boris_heckman <- data.frame(
  educ = 12,
  exper = 10,
  age = 40,
  kids5 = 1,
  nwifeinc = 20
)

# Оценки из MLE-модели
beta_hat <- coef(model_heckman_mle, type = "coef2", eq2 = 1)
beta_hat <- as.numeric(beta_hat[1, ])
names(beta_hat) <- c("(Intercept)", "educ", "exper", "age")

gamma_hat <- coef(model_heckman_mle, type = "coef", eq = 1)

sigma_hat <- as.numeric(sigma(model_heckman_mle))

rho_sigma_hat <- as.numeric(
  coef(model_heckman_mle, type = "cov12", eq = 1, regime = 0)
)

# cut1 из summary: -0.82263
cut1_hat <- -0.82263

# Линейный индекс уравнения зарплаты
mu_hat <- as.numeric(
  beta_hat["(Intercept)"] +
    beta_hat["educ"] * Boris_heckman$educ +
    beta_hat["exper"] * Boris_heckman$exper +
    beta_hat["age"] * Boris_heckman$age
)

# Индекс уравнения занятости
eta_hat <- as.numeric(
  gamma_hat["age"] * Boris_heckman$age +
    gamma_hat["kids5"] * Boris_heckman$kids5 +
    gamma_hat["nwifeinc"] * Boris_heckman$nwifeinc
)

# В параметризации msel:
# P(lfp = 1) = Phi(eta - cut1)
a_hat <- eta_hat - cut1_hat

Phi_hat <- pnorm(a_hat)
phi_hat <- dnorm(a_hat)

lambda_1 <- phi_hat / Phi_hat
lambda_0 <- phi_hat / (1 - Phi_hat)

# A) Условные ожидания
Ey_z1 <- mu_hat + rho_sigma_hat * lambda_1
Ey_z0 <- mu_hat - rho_sigma_hat * lambda_0

# Б) Предельные эффекты age
beta_age <- beta_hat["age"]
gamma_age <- gamma_hat["age"]

lambda_1_deriv <- -lambda_1 * (a_hat + lambda_1)
lambda_0_deriv <- lambda_0 * (lambda_0 - a_hat)

ME_age_z1 <- beta_age + rho_sigma_hat * lambda_1_deriv * gamma_age
ME_age_z0 <- beta_age - rho_sigma_hat * lambda_0_deriv * gamma_age

heckman_35_results <- data.frame(
  indicator = c(
    "mu_hat",
    "a_hat",
    "Phi(a_hat)",
    "phi(a_hat)",
    "lambda_1",
    "lambda_0",
    "E(y* | z = 1)",
    "E(y* | z = 0)",
    "dE(y* | z = 1) / d age",
    "dE(y* | z = 0) / d age"
  ),
  value = round(c(
    mu_hat,
    a_hat,
    Phi_hat,
    phi_hat,
    lambda_1,
    lambda_0,
    Ey_z1,
    Ey_z0,
    ME_age_z1,
    ME_age_z0
  ), 4)
)

heckman_35_results




#---------------------------------------------------
# Часть 4. Модель Ньюи
# 4.2. Оценивание модели Ньюи с LOOCV и bootstrap SE
#---------------------------------------------------

options(scipen = 999)

library(switchSelection)

#---------------------------------------------------
# 0. Подготовка данных
#---------------------------------------------------

data$wage_heckman <- ifelse(data$lfp == 1, data$wage, NA)

# Спецификации, как в модели Хекмана
selection_formula <- lfp ~ age + kids5 + nwifeinc
wage_formula_heckman <- wage_heckman ~ educ + exper + age

#---------------------------------------------------
# 1. Функция для расчета LOOCV RMSE для lm
#    Используем аналитическую формулу:
#    e_i^LOO = e_i / (1 - h_i)
#---------------------------------------------------

loocv_rmse_lm <- function(model) {
  e <- residuals(model)
  h <- hatvalues(model)
  sqrt(mean((e / (1 - h))^2, na.rm = TRUE))
}

#---------------------------------------------------
# 2. Функция оценки Newey-модели для заданной выборки
#---------------------------------------------------

estimate_newey <- function(df, degrees = 1:4) {
  
  # 1-й шаг: пробит-модель занятости
  model_sel <- glm(
    lfp ~ age + kids5 + nwifeinc,
    data = df,
    family = binomial(link = "probit")
  )
  
  # Линейный индекс отбора
  df$newey_index <- predict(model_sel, type = "link")
  
  # Сглаживающая функция:
  # g(w gamma) = lambda(w gamma) = phi(w gamma) / Phi(w gamma)
  # Чуть страхуемся от деления на очень маленькие вероятности
  Phi_index <- pmax(pnorm(df$newey_index), 1e-8)
  df$newey_lambda <- dnorm(df$newey_index) / Phi_index
  
  # Берем только тех, у кого wage наблюдается
  df_work <- df[df$lfp == 1, ]
  
  # Оцениваем модели с разными степенями полинома
  cv_results <- data.frame(
    degree = degrees,
    loocv_rmse = NA_real_
  )
  
  models <- vector("list", length(degrees))
  names(models) <- paste0("degree_", degrees)
  
  for (j in seq_along(degrees)) {
    k <- degrees[j]
    
    # Строим формулу:
    # wage_heckman ~ educ + exper + age + lambda + lambda^2 + ...
    poly_terms <- paste0("I(newey_lambda^", 1:k, ")", collapse = " + ")
    
    newey_formula <- as.formula(
      paste(
        "wage_heckman ~ educ + exper + age +",
        poly_terms
      )
    )
    
    model_k <- lm(newey_formula, data = df_work)
    
    models[[j]] <- model_k
    cv_results$loocv_rmse[j] <- loocv_rmse_lm(model_k)
  }
  
  # Выбираем степень с минимальным LOOCV RMSE
  best_id <- which.min(cv_results$loocv_rmse)
  best_degree <- cv_results$degree[best_id]
  best_model <- models[[best_id]]
  
  return(list(
    selection_model = model_sel,
    cv_results = cv_results,
    best_degree = best_degree,
    best_model = best_model,
    df_with_lambda = df
  ))
}

#---------------------------------------------------
# 3. Оценка Newey-модели на исходной выборке
#---------------------------------------------------

newey_result <- estimate_newey(data, degrees = 1:4)

newey_result$cv_results
newey_result$best_degree
summary(newey_result$best_model)

# Коэффициенты Newey-модели
coef_newey <- coef(newey_result$best_model)
coef_newey

#---------------------------------------------------
# 4. Bootstrap стандартные ошибки для Newey-модели
#---------------------------------------------------

set.seed(123)

B <- 300
degrees <- 1:4

# Основные коэффициенты, которые будем сравнивать с Хекманом
main_coef_names <- c("(Intercept)", "educ", "exper", "age")

boot_coef <- matrix(
  NA_real_,
  nrow = B,
  ncol = length(main_coef_names)
)

colnames(boot_coef) <- main_coef_names

boot_best_degree <- rep(NA_integer_, B)

for (b in 1:B) {
  
  if (b %% 25 == 0) {
    cat("Bootstrap iteration:", b, "\n")
  }
  
  # Bootstrap-выборка
  boot_id <- sample(seq_len(nrow(data)), size = nrow(data), replace = TRUE)
  data_b <- data[boot_id, ]
  
  # Оценка Newey-модели внутри bootstrap
  res_b <- tryCatch(
    estimate_newey(data_b, degrees = degrees),
    error = function(e) NULL
  )
  
  if (!is.null(res_b)) {
    boot_best_degree[b] <- res_b$best_degree
    
    cb <- coef(res_b$best_model)
    
    # Сохраняем только основные коэффициенты уравнения зарплаты
    boot_coef[b, ] <- cb[main_coef_names]
  }
}

# Bootstrap SE
boot_se_newey <- apply(boot_coef, 2, sd, na.rm = TRUE)

# Частота выбора степеней полинома
degree_freq <- table(boot_best_degree, useNA = "ifany")
degree_freq

#---------------------------------------------------
# 5. Таблица Newey: оценки и bootstrap SE
#---------------------------------------------------

newey_main_est <- coef_newey[main_coef_names]

newey_table <- data.frame(
  variable = main_coef_names,
  estimate = round(as.numeric(newey_main_est), 4),
  bootstrap_se = round(as.numeric(boot_se_newey[main_coef_names]), 4)
)

newey_table

#---------------------------------------------------
# 6. Сравнение Newey, Heckman MLE и Heckman 2-step
#---------------------------------------------------

# Если модели уже оценены ранее, этот блок можно не запускать повторно.
# Оставляю для воспроизводимости.

model_heckman_mle <- msel(
  formula   = selection_formula,
  formula2  = wage_formula_heckman,
  data      = data,
  estimator = "ml"
)

model_heckman_2step <- msel(
  formula   = selection_formula,
  formula2  = wage_formula_heckman,
  data      = data,
  estimator = "2step"
)

# Удобная функция
make_named_vector <- function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    v <- as.numeric(x[1, ])
    names(v) <- colnames(x)
  } else {
    v <- as.numeric(x)
    names(v) <- names(x)
  }
  return(v)
}

# Коэффициенты уравнения зарплаты MLE
coef_wage_mle <- coef(
  model_heckman_mle,
  type = "coef2",
  eq2 = 1
)

coef_wage_mle_vec <- make_named_vector(coef_wage_mle)

# Коэффициенты уравнения зарплаты 2-step
coef_wage_2step <- coef(
  model_heckman_2step,
  type = "coef2",
  eq2 = 1
)

coef_wage_2step_vec <- make_named_vector(coef_wage_2step)

# Сравнительная таблица
comparison_newey_heckman <- data.frame(
  variable = main_coef_names,
  Heckman_MLE = round(coef_wage_mle_vec[main_coef_names], 4),
  Heckman_2step = round(coef_wage_2step_vec[main_coef_names], 4),
  Newey = round(as.numeric(newey_main_est[main_coef_names]), 4),
  Newey_bootstrap_se = round(as.numeric(boot_se_newey[main_coef_names]), 4)
)

comparison_newey_heckman






















#---------------------------------------------------
# Часть 5. Модель с усечением по центру
# 5.1. Самостоятельная реализация центрально усеченной регрессии
#---------------------------------------------------

options(scipen = 999)

#---------------------------------------------------
# 1. Сюжет и симуляция данных
#---------------------------------------------------

# Сюжет:
# Исследуется потенциальный ежемесячный расход клиента на доставку еды.
# В аналитическую базу попадают только "нетипичные" клиенты:
# с очень низкими расходами или с очень высокими расходами.
# Клиенты со средними расходами не попадают в базу, то есть их y не наблюдается.
#
# Поэтому наблюдение попадает в выборку только если:
# y <= a или y >= b.
#
# Если a < y < b, наблюдение удаляется из выборки полностью.

set.seed(123)

n <- 50000

data_full <- data.frame(
  income_index   = rnorm(n, mean = 0, sd = 1),
  discount_index = rnorm(n, mean = 0, sd = 1),
  activity_index = rnorm(n, mean = 0, sd = 1)
)

# Истинные параметры
beta_true <- c(
  "(Intercept)"    = 7.0,
  "income_index"   = 1.2,
  "discount_index" = -0.8,
  "activity_index" = 0.6
)

sigma_true <- 2.0

# Латентная зависимая переменная
data_full$y_star <- beta_true["(Intercept)"] +
  beta_true["income_index"]   * data_full$income_index +
  beta_true["discount_index"] * data_full$discount_index +
  beta_true["activity_index"] * data_full$activity_index +
  rnorm(n, mean = 0, sd = sigma_true)

# Интервал центрального усечения
a <- 6
b <- 8

# Наблюдаются только значения вне интервала (a, b)
data_obs <- data_full[
  data_full$y_star <= a | data_full$y_star >= b,
]

data_obs$y <- data_obs$y_star

# Проверка доли наблюдений
n_full <- nrow(data_full)
n_obs <- nrow(data_obs)
n_drop <- n_full - n_obs

truncation_info <- data.frame(
  indicator = c(
    "Полная выборка",
    "Наблюдаемая выборка",
    "Удалено из-за центрального усечения",
    "Доля наблюдаемых"
  ),
  value = c(
    n_full,
    n_obs,
    n_drop,
    round(n_obs / n_full, 4)
  )
)

truncation_info
summary(data_obs$y)

#---------------------------------------------------
# 2. Наивный МНК по усеченной выборке
#---------------------------------------------------

model_ols_truncated <- lm(
  y ~ income_index + discount_index + activity_index,
  data = data_obs
)

summary(model_ols_truncated)

#---------------------------------------------------
# 3. Логарифм функции правдоподобия
#---------------------------------------------------

# Модель:
# y_i^* = x_i beta + eps_i,
# eps_i ~ N(0, sigma^2).
#
# Наблюдение попадает в выборку только если:
# y_i^* <= a или y_i^* >= b.
#
# Поэтому условная плотность наблюдаемого y:
#
# f(y_i | y_i <= a или y_i >= b, x_i)
# =
# [1 / sigma * phi((y_i - x_i beta) / sigma)]
# /
# [Phi((a - x_i beta) / sigma) + 1 - Phi((b - x_i beta) / sigma)].
#
# Максимизируем сумму логарифмов этой условной плотности.

lnL_center_trunc <- function(par, y, X, a, b) {
  
  k <- ncol(X)
  
  beta <- par[1:k]
  log_sigma <- par[k + 1]
  sigma <- exp(log_sigma)
  
  mu <- as.numeric(X %*% beta)
  
  z_y <- (y - mu) / sigma
  z_a <- (a - mu) / sigma
  z_b <- (b - mu) / sigma
  
  # Лог плотности числителя
  log_density <- dnorm(z_y, log = TRUE) - log_sigma
  
  # Вероятность попадания наблюдения в выборку:
  # P(y <= a | x) + P(y >= b | x)
  p_observed <- pnorm(z_a) + (1 - pnorm(z_b))
  
  # Защита от log(0)
  p_observed <- pmax(p_observed, 1e-12)
  
  log_likelihood <- sum(log_density - log(p_observed))
  
  return(log_likelihood)
}

#---------------------------------------------------
# 4. Оценивание методом максимального правдоподобия
#---------------------------------------------------

X_obs <- model.matrix(
  y ~ income_index + discount_index + activity_index,
  data = data_obs
)

y_obs <- data_obs$y

# Начальные значения берем из наивного МНК
start_beta <- coef(model_ols_truncated)
start_log_sigma <- log(sigma(model_ols_truncated))

start_par <- c(start_beta, start_log_sigma)

opt_center_trunc <- optim(
  par = start_par,
  fn = lnL_center_trunc,
  y = y_obs,
  X = X_obs,
  a = a,
  b = b,
  method = "BFGS",
  hessian = TRUE,
  control = list(
    fnscale = -1,
    maxit = 10000,
    reltol = 1e-12
  )
)

opt_center_trunc$convergence
opt_center_trunc$value
opt_center_trunc$par

#---------------------------------------------------
# 5. Извлекаем оценки
#---------------------------------------------------

k <- ncol(X_obs)

beta_hat <- opt_center_trunc$par[1:k]
names(beta_hat) <- colnames(X_obs)

log_sigma_hat <- opt_center_trunc$par[k + 1]
sigma_hat <- exp(log_sigma_hat)

beta_hat
sigma_hat

#---------------------------------------------------
# 6. Стандартные ошибки через обратный Гессиан
#---------------------------------------------------

# Так как мы максимизировали log-likelihood,
# асимптотическая ковариационная матрица:
# Var(theta_hat) = (-H)^(-1),
# где H — Гессиан log-likelihood в точке максимума.

H <- opt_center_trunc$hessian

vcov_par <- solve(-H)

se_par <- sqrt(diag(vcov_par))

se_beta <- se_par[1:k]
names(se_beta) <- colnames(X_obs)

se_log_sigma <- se_par[k + 1]

# Delta method:
# sigma = exp(log_sigma)
# se(sigma) = sigma * se(log_sigma)
se_sigma <- sigma_hat * se_log_sigma

se_beta
se_sigma

#---------------------------------------------------
# 7. Итоговая таблица коэффициентов
#---------------------------------------------------

coef_table_center_trunc <- data.frame(
  variable = names(beta_hat),
  true_value = round(beta_true[names(beta_hat)], 4),
  naive_ols = round(coef(model_ols_truncated)[names(beta_hat)], 4),
  mle_center_trunc = round(beta_hat, 4),
  mle_std_error = round(se_beta, 4),
  z_value = round(beta_hat / se_beta, 4),
  p_value = signif(2 * pnorm(abs(beta_hat / se_beta), lower.tail = FALSE), 4)
)

rownames(coef_table_center_trunc) <- NULL

coef_table_center_trunc

#---------------------------------------------------
# 8. Таблица для sigma
#---------------------------------------------------

sigma_table_center_trunc <- data.frame(
  parameter = "sigma",
  true_value = round(sigma_true, 4),
  naive_ols = round(sigma(model_ols_truncated), 4),
  mle_center_trunc = round(sigma_hat, 4),
  mle_std_error = round(se_sigma, 4)
)

sigma_table_center_trunc

#---------------------------------------------------
# 9. Доверительные интервалы для коэффициентов
#---------------------------------------------------

ci_center_trunc <- data.frame(
  variable = names(beta_hat),
  estimate = round(beta_hat, 4),
  std_error = round(se_beta, 4),
  ci_low_95 = round(beta_hat - qnorm(0.975) * se_beta, 4),
  ci_high_95 = round(beta_hat + qnorm(0.975) * se_beta, 4),
  true_value = round(beta_true[names(beta_hat)], 4)
)

rownames(ci_center_trunc) <- NULL

ci_center_trunc

#---------------------------------------------------
# 10. Проверка, что MLE ближе к истинным коэффициентам,
#     чем наивный МНК по усеченной выборке
#---------------------------------------------------

comparison_quality <- data.frame(
  variable = names(beta_hat),
  abs_error_ols = round(abs(coef(model_ols_truncated)[names(beta_hat)] -
                              beta_true[names(beta_hat)]), 4),
  abs_error_mle = round(abs(beta_hat -
                              beta_true[names(beta_hat)]), 4)
)

rownames(comparison_quality) <- NULL

comparison_quality



#---------------------------------------------------
# 5.2. Функция для условного математического ожидания
#      в модели с центральным усечением
#---------------------------------------------------

center_trunc_cond_mean <- function(newdata, beta_hat, sigma_hat, a, b) {
  
  # Матрица регрессоров для новых наблюдений
  X_new <- model.matrix(
    ~ income_index + discount_index + activity_index,
    data = newdata
  )
  
  # Линейный индекс mu = x beta
  mu_hat <- as.numeric(X_new %*% beta_hat)
  
  # Стандартизированные границы усечения
  z_a <- (a - mu_hat) / sigma_hat
  z_b <- (b - mu_hat) / sigma_hat
  
  # Вероятность попадания наблюдения в выборку
  p_observed <- pnorm(z_a) + (1 - pnorm(z_b))
  
  # Условное математическое ожидание:
  # E(y | y <= a or y >= b, x)
  Ey_cond <- mu_hat +
    sigma_hat * (dnorm(z_b) - dnorm(z_a)) / p_observed
  
  result <- data.frame(
    mu_hat = round(mu_hat, 4),
    z_a = round(z_a, 4),
    z_b = round(z_b, 4),
    p_observed = round(p_observed, 4),
    Ey_conditional = round(Ey_cond, 4)
  )
  
  return(result)
}

#---------------------------------------------------
# Пример: клиент с произвольными характеристиками
#---------------------------------------------------

new_client <- data.frame(
  income_index = 0.5,
  discount_index = -0.2,
  activity_index = 1.0
)

center_trunc_cond_mean(
  newdata = new_client,
  beta_hat = beta_hat,
  sigma_hat = sigma_hat,
  a = a,
  b = b
)



#---------------------------------------------------
# ДОПОЛНИТЕЛЬНО!!!
#---------------------------------------------------

#---------------------------------------------------
# Часть 3. Модель Хекмана с логарифмом зарплаты
#---------------------------------------------------

options(scipen = 999)

library(switchSelection)

#---------------------------------------------------
# 0. Подготовка зависимой переменной
#---------------------------------------------------

# В модели Хекмана зарплата наблюдается только для работающих женщин.
# Поэтому для lfp = 0 ставим NA.
# Для работающих берем log(wage).

table(data$lfp, data$wage <= 0)

data$log_wage_heckman <- ifelse(
  data$lfp == 1,
  log(data$wage),
  NA
)

# Проверка
table(data$lfp, is.na(data$log_wage_heckman))
summary(data$log_wage_heckman)

#---------------------------------------------------
# 1. Спецификации
#---------------------------------------------------

# Уравнение занятости / отбора
selection_formula_log <- lfp ~ age + kids5 + nwifeinc

# Уравнение логарифма зарплаты
log_wage_formula_heckman <- log_wage_heckman ~ educ + exper + age

#---------------------------------------------------
# 3.1. Оценивание модели Хекмана методом ММП
#      с log(wage) в уравнении зарплаты
#---------------------------------------------------

model_heckman_mle_log <- msel(
  formula   = selection_formula_log,
  formula2  = log_wage_formula_heckman,
  data      = data,
  estimator = "ml"
)

summary(model_heckman_mle_log)

#---------------------------------------------------
# Извлечение коэффициентов и параметров модели
#---------------------------------------------------

# Коэффициенты уравнения занятости
coef_selection_mle_log <- coef(
  model_heckman_mle_log,
  type = "coef",
  eq = 1
)

# Коэффициенты уравнения log(wage)
coef_log_wage_mle <- coef(
  model_heckman_mle_log,
  type = "coef2",
  eq2 = 1
)

# Стандартное отклонение ошибки уравнения log(wage)
sigma_mle_log <- sigma(model_heckman_mle_log)

# Ковариация между ошибками уравнения занятости и уравнения log(wage)
cov_mle_log <- coef(
  model_heckman_mle_log,
  type = "cov12",
  eq = 1,
  regime = 0
)

# Корреляция ошибок
rho_mle_log <- cov_mle_log / sigma_mle_log

coef_selection_mle_log
coef_log_wage_mle
sigma_mle_log
cov_mle_log
rho_mle_log

# Log-likelihood, AIC, BIC
logLik(model_heckman_mle_log)
AIC(model_heckman_mle_log)
BIC(model_heckman_mle_log)

# Количество наблюдений
n_total <- nrow(data)
n_work <- sum(data$lfp == 1)
n_not_work <- sum(data$lfp == 0)

data.frame(
  indicator = c("Всего наблюдений", "Работающие", "Неработающие"),
  value = c(n_total, n_work, n_not_work)
)

#---------------------------------------------------
# 3.3. Двухшаговый метод Хекмана с log(wage)
#---------------------------------------------------

model_heckman_2step_log <- msel(
  formula   = selection_formula_log,
  formula2  = log_wage_formula_heckman,
  data      = data,
  estimator = "2step"
)

summary(model_heckman_2step_log)

#---------------------------------------------------
# Функция для удобного извлечения коэффициентов
#---------------------------------------------------

make_named_vector <- function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    v <- as.numeric(x[1, ])
    names(v) <- colnames(x)
  } else {
    v <- as.numeric(x)
    names(v) <- names(x)
  }
  return(v)
}

#---------------------------------------------------
# Коэффициенты уравнения log(wage)
#---------------------------------------------------

coef_log_wage_mle <- coef(
  model_heckman_mle_log,
  type = "coef2",
  eq2 = 1
)

coef_log_wage_2step <- coef(
  model_heckman_2step_log,
  type = "coef2",
  eq2 = 1
)

coef_log_wage_mle_vec <- make_named_vector(coef_log_wage_mle)
coef_log_wage_2step_vec <- make_named_vector(coef_log_wage_2step)

comparison_log_wage <- data.frame(
  variable = names(coef_log_wage_mle_vec),
  Heckman_MLE_log = round(coef_log_wage_mle_vec, 4),
  Heckman_2step_log = round(
    coef_log_wage_2step_vec[names(coef_log_wage_mle_vec)],
    4
  ),
  difference_2step_minus_MLE = round(
    coef_log_wage_2step_vec[names(coef_log_wage_mle_vec)] -
      coef_log_wage_mle_vec,
    4
  )
)

comparison_log_wage

#---------------------------------------------------
# Коэффициенты уравнения занятости
#---------------------------------------------------

coef_selection_mle_log <- coef(
  model_heckman_mle_log,
  type = "coef",
  eq = 1
)

coef_selection_2step_log <- coef(
  model_heckman_2step_log,
  type = "coef",
  eq = 1
)

coef_selection_mle_log_vec <- make_named_vector(coef_selection_mle_log)
coef_selection_2step_log_vec <- make_named_vector(coef_selection_2step_log)

comparison_selection_log <- data.frame(
  variable = names(coef_selection_mle_log_vec),
  Heckman_MLE_log = round(coef_selection_mle_log_vec, 4),
  Heckman_2step_log = round(
    coef_selection_2step_log_vec[names(coef_selection_mle_log_vec)],
    4
  ),
  difference_2step_minus_MLE = round(
    coef_selection_2step_log_vec[names(coef_selection_mle_log_vec)] -
      coef_selection_mle_log_vec,
    4
  )
)

comparison_selection_log

#---------------------------------------------------
# Ручная двухшаговая процедура Хекмана с log(wage)
#---------------------------------------------------

# Первый шаг: пробит-модель занятости
model_selection_probit_log <- glm(
  lfp ~ age + kids5 + nwifeinc,
  data = data,
  family = binomial(link = "probit")
)

summary(model_selection_probit_log)

# Линейный индекс первого шага
data$selection_index_log <- predict(
  model_selection_probit_log,
  type = "link"
)

# Обратное отношение Миллса для работающих:
# lambda_i = phi(z_i gamma) / Phi(z_i gamma)
data$lambda_heckman_log <- NA

data$lambda_heckman_log[data$lfp == 1] <-
  dnorm(data$selection_index_log[data$lfp == 1]) /
  pnorm(data$selection_index_log[data$lfp == 1])

# Второй шаг: уравнение log(wage) по работающим с lambda
model_heckman_2step_manual_log <- lm(
  log_wage_heckman ~ educ + exper + age + lambda_heckman_log,
  data = data
)

summary(model_heckman_2step_manual_log)

# Коэффициент при lambda оценивает rho * sigma
lambda_coef_manual_log <- coef(
  model_heckman_2step_manual_log
)["lambda_heckman_log"]

lambda_coef_manual_log

#---------------------------------------------------
# 3.4. Корреляция ошибок и самоотбор
#---------------------------------------------------

sigma_mle_log <- sigma(model_heckman_mle_log)

cov_mle_log <- coef(
  model_heckman_mle_log,
  type = "cov12",
  eq = 1,
  regime = 0
)

rho_mle_log <- cov_mle_log / sigma_mle_log

# Для ручной двухшаговой модели:
# lambda_coef_manual_log оценивает rho * sigma.
# Приближенно rho можно получить как lambda_coef / residual sigma второго шага.
sigma_2step_manual_log <- sigma(model_heckman_2step_manual_log)

rho_2step_manual_log <- as.numeric(
  lambda_coef_manual_log / sigma_2step_manual_log
)

self_selection_log_summary <- data.frame(
  model = c(
    "Heckman MLE log",
    "Heckman 2-step manual log"
  ),
  indicator = c(
    "rho",
    "approx rho"
  ),
  value = round(c(
    as.numeric(rho_mle_log),
    rho_2step_manual_log
  ), 4)
)

self_selection_log_summary


#---------------------------------------------------
# 3.5. Условные ожидания и предельные эффекты
#      в MLE-модели Хекмана с log(wage)
#---------------------------------------------------

# Индивид с произвольными характеристиками
Boris_heckman_log <- data.frame(
  educ = 12,
  exper = 10,
  age = 40,
  kids5 = 1,
  nwifeinc = 20
)

#---------------------------------------------------
# Оценки из MLE-модели
#---------------------------------------------------

beta_hat_log <- coef(
  model_heckman_mle_log,
  type = "coef2",
  eq2 = 1
)

beta_hat_log <- as.numeric(beta_hat_log[1, ])
names(beta_hat_log) <- c("(Intercept)", "educ", "exper", "age")

gamma_hat_log <- coef(
  model_heckman_mle_log,
  type = "coef",
  eq = 1
)

sigma_hat_log <- as.numeric(
  sigma(model_heckman_mle_log)
)

rho_sigma_hat_log <- as.numeric(
  coef(
    model_heckman_mle_log,
    type = "cov12",
    eq = 1,
    regime = 0
  )
)

#---------------------------------------------------
# Извлекаем cut1
#---------------------------------------------------

cut1_hat_log <- tryCatch(
  as.numeric(coef(model_heckman_mle_log, type = "cuts", eq = 1)),
  error = function(e) NA_real_
)

cut1_hat_log

# Если cut1_hat_log вывел NA, нужно вручную взять cut1 из summary:
# cut1_hat_log <- значение cut1 из summary(model_heckman_mle_log)

#---------------------------------------------------
# Линейный индекс уравнения log(wage)
#---------------------------------------------------

mu_hat_log <- as.numeric(
  beta_hat_log["(Intercept)"] +
    beta_hat_log["educ"] * Boris_heckman_log$educ +
    beta_hat_log["exper"] * Boris_heckman_log$exper +
    beta_hat_log["age"] * Boris_heckman_log$age
)

#---------------------------------------------------
# Индекс уравнения занятости
# В параметризации msel:
# P(lfp = 1) = Phi(eta - cut1)
#---------------------------------------------------

eta_hat_log <- as.numeric(
  gamma_hat_log["age"] * Boris_heckman_log$age +
    gamma_hat_log["kids5"] * Boris_heckman_log$kids5 +
    gamma_hat_log["nwifeinc"] * Boris_heckman_log$nwifeinc
)

a_hat_log <- eta_hat_log - cut1_hat_log

Phi_hat_log <- pnorm(a_hat_log)
phi_hat_log <- dnorm(a_hat_log)

lambda_1_log <- phi_hat_log / Phi_hat_log
lambda_0_log <- phi_hat_log / (1 - Phi_hat_log)

#---------------------------------------------------
# A) Условные ожидания log(wage*)
#---------------------------------------------------

Elogw_z1 <- mu_hat_log + rho_sigma_hat_log * lambda_1_log

Elogw_z0 <- mu_hat_log - rho_sigma_hat_log * lambda_0_log

#---------------------------------------------------
# Б) Предельные эффекты age
#---------------------------------------------------

beta_age_log <- beta_hat_log["age"]
gamma_age_log <- gamma_hat_log["age"]

lambda_1_deriv_log <- -lambda_1_log * (a_hat_log + lambda_1_log)

lambda_0_deriv_log <- lambda_0_log * (lambda_0_log - a_hat_log)

ME_age_z1_log <- beta_age_log +
  rho_sigma_hat_log * lambda_1_deriv_log * gamma_age_log

ME_age_z0_log <- beta_age_log -
  rho_sigma_hat_log * lambda_0_deriv_log * gamma_age_log

heckman_35_results_log <- data.frame(
  indicator = c(
    "mu_hat_log",
    "eta_hat_log",
    "cut1_hat_log",
    "a_hat_log",
    "Phi(a_hat_log)",
    "phi(a_hat_log)",
    "lambda_1_log",
    "lambda_0_log",
    "E(log wage* | z = 1)",
    "E(log wage* | z = 0)",
    "dE(log wage* | z = 1) / d age",
    "dE(log wage* | z = 0) / d age"
  ),
  value = round(c(
    mu_hat_log,
    eta_hat_log,
    cut1_hat_log,
    a_hat_log,
    Phi_hat_log,
    phi_hat_log,
    lambda_1_log,
    lambda_0_log,
    Elogw_z1,
    Elogw_z0,
    ME_age_z1_log,
    ME_age_z0_log
  ), 4)
)

heckman_35_results_log


# ДОПОЛНИТЕЛЬНО 2

#---------------------------------------------------
# Часть 4. Модель Ньюи с log(wage)
# 4.2. Оценивание модели Ньюи с LOOCV и bootstrap SE
#---------------------------------------------------

options(scipen = 999)

library(switchSelection)

#---------------------------------------------------
# 0. Подготовка данных
#---------------------------------------------------

# В модели Ньюи / Хекмана зарплата наблюдается только для работающих.
# Поэтому для lfp = 0 ставим NA.
# Для работающих берем log(wage).

data$log_wage_heckman <- ifelse(
  data$lfp == 1,
  log(data$wage),
  NA
)

# Проверка
table(data$lfp, is.na(data$log_wage_heckman))
summary(data$log_wage_heckman)

# Спецификации, как в модели Хекмана
selection_formula_log <- lfp ~ age + kids5 + nwifeinc
log_wage_formula_heckman <- log_wage_heckman ~ educ + exper + age

#---------------------------------------------------
# 1. Функция для расчета LOOCV RMSE для lm
#---------------------------------------------------

loocv_rmse_lm <- function(model) {
  e <- residuals(model)
  h <- hatvalues(model)
  sqrt(mean((e / (1 - h))^2, na.rm = TRUE))
}

#---------------------------------------------------
# 2. Функция оценки Newey-модели для log(wage)
#---------------------------------------------------

estimate_newey_log <- function(df, degrees = 1:4) {
  
  #---------------------------------------------------
  # 1-й шаг: пробит-модель занятости
  #---------------------------------------------------
  
  model_sel <- glm(
    lfp ~ age + kids5 + nwifeinc,
    data = df,
    family = binomial(link = "probit")
  )
  
  # Линейный индекс отбора
  df$newey_index <- predict(model_sel, type = "link")
  
  # Сглаживающая функция:
  # g(w gamma) = lambda(w gamma) = phi(w gamma) / Phi(w gamma)
  Phi_index <- pmax(pnorm(df$newey_index), 1e-8)
  df$newey_lambda <- dnorm(df$newey_index) / Phi_index
  
  # Берем только работающих, у которых log(wage) наблюдается
  df_work <- df[df$lfp == 1 & !is.na(df$log_wage_heckman), ]
  
  #---------------------------------------------------
  # Оцениваем модели с разными степенями полинома
  #---------------------------------------------------
  
  cv_results <- data.frame(
    degree = degrees,
    loocv_rmse = NA_real_
  )
  
  models <- vector("list", length(degrees))
  names(models) <- paste0("degree_", degrees)
  
  for (j in seq_along(degrees)) {
    
    k <- degrees[j]
    
    poly_terms <- paste0("I(newey_lambda^", 1:k, ")", collapse = " + ")
    
    newey_formula <- as.formula(
      paste(
        "log_wage_heckman ~ educ + exper + age +",
        poly_terms
      )
    )
    
    model_k <- lm(newey_formula, data = df_work)
    
    models[[j]] <- model_k
    cv_results$loocv_rmse[j] <- loocv_rmse_lm(model_k)
  }
  
  #---------------------------------------------------
  # Выбор степени с минимальным LOOCV RMSE
  #---------------------------------------------------
  
  best_id <- which.min(cv_results$loocv_rmse)
  best_degree <- cv_results$degree[best_id]
  best_model <- models[[best_id]]
  
  return(list(
    selection_model = model_sel,
    cv_results = cv_results,
    best_degree = best_degree,
    best_model = best_model,
    df_with_lambda = df
  ))
}

#---------------------------------------------------
# 3. Оценка Newey-модели на исходной выборке
#---------------------------------------------------

newey_log_result <- estimate_newey_log(data, degrees = 1:4)

newey_log_result$cv_results
newey_log_result$best_degree

summary(newey_log_result$best_model)

# Коэффициенты Newey log-модели
coef_newey_log <- coef(newey_log_result$best_model)
coef_newey_log


#---------------------------------------------------
# 4. Bootstrap стандартные ошибки для Newey log-модели
#---------------------------------------------------

set.seed(123)

B <- 300
degrees <- 1:4

# Основные коэффициенты уравнения log(wage)
main_coef_names <- c("(Intercept)", "educ", "exper", "age")

boot_coef_log <- matrix(
  NA_real_,
  nrow = B,
  ncol = length(main_coef_names)
)

colnames(boot_coef_log) <- main_coef_names

boot_best_degree_log <- rep(NA_integer_, B)

for (b_iter in 1:B) {
  
  if (b_iter %% 25 == 0) {
    cat("Bootstrap iteration:", b_iter, "\n")
  }
  
  # Bootstrap-выборка
  boot_id <- sample(seq_len(nrow(data)), size = nrow(data), replace = TRUE)
  data_b <- data[boot_id, ]
  
  # В bootstrap-выборке заново создаем log_wage_heckman
  data_b$log_wage_heckman <- ifelse(
    data_b$lfp == 1,
    log(data_b$wage),
    NA
  )
  
  # Оценка Newey log-модели внутри bootstrap
  res_b <- tryCatch(
    estimate_newey_log(data_b, degrees = degrees),
    error = function(e) NULL
  )
  
  if (!is.null(res_b)) {
    
    boot_best_degree_log[b_iter] <- res_b$best_degree
    
    cb <- coef(res_b$best_model)
    
    # Сохраняем только основные коэффициенты уравнения log(wage)
    boot_coef_log[b_iter, ] <- cb[main_coef_names]
  }
}

# Bootstrap SE
boot_se_newey_log <- apply(boot_coef_log, 2, sd, na.rm = TRUE)

# Частота выбора степеней полинома
degree_freq_log <- table(boot_best_degree_log, useNA = "ifany")
degree_freq_log

#---------------------------------------------------
# 5. Таблица Newey log: оценки и bootstrap SE
#---------------------------------------------------

newey_log_main_est <- coef_newey_log[main_coef_names]

newey_log_table <- data.frame(
  variable = main_coef_names,
  estimate = round(as.numeric(newey_log_main_est), 4),
  bootstrap_se = round(as.numeric(boot_se_newey_log[main_coef_names]), 4)
)

newey_log_table

#---------------------------------------------------
# 6. Полная таблица выбранной Newey log-модели
#---------------------------------------------------

newey_log_full_coef <- coef(summary(newey_log_result$best_model))

newey_log_full_table <- data.frame(
  variable = rownames(newey_log_full_coef),
  estimate = round(newey_log_full_coef[, 1], 4),
  usual_std_error = round(newey_log_full_coef[, 2], 4),
  t_value = round(newey_log_full_coef[, 3], 4),
  p_value = signif(newey_log_full_coef[, 4], 4)
)

rownames(newey_log_full_table) <- NULL

newey_log_full_table

#---------------------------------------------------
# 7. Сравнение Newey log, Heckman MLE log и Heckman 2-step log
#---------------------------------------------------

# Если log-модели Хекмана уже оценены ранее, этот блок можно не запускать.
# Оставляю для воспроизводимости.

model_heckman_mle_log <- msel(
  formula   = selection_formula_log,
  formula2  = log_wage_formula_heckman,
  data      = data,
  estimator = "ml"
)

model_heckman_2step_log <- msel(
  formula   = selection_formula_log,
  formula2  = log_wage_formula_heckman,
  data      = data,
  estimator = "2step"
)

# Удобная функция
make_named_vector <- function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    v <- as.numeric(x[1, ])
    names(v) <- colnames(x)
  } else {
    v <- as.numeric(x)
    names(v) <- names(x)
  }
  return(v)
}

# Коэффициенты уравнения log(wage) MLE
coef_log_wage_mle <- coef(
  model_heckman_mle_log,
  type = "coef2",
  eq2 = 1
)

coef_log_wage_mle_vec <- make_named_vector(coef_log_wage_mle)

# Коэффициенты уравнения log(wage) 2-step
coef_log_wage_2step <- coef(
  model_heckman_2step_log,
  type = "coef2",
  eq2 = 1
)

coef_log_wage_2step_vec <- make_named_vector(coef_log_wage_2step)

# Сравнительная таблица
comparison_newey_heckman_log <- data.frame(
  variable = main_coef_names,
  Heckman_MLE_log = round(coef_log_wage_mle_vec[main_coef_names], 4),
  Heckman_2step_log = round(coef_log_wage_2step_vec[main_coef_names], 4),
  Newey_log = round(as.numeric(newey_log_main_est[main_coef_names]), 4),
  Newey_log_bootstrap_se = round(
    as.numeric(boot_se_newey_log[main_coef_names]),
    4
  )
)

comparison_newey_heckman_log

#---------------------------------------------------
# 8. Все ключевые результаты для отчета
#---------------------------------------------------

newey_log_result$cv_results
newey_log_result$best_degree
degree_freq_log

newey_log_table
newey_log_full_table
comparison_newey_heckman_log

