setwd("C:/Users/smirn/OneDrive/Рабочий стол/Microeconometrics")

# Загрузка данных
data <- readRDS("homework.rds")

# Проверка, что данные загрузились
str(data)
head(data)
summary(data)


# Преобразуем категориальные переменные в факторы
data$health <- factor(data$health,
                      levels = c("bad", "medium", "good"))

data$residence <- factor(data$residence,
                         levels = c("village", "city", "capital"))

data$educ <- factor(data$educ,
                    levels = c("basic", "vocational", "higher", "phd"))

# Бинарные переменные тоже можно оставить числовыми 0/1,
# потому что они будут удобно интерпретироваться в регрессиях

# EDA

# Размер выборки
dim(data)

# Названия переменных
names(data)

# Проверка пропусков по переменным
colSums(is.na(data))

# Распределение исходной зависимой переменной game
table(data$game)
prop.table(table(data$game))

# Распределение переменной choice: кошки / собаки
table(data$choice)
prop.table(table(data$choice))

# Описательные статистики по числовым переменным
summary(data[, c("income", "age", "hours", "bugs", "price")])

# Частотные таблицы по категориальным переменным
table(data$health)
table(data$residence)
table(data$educ)

# Частотные таблицы по бинарным переменным
table(data$marriage)
table(data$chl)
table(data$male)
table(data$cat)
table(data$dog)
table(data$design)

# Создаем бинарную переменную удовлетворенности
# 1 = игрок как минимум средне удовлетворен игрой: game = 1 или game = 2
# 0 = низкая удовлетворенность: game = 0
data$satisfied <- ifelse(data$game >= 1, 1, 0)

# Проверяем распределение исходной и новой зависимой переменной
table(data$game)
prop.table(table(data$game))

table(data$satisfied)
prop.table(table(data$satisfied))

# Линейно-вероятностная модель

lpm <- lm(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = data
)

# Результаты оценки
summary(lpm)


### ПРЕДЕЛЬНЫЕ ЭФФЕКТЫ

# Извлекаем коэффициенты из ЛВМ
b <- coef(lpm)

b0 <- b["(Intercept)"]
b_choice <- b["choice"]
b_age <- b["age"]
b_age2 <- b["I(age^2)"]
b_male <- b["male"]
b_choice_age <- b["choice:age"]

# Предельный эффект возраста для каждого наблюдения
data$me_age <- b_age + 2 * b_age2 * data$age + b_choice_age * data$choice

# Дискретный эффект choice: переход от собак к кошкам
data$me_choice <- b_choice + b_choice_age * data$age

# Дискретный эффект male
me_male <- b_male

# Средние предельные эффекты
ame_age <- mean(data$me_age)
ame_choice <- mean(data$me_choice)
ame_male <- me_male

# Пороговые значения
age_threshold_dogs <- -b_age / (2 * b_age2)
age_threshold_cats <- -(b_age + b_choice_age) / (2 * b_age2)
choice_threshold <- -b_choice / b_choice_age

# Таблица предельных эффектов
me_table <- data.frame(
  Переменная = c("choice", "age", "male"),
  Тип_эффекта = c(
    "Средний дискретный эффект",
    "Средний предельный эффект",
    "Дискретный эффект"
  ),
  Формула = c(
    "β_choice + β_choice:age * age",
    "β_age + 2β_age2 * age + β_choice:age * choice",
    "β_male"
  ),
  Эффект = c(ame_choice, ame_age, ame_male)
)

# Округляем только числовой столбец
me_table$Эффект <- round(me_table$Эффект, 4)

me_table

# Пороговые значения
age_threshold_dogs
age_threshold_cats
choice_threshold


# 2.4 Бутстрап

# 2.4. Бутстрап для проверки значимости коэффициентов ЛВМ

set.seed(123)

# Число бутстрап-итераций
boot_iter <- 1000

# Исходная модель
lpm <- lm(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = data
)

# Исходные МНК-оценки коэффициентов
coef_lpm <- coef(lpm)

# Матрица для хранения бутстрап-оценок
coef_boot <- matrix(
  NA,
  nrow = boot_iter,
  ncol = length(coef_lpm)
)

colnames(coef_boot) <- names(coef_lpm)

# Размер исходной выборки
n <- nrow(data)

# Бутстрап
for (i in 1:boot_iter) {
  
  # Случайно выбираем строки с возвращением
  boot_ind <- sample(1:n, size = n, replace = TRUE)
  
  # Формируем бутстрап-выборку
  data_boot <- data[boot_ind, ]
  
  # Оцениваем модель на бутстрап-выборке
  lpm_boot <- lm(
    satisfied ~ choice + age + I(age^2) + male + choice:age,
    data = data_boot
  )
  
  # Сохраняем оценки коэффициентов
  coef_boot[i, ] <- coef(lpm_boot)
}

# Бутстрапированные стандартные ошибки
boot_se <- apply(coef_boot, 2, sd)

# Бутстрапированные 95% доверительные интервалы
boot_ci_left <- apply(coef_boot, 2, quantile, probs = 0.025)
boot_ci_right <- apply(coef_boot, 2, quantile, probs = 0.975)

# Проверка значимости:
# если 0 не попадает в 95% ДИ, коэффициент значим на 5%-м уровне
significant_5 <- ifelse(
  boot_ci_left > 0 | boot_ci_right < 0,
  "Да",
  "Нет"
)

# Таблица результатов
boot_table <- data.frame(
  Переменная = names(coef_lpm),
  Оценка = coef_lpm,
  Бутстрап_SE = boot_se,
  CI_2.5 = boot_ci_left,
  CI_97.5 = boot_ci_right,
  Значим_на_5_процентах = significant_5
)

# Округление числовых столбцов
num_cols <- sapply(boot_table, is.numeric)
boot_table[, num_cols] <- round(boot_table[, num_cols], 4)

boot_table

# 2.5. Робастная ковариационная матрица для ЛВМ

# Исходная линейно-вероятностная модель
lpm <- lm(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = data
)

# Оценки коэффициентов
coef_lpm <- coef(lpm)

# Матрица регрессоров X
X <- model.matrix(lpm)

# Предсказанные вероятности p_hat
p_hat <- as.vector(X %*% coef_lpm)

# Матрица (X'X)^(-1)
XX_inv <- solve(t(X) %*% X)

# Средняя часть сэндвича: sum x_i x_i' * p_i * (1 - p_i)
middle <- matrix(0, nrow = ncol(X), ncol = ncol(X))

for (i in 1:nrow(X)) {
  x_i <- matrix(X[i, ], ncol = 1)
  middle <- middle + x_i %*% t(x_i) * p_hat[i] * (1 - p_hat[i])
}

# Состоятельная оценка ковариационной матрицы
cov_robust <- XX_inv %*% middle %*% XX_inv

# Робастные стандартные ошибки
se_robust <- sqrt(diag(cov_robust))

# Тестовые статистики
t_robust <- coef_lpm / se_robust

# p-value на основе асимптотического нормального распределения
p_value_robust <- 2 * pmin(
  pnorm(t_robust),
  1 - pnorm(t_robust)
)

# Таблица результатов
robust_table <- data.frame(
  Переменная = names(coef_lpm),
  Оценка = coef_lpm,
  Робастная_SE = se_robust,
  t_статистика = t_robust,
  p_value = p_value_robust
)

# Округляем числовые столбцы
num_cols <- sapply(robust_table, is.numeric)
robust_table[, num_cols] <- round(robust_table[, num_cols], 4)

robust_table

#_______________________________________________________________________________

# БЛОК 3

model_probit <- glm(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = data,
  family = binomial(link = "probit")
)

summary(model_probit)

# Линейный индекс пробит-модели
xb_probit <- predict(model_probit, type = "link")

# Коэффициенты
b <- coef(model_probit)

# Предельный эффект age для каждого наблюдения
data$me_age_probit <- dnorm(xb_probit) * (
  b["age"] +
    2 * b["I(age^2)"] * data$age +
    b["choice:age"] * data$choice
)

# Средний предельный эффект age
ame_age_probit <- mean(data$me_age_probit)

# Дискретный эффект male
data_male_1 <- data
data_male_0 <- data

data_male_1$male <- 1
data_male_0$male <- 0

p_male_1 <- predict(model_probit, newdata = data_male_1, type = "response")
p_male_0 <- predict(model_probit, newdata = data_male_0, type = "response")

data$me_male_probit <- p_male_1 - p_male_0

# Средний дискретный эффект male
ame_male_probit <- mean(data$me_male_probit)

# Таблица результатов
me_probit_table <- data.frame(
  Переменная = c("age", "male"),
  Тип_эффекта = c(
    "Средний предельный эффект",
    "Средний дискретный эффект"
  ),
  Формула = c(
    "φ(xβ) * (β_age + 2β_age2 * age + β_choice:age * choice)",
    "Φ(xβ | male = 1) - Φ(xβ | male = 0)"
  ),
  Эффект = c(ame_age_probit, ame_male_probit)
)

me_probit_table$Эффект <- round(me_probit_table$Эффект, 4)

me_probit_table


# СРАВНЕНИЕ

# 3.5. Доля верных прогнозов

# Пробит-модель: предсказанные вероятности
p_probit <- predict(model_probit, type = "response")

# Прогноз класса: 1, если вероятность >= 0.5; 0 иначе
pred_probit <- ifelse(p_probit >= 0.5, 1, 0)

# Доля верных прогнозов пробит-модели
accuracy_probit <- mean(pred_probit == data$satisfied)

# Наивный прогноз:
# всегда предсказываем самый частый класс
majority_class <- as.numeric(names(which.max(table(data$satisfied))))
pred_naive <- rep(majority_class, nrow(data))

accuracy_naive <- mean(pred_naive == data$satisfied)

# Линейно-вероятностная модель
p_lpm <- predict(lpm)

# Прогноз класса по ЛВМ
pred_lpm <- ifelse(p_lpm >= 0.5, 1, 0)

accuracy_lpm <- mean(pred_lpm == data$satisfied)

# Итоговая таблица
accuracy_table <- data.frame(
  Модель = c(
    "Наивный прогноз",
    "Линейно-вероятностная модель",
    "Пробит-модель"
  ),
  Доля_верных_прогнозов = c(
    accuracy_naive,
    accuracy_lpm,
    accuracy_probit
  )
)

accuracy_table$Доля_верных_прогнозов <- round(
  accuracy_table$Доля_верных_прогнозов,
  4
)

accuracy_table

library(numDeriv)

# Функция дискретного эффекта male для Владимира
me_male_vladimir <- function(beta) {
  
  xb_male_1 <- beta["(Intercept)"] +
    beta["choice"] * 1 +
    beta["age"] * 22 +
    beta["I(age^2)"] * 22^2 +
    beta["male"] * 1 +
    beta["choice:age"] * 22
  
  xb_male_0 <- beta["(Intercept)"] +
    beta["choice"] * 1 +
    beta["age"] * 22 +
    beta["I(age^2)"] * 22^2 +
    beta["male"] * 0 +
    beta["choice:age"] * 22
  
  pnorm(xb_male_1) - pnorm(xb_male_0)
}

# Оценка предельного эффекта
me_hat <- me_male_vladimir(coef(model_probit))

# Градиент предельного эффекта по оцениваемым параметрам
grad_me <- grad(
  func = me_male_vladimir,
  x = coef(model_probit)
)

# Дельта-метод: оценка дисперсии предельного эффекта
var_me <- t(grad_me) %*% vcov(model_probit) %*% grad_me

# Стандартная ошибка
se_me <- sqrt(var_me)

# z-статистика
z_me <- me_hat / se_me

# p-value
p_value_me <- 2 * (1 - pnorm(abs(z_me)))

# Результаты
me_test_table <- data.frame(
  Переменная = "male",
  Предельный_эффект = me_hat,
  Стандартная_ошибка = se_me,
  z_статистика = z_me,
  p_value = p_value_me
)

me_test_table[, 2:5] <- round(me_test_table[, 2:5], 4)

me_test_table


# 3.6

library(numDeriv)

# Функция дискретного эффекта male для Владимира
me_male_vladimir <- function(beta) {
  
  xb_male_1 <- beta["(Intercept)"] +
    beta["choice"] * 1 +
    beta["age"] * 22 +
    beta["I(age^2)"] * 22^2 +
    beta["male"] * 1 +
    beta["choice:age"] * 22
  
  xb_male_0 <- beta["(Intercept)"] +
    beta["choice"] * 1 +
    beta["age"] * 22 +
    beta["I(age^2)"] * 22^2 +
    beta["male"] * 0 +
    beta["choice:age"] * 22
  
  pnorm(xb_male_1) - pnorm(xb_male_0)
}

# Оценка предельного эффекта
me_hat <- me_male_vladimir(coef(model_probit))

# Градиент предельного эффекта по оцениваемым параметрам
grad_me <- grad(
  func = me_male_vladimir,
  x = coef(model_probit)
)

# Дельта-метод: оценка дисперсии предельного эффекта
var_me <- t(grad_me) %*% vcov(model_probit) %*% grad_me

# Стандартная ошибка
se_me <- sqrt(var_me)

# z-статистика
z_me <- me_hat / se_me

# p-value
p_value_me <- 2 * (1 - pnorm(abs(z_me)))

# Результаты
me_test_table <- data.frame(
  Переменная = "male",
  Предельный_эффект = me_hat,
  Стандартная_ошибка = se_me,
  z_статистика = z_me,
  p_value = p_value_me
)

me_test_table[, 2:5] <- round(me_test_table[, 2:5], 4)

me_test_table


# 3.7

library(numDeriv)

# Функция дискретного эффекта choice для Владимира
me_choice_vladimir <- function(beta) {
  
  age_v <- 22
  male_v <- 1
  
  xb_choice_1 <- beta["(Intercept)"] +
    beta["choice"] * 1 +
    beta["age"] * age_v +
    beta["I(age^2)"] * age_v^2 +
    beta["male"] * male_v +
    beta["choice:age"] * (1 * age_v)
  
  xb_choice_0 <- beta["(Intercept)"] +
    beta["choice"] * 0 +
    beta["age"] * age_v +
    beta["I(age^2)"] * age_v^2 +
    beta["male"] * male_v +
    beta["choice:age"] * (0 * age_v)
  
  pnorm(xb_choice_1) - pnorm(xb_choice_0)
}

# Оценка предельного эффекта
me_choice_hat <- me_choice_vladimir(coef(model_probit))

# Градиент эффекта по оцениваемым параметрам
grad_choice_me <- grad(
  func = me_choice_vladimir,
  x = coef(model_probit)
)

# Дельта-метод
var_choice_me <- t(grad_choice_me) %*%
  vcov(model_probit) %*%
  grad_choice_me

se_choice_me <- sqrt(var_choice_me)

# z-статистика и p-value
z_choice_me <- me_choice_hat / se_choice_me

p_value_choice_me <- 2 * (1 - pnorm(abs(z_choice_me)))

# Таблица результатов
choice_me_test_table <- data.frame(
  Переменная = "choice",
  Предельный_эффект = me_choice_hat,
  Стандартная_ошибка = se_choice_me,
  z_статистика = z_choice_me,
  p_value = p_value_choice_me
)

choice_me_test_table[, 2:5] <- round(choice_me_test_table[, 2:5], 4)

choice_me_test_table


# 3.8


# Ковариационная матрица оценок пробит-модели
V <- vcov(model_probit)

# Градиент, рассчитанный вручную
grad_age <- c(
  -0.0046,
  -0.0046,
  0.1012,
  6.6891,
  -0.0046,
  0.1012
)

# Асимптотическая дисперсия предельного эффекта
asvar_me_age <- t(grad_age) %*% V %*% grad_age

# Стандартная ошибка
se_me_age <- sqrt(asvar_me_age)

asvar_me_age
se_me_age




#_______________________________________________________________________________

# ЧАСТЬ 4

# 4.1. LM-тест нормальности случайных ошибок в пробит-модели

# Матрица регрессоров из пробит-модели
X_mat <- model.matrix(model_probit)

# Зависимая переменная
y_vec <- data$satisfied

# Оценённый линейный индекс
eta_hat <- predict(model_probit, type = "link")

# Оценённые F и f стандартного нормального распределения
F_hat <- pnorm(eta_hat)
f_hat <- dnorm(eta_hat)

# Защита от деления на 0
eps <- 1e-10
F_hat <- pmin(pmax(F_hat, eps), 1 - eps)

# Обобщённый остаток / общая часть скор-функции
gr <- ((y_vec - F_hat) / (F_hat * (1 - F_hat))) * f_hat

# Производные по beta
score_beta <- X_mat * as.vector(gr)

# Дополнительные производные по theta1 и theta2
# Альтернатива стандартной нормальности:
# Phi(eta + theta1 * eta^2 + theta2 * eta^3)
score_theta1 <- gr * eta_hat^2
score_theta2 <- gr * eta_hat^3

# Матрица вкладов наблюдений в скор-функцию
jac <- cbind(score_beta, score_theta1, score_theta2)

# Суммарный скор
score_sum <- colSums(jac)

# LM-статистика через произведение Якобианов
LM_value <- as.numeric(
  t(score_sum) %*% solve(t(jac) %*% jac) %*% score_sum
)

# Число ограничений: theta1 = 0 и theta2 = 0
df_LM <- 2

# p-value
p_value_LM <- 1 - pchisq(LM_value, df = df_LM)

LM_normality_table <- data.frame(
  Тест = "LM-тест нормальности ошибок",
  LM_статистика = LM_value,
  df = df_LM,
  p_value = p_value_LM
)

LM_normality_table[, c("LM_статистика", "p_value")] <-
  round(LM_normality_table[, c("LM_статистика", "p_value")], 4)

LM_normality_table


# 4.2. LR-тест гомоскедастичности в пробит-модели

# Стандартизируем переменные уравнения дисперсии,
# чтобы избежать численных проблем из-за масштаба price
data$age_s <- as.numeric(scale(data$age))
data$bugs_s <- as.numeric(scale(data$bugs))
data$price_s <- as.numeric(scale(data$price))

# Основное уравнение
X <- model.matrix(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = data
)

# Уравнение дисперсии: без константы
W <- model.matrix(
  ~ age_s + bugs_s + price_s,
  data = data
)[, -1]

y <- data$satisfied

# Логарифм правдоподобия гетероскедастичной пробит-модели
lnL_hetprobit <- function(par, y, X, W) {
  
  k_beta <- ncol(X)
  
  beta <- par[1:k_beta]
  gamma <- par[(k_beta + 1):length(par)]
  
  xb <- as.vector(X %*% beta)
  wg <- as.vector(W %*% gamma)
  
  sigma_i <- exp(wg)
  
  p <- pnorm(xb / sigma_i)
  
  # защита от log(0)
  eps <- 1e-10
  p <- pmin(pmax(p, eps), 1 - eps)
  
  lnL <- sum(y * log(p) + (1 - y) * log(1 - p))
  
  return(lnL)
}

# Начальные значения: beta из обычной пробит-модели, gamma = 0
start_beta <- coef(model_probit)
start_gamma <- rep(0, ncol(W))

start_par <- c(start_beta, start_gamma)

# Оцениваем гетероскедастичную пробит-модель
opt_hetprobit <- optim(
  par = start_par,
  fn = lnL_hetprobit,
  y = y,
  X = X,
  W = W,
  method = "BFGS",
  hessian = TRUE,
  control = list(
    fnscale = -1,
    maxit = 10000,
    reltol = 1e-10
  )
)

# Логарифм правдоподобия полной модели
lnL_UR <- opt_hetprobit$value

# Логарифм правдоподобия ограниченной модели
lnL_R <- as.numeric(logLik(model_probit))

# LR-статистика
LR_hetero <- 2 * (lnL_UR - lnL_R)

# Число ограничений
df_LR <- ncol(W)

# p-value
p_value_LR_hetero <- 1 - pchisq(LR_hetero, df = df_LR)

# Таблица результатов
LR_hetero_table <- data.frame(
  Тест = "LR-тест гомоскедастичности",
  lnL_R = lnL_R,
  lnL_UR = lnL_UR,
  LR_статистика = LR_hetero,
  df = df_LR,
  p_value = p_value_LR_hetero
)

LR_hetero_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")] <-
  round(LR_hetero_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")], 4)

LR_hetero_table


# 4.3. Предельные эффекты в гетероскедастичной пробит-модели

# Коэффициенты гетероскедастичной пробит-модели
par_hat <- opt_hetprobit$par

k_beta <- ncol(X)

beta_hat <- par_hat[1:k_beta]
gamma_hat <- par_hat[(k_beta + 1):length(par_hat)]

names(beta_hat) <- colnames(X)
names(gamma_hat) <- colnames(W)

# Линейные индексы
xb_het <- as.vector(X %*% beta_hat)
wg_het <- as.vector(W %*% gamma_hat)

# sigma_i = exp(w_i gamma)
sigma_het <- exp(wg_het)

# z_i = x_i beta / exp(w_i gamma)
z_het <- xb_het / sigma_het

# sd(age), так как в уравнении дисперсии используется age_s
sd_age <- sd(data$age)

# Производная основного уравнения по age
d_xb_d_age <- beta_hat["age"] +
  2 * beta_hat["I(age^2)"] * data$age +
  beta_hat["choice:age"] * data$choice

# Производная w_i gamma по исходному age
d_wg_d_age <- gamma_hat["age_s"] / sd_age

# 1) Предельный эффект age на вероятность удовлетворенности
data$me_age_prob_het <- dnorm(z_het) *
  (d_xb_d_age - xb_het * d_wg_d_age) / sigma_het

# 2) Предельный эффект age на дисперсию случайной ошибки
data$me_age_var_het <- 2 * exp(2 * wg_het) * d_wg_d_age

# Средние предельные эффекты
ame_age_prob_het <- mean(data$me_age_prob_het)
ame_age_var_het <- mean(data$me_age_var_het)

# Таблица результатов
het_me_table <- data.frame(
  Переменная = c("age", "age"),
  Зависимая_величина = c(
    "Вероятность удовлетворенности",
    "Дисперсия случайной ошибки"
  ),
  Формула = c(
    "φ(z_i) * [(β_age + 2β_age2*age_i + β_choice:age*choice_i) - xβ_i*γ_age_s/sd(age)] / exp(wγ_i)",
    "2 * exp(2wγ_i) * γ_age_s/sd(age)"
  ),
  Средний_эффект = c(
    ame_age_prob_het,
    ame_age_var_het
  )
)

het_me_table$Средний_эффект <- round(het_me_table$Средний_эффект, 6)

het_me_table




# 4.4. LR-тесты ограничений для переменной age

# Неограниченная модель
model_UR <- model_probit
lnL_UR <- as.numeric(logLik(model_UR))

# -----------------------------
# 1) H0: beta_age = 0
# -----------------------------

model_R1 <- glm(
  satisfied ~ choice + I(age^2) + male + choice:age,
  data = data,
  family = binomial(link = "probit")
)

lnL_R1 <- as.numeric(logLik(model_R1))

LR_1 <- 2 * (lnL_UR - lnL_R1)
df_1 <- 1
p_1 <- 1 - pchisq(LR_1, df = df_1)


# -----------------------------
# 2) H0: beta_age = 0 и beta_age2 = 0
# -----------------------------

model_R2 <- glm(
  satisfied ~ choice + male + choice:age,
  data = data,
  family = binomial(link = "probit")
)

lnL_R2 <- as.numeric(logLik(model_R2))

LR_2 <- 2 * (lnL_UR - lnL_R2)
df_2 <- 2
p_2 <- 1 - pchisq(LR_2, df = df_2)


# -----------------------------
# 3) H0: beta_age = k * beta_age2
# -----------------------------

k <- 100

# Логарифм правдоподобия при ограничении beta_age = k * beta_age2
lnL_R3_fun <- function(par, data, k) {
  
  beta0 <- par[1]
  beta_choice <- par[2]
  beta_age2 <- par[3]
  beta_male <- par[4]
  beta_choice_age <- par[5]
  
  beta_age <- k * beta_age2
  
  xb <- beta0 +
    beta_choice * data$choice +
    beta_age * data$age +
    beta_age2 * data$age^2 +
    beta_male * data$male +
    beta_choice_age * data$choice * data$age
  
  p <- pnorm(xb)
  
  eps <- 1e-10
  p <- pmin(pmax(p, eps), 1 - eps)
  
  lnL <- sum(data$satisfied * log(p) +
               (1 - data$satisfied) * log(1 - p))
  
  return(lnL)
}

# Начальные значения на основе неограниченной модели
b_UR <- coef(model_UR)

start_R3 <- c(
  b_UR["(Intercept)"],
  b_UR["choice"],
  b_UR["I(age^2)"],
  b_UR["male"],
  b_UR["choice:age"]
)

opt_R3 <- optim(
  par = start_R3,
  fn = lnL_R3_fun,
  data = data,
  k = k,
  method = "BFGS",
  control = list(
    fnscale = -1,
    maxit = 10000,
    reltol = 1e-10
  )
)

lnL_R3 <- opt_R3$value

LR_3 <- 2 * (lnL_UR - lnL_R3)
df_3 <- 1
p_3 <- 1 - pchisq(LR_3, df = df_3)


# -----------------------------
# 4) H0: beta_age = k * beta_age2 и beta_male = t
# -----------------------------

t_value <- -0.2

lnL_R4_fun <- function(par, data, k, t_value) {
  
  beta0 <- par[1]
  beta_choice <- par[2]
  beta_age2 <- par[3]
  beta_choice_age <- par[4]
  
  beta_age <- k * beta_age2
  beta_male <- t_value
  
  xb <- beta0 +
    beta_choice * data$choice +
    beta_age * data$age +
    beta_age2 * data$age^2 +
    beta_male * data$male +
    beta_choice_age * data$choice * data$age
  
  p <- pnorm(xb)
  
  eps <- 1e-10
  p <- pmin(pmax(p, eps), 1 - eps)
  
  lnL <- sum(data$satisfied * log(p) +
               (1 - data$satisfied) * log(1 - p))
  
  return(lnL)
}

start_R4 <- c(
  b_UR["(Intercept)"],
  b_UR["choice"],
  b_UR["I(age^2)"],
  b_UR["choice:age"]
)

opt_R4 <- optim(
  par = start_R4,
  fn = lnL_R4_fun,
  data = data,
  k = k,
  t_value = t_value,
  method = "BFGS",
  control = list(
    fnscale = -1,
    maxit = 10000,
    reltol = 1e-10
  )
)

lnL_R4 <- opt_R4$value

LR_4 <- 2 * (lnL_UR - lnL_R4)
df_4 <- 2
p_4 <- 1 - pchisq(LR_4, df = df_4)


# -----------------------------
# Итоговая таблица
# -----------------------------

LR_tests_table <- data.frame(
  Номер = c(1, 2, 3, 4),
  Нулевая_гипотеза = c(
    "beta_age = 0",
    "beta_age = 0, beta_age2 = 0",
    "beta_age = 100 * beta_age2",
    "beta_age = 100 * beta_age2, beta_male = -0.2"
  ),
  lnL_R = c(lnL_R1, lnL_R2, lnL_R3, lnL_R4),
  lnL_UR = rep(lnL_UR, 4),
  LR_статистика = c(LR_1, LR_2, LR_3, LR_4),
  df = c(df_1, df_2, df_3, df_4),
  p_value = c(p_1, p_2, p_3, p_4)
)

num_cols <- sapply(LR_tests_table, is.numeric)
LR_tests_table[, num_cols] <- round(LR_tests_table[, num_cols], 4)

LR_tests_table

# 4.5. LR-тест: совместная модель или отдельные модели для мужчин и женщин

# Ограниченная модель: совместная модель для всех наблюдений
model_gender_R <- glm(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = data,
  family = binomial(link = "probit")
)

lnL_R <- as.numeric(logLik(model_gender_R))

# Неограниченная модель: отдельная модель для женщин
model_female <- glm(
  satisfied ~ choice + age + I(age^2) + choice:age,
  data = subset(data, male == 0),
  family = binomial(link = "probit")
)

lnL_female <- as.numeric(logLik(model_female))

# Неограниченная модель: отдельная модель для мужчин
model_male <- glm(
  satisfied ~ choice + age + I(age^2) + choice:age,
  data = subset(data, male == 1),
  family = binomial(link = "probit")
)

lnL_male <- as.numeric(logLik(model_male))

# Логарифм правдоподобия неограниченной модели
lnL_UR <- lnL_female + lnL_male

# LR-статистика
LR_gender <- 2 * (lnL_UR - lnL_R)

# Число ограничений
df_gender <- 4

# p-value
p_value_gender <- 1 - pchisq(LR_gender, df = df_gender)

# Таблица результатов
LR_gender_table <- data.frame(
  Тест = "LR-тест совместной модели для мужчин и женщин",
  lnL_R = lnL_R,
  lnL_UR = lnL_UR,
  LR_статистика = LR_gender,
  df = df_gender,
  p_value = p_value_gender
)

LR_gender_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")] <-
  round(LR_gender_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")], 4)

LR_gender_table

# 4.6. LR-тест: совместная модель или отдельные модели по residence

# Ограниченная модель: одна совместная модель с дамми residence
model_residence_R <- glm(
  satisfied ~ choice + age + I(age^2) + male + choice:age + residence,
  data = data,
  family = binomial(link = "probit")
)

lnL_R_residence <- as.numeric(logLik(model_residence_R))

# Неограниченные модели: отдельно для каждого типа населенного пункта

model_village <- glm(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = subset(data, residence == "village"),
  family = binomial(link = "probit")
)

lnL_village <- as.numeric(logLik(model_village))

model_city <- glm(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = subset(data, residence == "city"),
  family = binomial(link = "probit")
)

lnL_city <- as.numeric(logLik(model_city))

model_capital <- glm(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = subset(data, residence == "capital"),
  family = binomial(link = "probit")
)

lnL_capital <- as.numeric(logLik(model_capital))

# Логарифм правдоподобия неограниченной модели
lnL_UR_residence <- lnL_village + lnL_city + lnL_capital

# LR-статистика
LR_residence <- 2 * (lnL_UR_residence - lnL_R_residence)

# Число ограничений
df_residence <- 10

# p-value
p_value_residence <- 1 - pchisq(LR_residence, df = df_residence)

# Таблица результатов
LR_residence_table <- data.frame(
  Тест = "LR-тест совместной модели для типов населенного пункта",
  lnL_R = lnL_R_residence,
  lnL_UR = lnL_UR_residence,
  LR_статистика = LR_residence,
  df = df_residence,
  p_value = p_value_residence
)

LR_residence_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")] <-
  round(LR_residence_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")], 4)

LR_residence_table

library(numDeriv)

# 4.7. Асимптотическая дисперсия предельного эффекта age
# в гетероскедастичной пробит-модели

# Оцененные параметры
par_hat <- opt_hetprobit$par

k_beta <- ncol(X)

beta_hat <- par_hat[1:k_beta]
gamma_hat <- par_hat[(k_beta + 1):length(par_hat)]

names(beta_hat) <- colnames(X)
names(gamma_hat) <- colnames(W)

# Ковариационная матрица оценок гетероскедастичной пробит-модели
# Так как optim максимизировал logLik, берем обратный минус Гессиан
cov_hetprobit <- solve(-opt_hetprobit$hessian)

# Характеристики произвольного индивида
age_v <- 22
choice_v <- 1
male_v <- 1

bugs_v <- mean(data$bugs)
price_v <- mean(data$price)

# Стандартизированные значения для уравнения дисперсии
age_s_v <- (age_v - mean(data$age)) / sd(data$age)
bugs_s_v <- (bugs_v - mean(data$bugs)) / sd(data$bugs)
price_s_v <- (price_v - mean(data$price)) / sd(data$price)

# Функция предельного эффекта age
me_age_het_fun <- function(par) {
  
  beta <- par[1:k_beta]
  gamma <- par[(k_beta + 1):length(par)]
  
  names(beta) <- colnames(X)
  names(gamma) <- colnames(W)
  
  # Основное уравнение
  mu <- beta["(Intercept)"] +
    beta["choice"] * choice_v +
    beta["age"] * age_v +
    beta["I(age^2)"] * age_v^2 +
    beta["male"] * male_v +
    beta["choice:age"] * choice_v * age_v
  
  # Уравнение дисперсии
  wg <- gamma["age_s"] * age_s_v +
    gamma["bugs_s"] * bugs_s_v +
    gamma["price_s"] * price_s_v
  
  sigma <- exp(wg)
  
  z <- mu / sigma
  
  # Производная основного уравнения по age
  d_mu_d_age <- beta["age"] +
    2 * beta["I(age^2)"] * age_v +
    beta["choice:age"] * choice_v
  
  # Производная ln(sigma) по исходному age
  d_logsigma_d_age <- gamma["age_s"] / sd(data$age)
  
  # Предельный эффект age на вероятность
  me <- dnorm(z) *
    (d_mu_d_age - mu * d_logsigma_d_age) / sigma
  
  return(as.numeric(me))
}

# Оценка предельного эффекта
me_age_het_hat <- me_age_het_fun(par_hat)

# Градиент предельного эффекта по всем параметрам beta и gamma
grad_me_age_het <- grad(
  func = me_age_het_fun,
  x = par_hat
)

# Асимптотическая дисперсия по дельта-методу
asvar_me_age_het <- t(grad_me_age_het) %*%
  cov_hetprobit %*%
  grad_me_age_het

# Стандартная ошибка
se_me_age_het <- sqrt(asvar_me_age_het)

# Таблица результата
me_age_het_delta_table <- data.frame(
  Переменная = "age",
  Предельный_эффект = me_age_het_hat,
  Асимптотическая_дисперсия = as.numeric(asvar_me_age_het),
  Стандартная_ошибка = as.numeric(se_me_age_het)
)

me_age_het_delta_table[, 2:4] <-
  round(me_age_het_delta_table[, 2:4], 6)

me_age_het_delta_table

#_______________________________________________________________________________


# Часть 5. Логит модель.
#_______________________________________________________________________________

model_logit <- glm(
  satisfied ~ choice + age + I(age^2) + male + choice:age,
  data = data,
  family = binomial(link = "logit")
)

summary(model_logit)

# 5.4. Проверка гипотезы: OR_age = 1.10

library(numDeriv)

# Характеристики индивида
age_v <- 22
choice_v <- 1

# Функция изменения отношения шансов при увеличении age на 1 год
or_age_fun <- function(beta) {
  
  delta_log_odds <- beta["age"] +
    beta["I(age^2)"] * (2 * age_v + 1) +
    beta["choice:age"] * choice_v
  
  OR <- exp(delta_log_odds)
  
  return(as.numeric(OR))
}

# Оценка изменения отношения шансов
OR_age_hat <- or_age_fun(coef(model_logit))

# Градиент функции по коэффициентам логит-модели
grad_OR_age <- grad(
  func = or_age_fun,
  x = coef(model_logit)
)

# Стандартная ошибка по дельта-методу
var_OR_age <- t(grad_OR_age) %*%
  vcov(model_logit) %*%
  grad_OR_age

se_OR_age <- sqrt(var_OR_age)

# Проверяем H0: OR_age = 1.10
OR_H0 <- 1.10

z_OR_age <- (OR_age_hat - OR_H0) / se_OR_age

p_value_OR_age <- 2 * (1 - pnorm(abs(z_OR_age)))

# Таблица результата
OR_age_test_table <- data.frame(
  Переменная = "age",
  Изменение = "age + 1",
  OR_hat = OR_age_hat,
  SE = as.numeric(se_OR_age),
  H0 = OR_H0,
  z_статистика = as.numeric(z_OR_age),
  p_value = as.numeric(p_value_OR_age)
)

OR_age_test_table[, 3:7] <- round(OR_age_test_table[, 3:7], 4)

OR_age_test_table

# 5.4. Проверка гипотезы: OR_age = 1.10

library(numDeriv)

# Характеристики индивида
age_v <- 22
choice_v <- 1

# Функция изменения отношения шансов при увеличении age на 1 год
or_age_fun <- function(beta) {
  
  delta_log_odds <- beta["age"] +
    beta["I(age^2)"] * (2 * age_v + 1) +
    beta["choice:age"] * choice_v
  
  OR <- exp(delta_log_odds)
  
  return(as.numeric(OR))
}

# Оценка изменения отношения шансов
OR_age_hat <- or_age_fun(coef(model_logit))

# Градиент функции по коэффициентам логит-модели
grad_OR_age <- grad(
  func = or_age_fun,
  x = coef(model_logit)
)

# Стандартная ошибка по дельта-методу
var_OR_age <- t(grad_OR_age) %*%
  vcov(model_logit) %*%
  grad_OR_age

se_OR_age <- sqrt(var_OR_age)

# Проверяем H0: OR_age = 1.10
OR_H0 <- 1.10

z_OR_age <- (OR_age_hat - OR_H0) / se_OR_age

p_value_OR_age <- 2 * (1 - pnorm(abs(z_OR_age)))

# Таблица результата
OR_age_test_table <- data.frame(
  Переменная = "age",
  Изменение = "age + 1",
  OR_hat = OR_age_hat,
  SE = as.numeric(se_OR_age),
  H0 = OR_H0,
  z_статистика = as.numeric(z_OR_age),
  p_value = as.numeric(p_value_OR_age)
)

OR_age_test_table[, 3:7] <- round(OR_age_test_table[, 3:7], 4)

OR_age_test_table


############################################################
# Часть 6. Система бинарных уравнений
# 6.1. Оценка системы бинарных уравнений
############################################################

# install.packages("switchSelection")
library(switchSelection)

# Создаем взаимодействие вручную
data$choice_age <- data$choice * data$age

# Уравнение выбора стороны прохождения:
# choice = 1, если игрок выбрал кошек; 0, если собак
choice_formula <- choice ~ age + male + design + cat + dog

# Уравнение удовлетворенности:
# satisfied = 1, если игрок удовлетворен игрой; 0 иначе
satisfied_formula <- satisfied ~ choice + age + I(age^2) + male + choice_age

# Оценка системы бинарных уравнений
system_binary_model <- msel(
  formula = list(
    choice_formula,
    satisfied_formula
  ),
  data = data
)

summary(system_binary_model)



# 6.3. LR-тест необходимости совместного оценивания

# Неограниченная модель: совместная система бинарных уравнений
lnL_UR <- as.numeric(logLik(system_binary_model))

# Ограниченная модель: два отдельных пробита

# Уравнение выбора кошек
model_choice_sep <- glm(
  choice ~ age + male + design + cat + dog,
  data = data,
  family = binomial(link = "probit")
)

# Уравнение удовлетворенности
model_satisfied_sep <- glm(
  satisfied ~ choice + age + I(age^2) + male + choice_age,
  data = data,
  family = binomial(link = "probit")
)

# Логарифм правдоподобия ограниченной модели
lnL_R <- as.numeric(logLik(model_choice_sep)) +
  as.numeric(logLik(model_satisfied_sep))

# LR-статистика
LR_joint <- 2 * (lnL_UR - lnL_R)

# Число ограничений
df_LR <- 1

# p-value
p_value_LR <- 1 - pchisq(LR_joint, df = df_LR)

LR_joint_table <- data.frame(
  Тест = "LR-тест необходимости совместного оценивания",
  lnL_R = lnL_R,
  lnL_UR = lnL_UR,
  LR_статистика = LR_joint,
  df = df_LR,
  p_value = p_value_LR
)

LR_joint_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")] <-
  round(LR_joint_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")], 4)

LR_joint_table




############################################################
# 6.5*. Предельный эффект age на P(satisfied = 1 | choice = 1)
############################################################

library(switchSelection)

# На всякий случай проверяем, что взаимодействие создано корректно
data$choice_age <- data$choice * data$age

# Если модель уже оценена как system_binary_model, переоценивать не обязательно.
# Но если нужно воспроизвести модель с нуля, используем тот же код:

choice_formula <- choice ~ age + male + design + cat + dog

satisfied_formula <- satisfied ~ choice + age + I(age^2) + male + choice_age

system_binary_model <- msel(
  formula = list(
    choice_formula,
    satisfied_formula
  ),
  data = data
)

summary(system_binary_model)


############################################################
# Характеристики произвольного индивида
############################################################

individual_6_5 <- data.frame(
  age = 22,
  male = 1,
  design = 1,
  cat = 1,
  dog = 0,
  choice = 1
)

# Важно: так как choice_age создан вручную,
# его тоже нужно явно задать
individual_6_5$choice_age <- individual_6_5$choice * individual_6_5$age

individual_6_5


############################################################
# Условная вероятность P(satisfied = 1 | choice = 1)
############################################################

p_cond <- predict(
  system_binary_model,
  type = "prob",
  group = c(1, 1),
  newdata = individual_6_5,
  given_ind = 1,
  exogenous = list(choice = 1)
)

p_cond


############################################################
# Предельный эффект age на P(satisfied = 1 | choice = 1)
# Важно: считаем через test_msel(), потому что choice_age
# нужно обновлять вручную при изменении age
############################################################

ME_age_cond_fn <- function(object) {
  
  h <- 0.0001
  
  # Индивид с age + h
  individual_plus <- individual_6_5
  individual_plus$age <- individual_6_5$age + h
  individual_plus$choice <- 1
  individual_plus$choice_age <- individual_plus$choice * individual_plus$age
  
  # Индивид с age - h
  individual_minus <- individual_6_5
  individual_minus$age <- individual_6_5$age - h
  individual_minus$choice <- 1
  individual_minus$choice_age <- individual_minus$choice * individual_minus$age
  
  # P(satisfied = 1 | choice = 1) при age + h
  p_plus <- predict(
    object,
    type = "prob",
    group = c(1, 1),
    newdata = individual_plus,
    given_ind = 1,
    exogenous = list(choice = 1)
  )
  
  # P(satisfied = 1 | choice = 1) при age - h
  p_minus <- predict(
    object,
    type = "prob",
    group = c(1, 1),
    newdata = individual_minus,
    given_ind = 1,
    exogenous = list(choice = 1)
  )
  
  # Центральная численная производная
  ME <- (as.numeric(p_plus) - as.numeric(p_minus)) / (2 * h)
  
  return(ME)
}

# Тест значимости предельного эффекта
me_age_cond_test <- test_msel(
  system_binary_model,
  ME_age_cond_fn
)

summary(me_age_cond_test)


############################################################
# Итоговая таблица
############################################################

me_age_cond_table <- data.frame(
  Переменная = "age",
  Условие = "choice = 1",
  Условная_вероятность = as.numeric(p_cond),
  Предельный_эффект = as.numeric(me_age_cond_test$tbl$val),
  Стандартная_ошибка = as.numeric(me_age_cond_test$tbl$se),
  z_статистика = as.numeric(me_age_cond_test$tbl$stat),
  p_value = as.numeric(me_age_cond_test$tbl$p_value)
)

me_age_cond_table[, 3:7] <- round(me_age_cond_table[, 3:7], 4)

me_age_cond_table


############################################################
# 6.6*. ATE и ATET воздействия choice на satisfied
############################################################

library(switchSelection)

# Убедимся, что взаимодействие задано
data$choice_age <- data$choice * data$age

############################################################
# ATE: средний эффект воздействия для всей выборки
############################################################

ATE_choice_fn <- function(object) {
  
  all_data <- object$data
  
  # Сценарий 1: все игроки проходят за кошек
  data_choice_1 <- all_data
  data_choice_1$choice <- 1
  data_choice_1$choice_age <- data_choice_1$choice * data_choice_1$age
  
  # Сценарий 0: все игроки проходят за собак
  data_choice_0 <- all_data
  data_choice_0$choice <- 0
  data_choice_0$choice_age <- data_choice_0$choice * data_choice_0$age
  
  # Вероятность satisfied = 1 при choice = 1
  p1 <- predict(
    object,
    type = "prob",
    group = c(-1, 1),
    newdata = data_choice_1,
    exogenous = list(
      choice = 1,
      choice_age = data_choice_1$choice_age
    )
  )
  
  # Вероятность satisfied = 1 при choice = 0
  p0 <- predict(
    object,
    type = "prob",
    group = c(-1, 1),
    newdata = data_choice_0,
    exogenous = list(
      choice = 0,
      choice_age = data_choice_0$choice_age
    )
  )
  
  ATE <- mean(as.numeric(p1) - as.numeric(p0))
  
  return(ATE)
}

ATE_choice <- test_msel(
  system_binary_model,
  ATE_choice_fn
)

summary(ATE_choice)


############################################################
# ATET: средний эффект воздействия среди подвергшихся
# то есть среди тех, кто реально выбрал кошек
############################################################

ATET_choice_fn <- function(object) {
  
  treated <- object$data[object$data$choice == 1, ]
  
  # Сценарий 1: реально выбравшие кошек остаются с choice = 1
  treated_choice_1 <- treated
  treated_choice_1$choice <- 1
  treated_choice_1$choice_age <- treated_choice_1$choice * treated_choice_1$age
  
  # Контрфактический сценарий 0: те же игроки как будто выбрали собак
  treated_choice_0 <- treated
  treated_choice_0$choice <- 0
  treated_choice_0$choice_age <- treated_choice_0$choice * treated_choice_0$age
  
  # P(satisfied = 1 | choice = 1) при choice = 1
  p1 <- predict(
    object,
    type = "prob",
    group = c(1, 1),
    given_ind = 1,
    newdata = treated_choice_1,
    exogenous = list(
      choice = 1,
      choice_age = treated_choice_1$choice_age
    )
  )
  
  # P(satisfied = 1 | choice = 1) при контрфактическом choice = 0
  p0 <- predict(
    object,
    type = "prob",
    group = c(1, 1),
    given_ind = 1,
    newdata = treated_choice_0,
    exogenous = list(
      choice = 0,
      choice_age = treated_choice_0$choice_age
    )
  )
  
  ATET <- mean(as.numeric(p1) - as.numeric(p0))
  
  return(ATET)
}

ATET_choice <- test_msel(
  system_binary_model,
  ATET_choice_fn
)

summary(ATET_choice)


############################################################
# Итоговая таблица
############################################################

effects_table <- data.frame(
  Эффект = c("ATE", "ATET"),
  Описание = c(
    "Средний эффект воздействия по всей выборке",
    "Средний эффект воздействия среди выбравших кошек"
  ),
  Оценка = c(
    as.numeric(ATE_choice$tbl$val),
    as.numeric(ATET_choice$tbl$val)
  ),
  Стандартная_ошибка = c(
    as.numeric(ATE_choice$tbl$se),
    as.numeric(ATET_choice$tbl$se)
  ),
  z_статистика = c(
    as.numeric(ATE_choice$tbl$stat),
    as.numeric(ATET_choice$tbl$stat)
  ),
  p_value = c(
    as.numeric(ATE_choice$tbl$p_value),
    as.numeric(ATET_choice$tbl$p_value)
  )
)

effects_table[, 3:6] <- round(effects_table[, 3:6], 4)

effects_table



############################################################
# 6.7**. Система с порядковой переменной game
############################################################

library(switchSelection)

# Взаимодействие из нашей спецификации
data$choice_age <- data$choice * data$age

# Уравнение выбора стороны прохождения
choice_formula_ord <- choice ~ age + male + design + cat + dog

# Уравнение порядковой удовлетворенности game
game_formula_ord <- game ~ choice + age + I(age^2) + male + choice_age

# Оценка системы: choice — бинарное уравнение, game — порядковое уравнение
system_ord_model <- msel(
  formula = list(
    choice_formula_ord,
    game_formula_ord
  ),
  data = data
)

summary(system_ord_model)

############################################################
# ATE 1: эффект choice на вероятность высокой удовлетворенности
# P(game = 2)
############################################################

ATE_high_fn <- function(object) {
  
  all_data <- object$data
  
  # Сценарий 1: все игроки проходят за кошек
  data_choice_1 <- all_data
  data_choice_1$choice <- 1
  data_choice_1$choice_age <- data_choice_1$choice * data_choice_1$age
  
  # Сценарий 0: все игроки проходят за собак
  data_choice_0 <- all_data
  data_choice_0$choice <- 0
  data_choice_0$choice_age <- data_choice_0$choice * data_choice_0$age
  
  # Вероятность game = 2 при choice = 1
  p1 <- predict(
    object,
    type = "prob",
    group = c(-1, 2),
    newdata = data_choice_1,
    exogenous = list(
      choice = 1,
      choice_age = data_choice_1$choice_age
    )
  )
  
  # Вероятность game = 2 при choice = 0
  p0 <- predict(
    object,
    type = "prob",
    group = c(-1, 2),
    newdata = data_choice_0,
    exogenous = list(
      choice = 0,
      choice_age = data_choice_0$choice_age
    )
  )
  
  ATE_high <- mean(as.numeric(p1) - as.numeric(p0))
  
  return(ATE_high)
}

ATE_high <- test_msel(
  system_ord_model,
  ATE_high_fn
)

summary(ATE_high)


############################################################
# ATE 2: эффект choice на вероятность низкой удовлетворенности
# P(game = 0)
############################################################

ATE_low_fn <- function(object) {
  
  all_data <- object$data
  
  # Сценарий 1: все игроки проходят за кошек
  data_choice_1 <- all_data
  data_choice_1$choice <- 1
  data_choice_1$choice_age <- data_choice_1$choice * data_choice_1$age
  
  # Сценарий 0: все игроки проходят за собак
  data_choice_0 <- all_data
  data_choice_0$choice <- 0
  data_choice_0$choice_age <- data_choice_0$choice * data_choice_0$age
  
  # Вероятность game = 0 при choice = 1
  p1 <- predict(
    object,
    type = "prob",
    group = c(-1, 0),
    newdata = data_choice_1,
    exogenous = list(
      choice = 1,
      choice_age = data_choice_1$choice_age
    )
  )
  
  # Вероятность game = 0 при choice = 0
  p0 <- predict(
    object,
    type = "prob",
    group = c(-1, 0),
    newdata = data_choice_0,
    exogenous = list(
      choice = 0,
      choice_age = data_choice_0$choice_age
    )
  )
  
  ATE_low <- mean(as.numeric(p1) - as.numeric(p0))
  
  return(ATE_low)
}

ATE_low <- test_msel(
  system_ord_model,
  ATE_low_fn
)

summary(ATE_low)


############################################################
# Итоговая таблица
############################################################

ATE_ord_table <- data.frame(
  Эффект = c(
    "ATE на P(game = 2)",
    "ATE на P(game = 0)"
  ),
  Интерпретация = c(
    "Эффект прохождения за кошек на вероятность высокой удовлетворенности",
    "Эффект прохождения за кошек на вероятность низкой удовлетворенности"
  ),
  Оценка = c(
    as.numeric(ATE_high$tbl$val),
    as.numeric(ATE_low$tbl$val)
  ),
  Стандартная_ошибка = c(
    as.numeric(ATE_high$tbl$se),
    as.numeric(ATE_low$tbl$se)
  ),
  z_статистика = c(
    as.numeric(ATE_high$tbl$stat),
    as.numeric(ATE_low$tbl$stat)
  ),
  p_value = c(
    as.numeric(ATE_high$tbl$p_value),
    as.numeric(ATE_low$tbl$p_value)
  )
)

ATE_ord_table[, 3:6] <- round(ATE_ord_table[, 3:6], 4)

ATE_ord_table





############################################################
# Часть 7. Сравнение моделей
# 7.1. Какая модель лучше прогнозирует удовлетворенность?
############################################################

# Фактические значения бинарной удовлетворенности
y_true <- data$satisfied

############################################################
# 1. Наивный прогноз
############################################################

majority_class <- as.numeric(names(which.max(table(y_true))))
pred_naive <- rep(majority_class, length(y_true))

accuracy_naive <- mean(pred_naive == y_true)


############################################################
# 2. Линейно-вероятностная модель
############################################################

p_lpm <- predict(lpm)

pred_lpm <- ifelse(p_lpm >= 0.5, 1, 0)

accuracy_lpm <- mean(pred_lpm == y_true)


############################################################
# 3. Пробит-модель
############################################################

p_probit <- predict(model_probit, type = "response")

pred_probit <- ifelse(p_probit >= 0.5, 1, 0)

accuracy_probit <- mean(pred_probit == y_true)


############################################################
# 4. Логит-модель
############################################################

p_logit <- predict(model_logit, type = "response")

pred_logit <- ifelse(p_logit >= 0.5, 1, 0)

accuracy_logit <- mean(pred_logit == y_true)


############################################################
# 5. Гетероскедастичная пробит-модель
############################################################

# Используем уже оцененные параметры opt_hetprobit
par_hat <- opt_hetprobit$par

k_beta <- ncol(X)

beta_hat <- par_hat[1:k_beta]
gamma_hat <- par_hat[(k_beta + 1):length(par_hat)]

# Линейные индексы
xb_het <- as.vector(X %*% beta_hat)
wg_het <- as.vector(W %*% gamma_hat)

# Вероятности гетероскедастичной пробит-модели
p_hetprobit <- pnorm(xb_het / exp(wg_het))

pred_hetprobit <- ifelse(p_hetprobit >= 0.5, 1, 0)

accuracy_hetprobit <- mean(pred_hetprobit == y_true)


############################################################
# 6. Система бинарных уравнений
############################################################

# Вероятность satisfied = 1 без условия на choice
p_system_binary <- predict(
  system_binary_model,
  type = "prob",
  group = c(-1, 1)
)

pred_system_binary <- ifelse(as.numeric(p_system_binary) >= 0.5, 1, 0)

accuracy_system_binary <- mean(pred_system_binary == y_true)


############################################################
# 7. Система с порядковой переменной game
# P(satisfied = 1) = P(game = 1) + P(game = 2)
############################################################

p_game_1 <- predict(
  system_ord_model,
  type = "prob",
  group = c(-1, 1)
)

p_game_2 <- predict(
  system_ord_model,
  type = "prob",
  group = c(-1, 2)
)

p_system_ord_satisfied <- as.numeric(p_game_1) + as.numeric(p_game_2)

pred_system_ord <- ifelse(p_system_ord_satisfied >= 0.5, 1, 0)

accuracy_system_ord <- mean(pred_system_ord == y_true)


############################################################
# Итоговая таблица
############################################################

accuracy_comparison_table <- data.frame(
  Модель = c(
    "Наивный прогноз",
    "Линейно-вероятностная модель",
    "Пробит-модель",
    "Логит-модель",
    "Гетероскедастичная пробит-модель",
    "Система бинарных уравнений",
    "Система с порядковой переменной game"
  ),
  Доля_верных_прогнозов = c(
    accuracy_naive,
    accuracy_lpm,
    accuracy_probit,
    accuracy_logit,
    accuracy_hetprobit,
    accuracy_system_binary,
    accuracy_system_ord
  )
)

accuracy_comparison_table$Доля_верных_прогнозов <- round(
  accuracy_comparison_table$Доля_верных_прогнозов,
  4
)

accuracy_comparison_table <- accuracy_comparison_table[
  order(-accuracy_comparison_table$Доля_верных_прогнозов),
]

accuracy_comparison_table


# ПО F1

############################################################
# Часть 7. Сравнение моделей по F1-score
# с подбором оптимального порога
############################################################

# Фактические значения
y_true <- data$satisfied

############################################################
# Функции для расчета F1 и подбора порога
############################################################

f1_metrics <- function(y_true, y_pred) {
  
  TP <- sum(y_true == 1 & y_pred == 1)
  FP <- sum(y_true == 0 & y_pred == 1)
  FN <- sum(y_true == 1 & y_pred == 0)
  TN <- sum(y_true == 0 & y_pred == 0)
  
  precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  f1 <- ifelse((precision + recall) == 0, 0,
               2 * precision * recall / (precision + recall))
  
  return(c(
    TP = TP,
    FP = FP,
    FN = FN,
    TN = TN,
    Precision = precision,
    Recall = recall,
    F1 = f1
  ))
}

find_best_threshold <- function(y_true, p_hat) {
  
  thresholds <- seq(0.01, 0.99, by = 0.01)
  
  result <- data.frame(
    threshold = thresholds,
    Precision = NA,
    Recall = NA,
    F1 = NA
  )
  
  for (i in seq_along(thresholds)) {
    
    pred <- ifelse(p_hat >= thresholds[i], 1, 0)
    metrics <- f1_metrics(y_true, pred)
    
    result$Precision[i] <- metrics["Precision"]
    result$Recall[i] <- metrics["Recall"]
    result$F1[i] <- metrics["F1"]
  }
  
  best_row <- result[which.max(result$F1), ]
  
  return(best_row)
}

############################################################
# Предсказанные вероятности по всем моделям
############################################################

# 1. Наивный прогноз
# Для наивного прогноза вероятность равна доле самого частого класса
p_naive <- rep(mean(y_true), length(y_true))

# 2. Линейно-вероятностная модель
p_lpm <- predict(lpm)

# На всякий случай ограничим прогнозы ЛВМ интервалом [0; 1]
p_lpm <- pmin(pmax(p_lpm, 0), 1)

# 3. Пробит-модель
p_probit <- predict(model_probit, type = "response")

# 4. Логит-модель
p_logit <- predict(model_logit, type = "response")

# 5. Гетероскедастичная пробит-модель
par_hat <- opt_hetprobit$par

k_beta <- ncol(X)

beta_hat <- par_hat[1:k_beta]
gamma_hat <- par_hat[(k_beta + 1):length(par_hat)]

xb_het <- as.vector(X %*% beta_hat)
wg_het <- as.vector(W %*% gamma_hat)

p_hetprobit <- pnorm(xb_het / exp(wg_het))

# 6. Система бинарных уравнений
p_system_binary <- predict(
  system_binary_model,
  type = "prob",
  group = c(-1, 1)
)

p_system_binary <- as.numeric(p_system_binary)

# 7. Система с порядковой переменной game
# P(satisfied = 1) = P(game = 1) + P(game = 2)

p_game_1 <- predict(
  system_ord_model,
  type = "prob",
  group = c(-1, 1)
)

p_game_2 <- predict(
  system_ord_model,
  type = "prob",
  group = c(-1, 2)
)

p_system_ord_satisfied <- as.numeric(p_game_1) + as.numeric(p_game_2)

############################################################
# Подбор оптимального порога для каждой модели
############################################################

best_naive <- find_best_threshold(y_true, p_naive)
best_lpm <- find_best_threshold(y_true, p_lpm)
best_probit <- find_best_threshold(y_true, p_probit)
best_logit <- find_best_threshold(y_true, p_logit)
best_hetprobit <- find_best_threshold(y_true, p_hetprobit)
best_system_binary <- find_best_threshold(y_true, p_system_binary)
best_system_ord <- find_best_threshold(y_true, p_system_ord_satisfied)

############################################################
# Итоговая таблица
############################################################

f1_best_threshold_table <- data.frame(
  Модель = c(
    "Наивный прогноз",
    "Линейно-вероятностная модель",
    "Пробит-модель",
    "Логит-модель",
    "Гетероскедастичная пробит-модель",
    "Система бинарных уравнений",
    "Система с порядковой переменной game"
  ),
  Оптимальный_порог = c(
    best_naive$threshold,
    best_lpm$threshold,
    best_probit$threshold,
    best_logit$threshold,
    best_hetprobit$threshold,
    best_system_binary$threshold,
    best_system_ord$threshold
  ),
  Precision = c(
    best_naive$Precision,
    best_lpm$Precision,
    best_probit$Precision,
    best_logit$Precision,
    best_hetprobit$Precision,
    best_system_binary$Precision,
    best_system_ord$Precision
  ),
  Recall = c(
    best_naive$Recall,
    best_lpm$Recall,
    best_probit$Recall,
    best_logit$Recall,
    best_hetprobit$Recall,
    best_system_binary$Recall,
    best_system_ord$Recall
  ),
  F1_score = c(
    best_naive$F1,
    best_lpm$F1,
    best_probit$F1,
    best_logit$F1,
    best_hetprobit$F1,
    best_system_binary$F1,
    best_system_ord$F1
  )
)

f1_best_threshold_table[, 2:5] <- round(f1_best_threshold_table[, 2:5], 4)

f1_best_threshold_table <- f1_best_threshold_table[
  order(-f1_best_threshold_table$F1_score),
]

f1_best_threshold_table


############################################################
# 7.2. Сравнение моделей по информационным критериям
# с включением ЛВМ
############################################################

n <- nrow(data)

############################################################
# 1. Линейно-вероятностная модель
############################################################

lnL_lpm <- as.numeric(logLik(lpm))
k_lpm <- attr(logLik(lpm), "df")

AIC_lpm <- AIC(lpm)
BIC_lpm <- BIC(lpm)


############################################################
# 2. Пробит-модель
############################################################

lnL_probit <- as.numeric(logLik(model_probit))
k_probit <- attr(logLik(model_probit), "df")

AIC_probit <- AIC(model_probit)
BIC_probit <- BIC(model_probit)


############################################################
# 3. Логит-модель
############################################################

lnL_logit <- as.numeric(logLik(model_logit))
k_logit <- attr(logLik(model_logit), "df")

AIC_logit <- AIC(model_logit)
BIC_logit <- BIC(model_logit)


############################################################
# 4. Гетероскедастичная пробит-модель
############################################################

lnL_hetprobit <- opt_hetprobit$value
k_hetprobit <- length(opt_hetprobit$par)

AIC_hetprobit <- -2 * lnL_hetprobit + 2 * k_hetprobit
BIC_hetprobit <- -2 * lnL_hetprobit + log(n) * k_hetprobit


############################################################
# 5. Система бинарных уравнений
############################################################

lnL_system_binary <- as.numeric(logLik(system_binary_model))

# Число параметров берём из AIC:
# AIC = -2 lnL + 2k => k = (AIC + 2lnL) / 2
AIC_system_binary <- AIC(system_binary_model)
k_system_binary <- (AIC_system_binary + 2 * lnL_system_binary) / 2
BIC_system_binary <- -2 * lnL_system_binary + log(n) * k_system_binary


############################################################
# 6. Система с порядковой переменной game
############################################################

lnL_system_ord <- as.numeric(logLik(system_ord_model))

AIC_system_ord <- AIC(system_ord_model)
k_system_ord <- (AIC_system_ord + 2 * lnL_system_ord) / 2
BIC_system_ord <- -2 * lnL_system_ord + log(n) * k_system_ord


############################################################
# Итоговая таблица
############################################################

ic_table_all <- data.frame(
  Модель = c(
    "Линейно-вероятностная модель",
    "Пробит-модель",
    "Логит-модель",
    "Гетероскедастичная пробит-модель",
    "Система бинарных уравнений",
    "Система с порядковой переменной game"
  ),
  LogLik = c(
    lnL_lpm,
    lnL_probit,
    lnL_logit,
    lnL_hetprobit,
    lnL_system_binary,
    lnL_system_ord
  ),
  Число_параметров = c(
    k_lpm,
    k_probit,
    k_logit,
    k_hetprobit,
    k_system_binary,
    k_system_ord
  ),
  AIC = c(
    AIC_lpm,
    AIC_probit,
    AIC_logit,
    AIC_hetprobit,
    AIC_system_binary,
    AIC_system_ord
  ),
  BIC = c(
    BIC_lpm,
    BIC_probit,
    BIC_logit,
    BIC_hetprobit,
    BIC_system_binary,
    BIC_system_ord
  )
)

ic_table_all[, 2:5] <- round(ic_table_all[, 2:5], 4)

ic_table_all <- ic_table_all[order(ic_table_all$AIC), ]

ic_table_all


############################################################
# 7.3**. Выбор копулы и маржинальных распределений
#        по AIC и вневыборочному прогнозу
############################################################

install.packages("gsl", type = "win.binary")
install.packages("GJRM")

library(gsl)
library(GJRM)

############################################################
# Подготовка данных
############################################################

# Взаимодействие из нашей спецификации
data$choice_age <- data$choice * data$age

# Уравнение удовлетворенности
satisfied_formula_73 <- satisfied ~ choice + age + I(age^2) + male + choice_age

# Уравнение выбора кошек
choice_formula_73 <- choice ~ age + male + design + cat + dog

############################################################
# Train-test split: 70% train, 30% test
############################################################

set.seed(123)

n <- nrow(data)

idx_train <- sample(seq_len(n), size = floor(0.7 * n))

data_train <- data[idx_train, ]
data_test <- data[-idx_train, ]

############################################################
# Сетка моделей:
# 5 копул x 2 комбинации маржинальных распределений
############################################################

copulas <- c("N", "F", "G0", "C0", "J0")

copula_names <- c(
  "Gaussian",
  "Frank",
  "Gumbel",
  "Clayton",
  "Joe"
)

margins_list <- list(
  c("probit", "probit"),
  c("logit", "logit")
)

margins_names <- c(
  "probit-probit",
  "logit-logit"
)

############################################################
# Функция для расчета F1-score
############################################################

f1_score_fun <- function(y_true, y_pred) {
  
  TP <- sum(y_true == 1 & y_pred == 1)
  FP <- sum(y_true == 0 & y_pred == 1)
  FN <- sum(y_true == 1 & y_pred == 0)
  
  precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  
  f1 <- ifelse(
    (precision + recall) == 0,
    0,
    2 * precision * recall / (precision + recall)
  )
  
  return(f1)
}

############################################################
# Каркас результатов
############################################################

res_73 <- data.frame(
  Copula = character(),
  Margins = character(),
  AIC_train = numeric(),
  Accuracy_test = numeric(),
  F1_test = numeric(),
  stringsAsFactors = FALSE
)

############################################################
# Цикл по копулам и маржинальным распределениям
############################################################

for (m_idx in seq_along(margins_list)) {
  
  for (c_idx in seq_along(copulas)) {
    
    cat(
      "Fitting:",
      copula_names[c_idx],
      "+",
      margins_names[m_idx],
      "\n"
    )
    
    fit <- try(
      gjrm(
        formula = list(
          satisfied_formula_73,
          choice_formula_73
        ),
        data = data_train,
        model = "B",
        margins = margins_list[[m_idx]],
        copula = copulas[c_idx]
      ),
      silent = TRUE
    )
    
    if (inherits(fit, "try-error")) {
      
      res_73 <- rbind(
        res_73,
        data.frame(
          Copula = copula_names[c_idx],
          Margins = margins_names[m_idx],
          AIC_train = NA,
          Accuracy_test = NA,
          F1_test = NA
        )
      )
      
      next
    }
    
    ########################################################
    # AIC на тренировочной выборке
    ########################################################
    
    aic_val <- AIC(fit)
    
    ########################################################
    # Прогноз на тестовой выборке
    ########################################################
    
    eta1_test <- predict(
      fit,
      eq = 1,
      newdata = data_test
    )
    
    # Преобразуем линейный индекс первого уравнения
    # в вероятность satisfied = 1 в зависимости от margin
    p_sat_test <- switch(
      margins_list[[m_idx]][1],
      "probit" = pnorm(eta1_test),
      "logit" = plogis(eta1_test),
      "cloglog" = 1 - exp(-exp(eta1_test)),
      "cauchit" = pcauchy(eta1_test)
    )
    
    # Классификация по стандартному порогу 0.5
    y_hat_test <- as.numeric(p_sat_test >= 0.5)
    
    accuracy_test <- mean(y_hat_test == data_test$satisfied)
    
    f1_test <- f1_score_fun(
      y_true = data_test$satisfied,
      y_pred = y_hat_test
    )
    
    ########################################################
    # Сохраняем результат
    ########################################################
    
    res_73 <- rbind(
      res_73,
      data.frame(
        Copula = copula_names[c_idx],
        Margins = margins_names[m_idx],
        AIC_train = aic_val,
        Accuracy_test = accuracy_test,
        F1_test = f1_test
      )
    )
  }
}

############################################################
# Сводные таблицы
############################################################

res_73$AIC_train <- round(res_73$AIC_train, 4)
res_73$Accuracy_test <- round(res_73$Accuracy_test, 4)
res_73$F1_test <- round(res_73$F1_test, 4)

# Таблица по AIC
res_by_aic_73 <- res_73[order(res_73$AIC_train), ]

# Таблица по вневыборочной accuracy
res_by_acc_73 <- res_73[order(-res_73$Accuracy_test), ]

# Таблица по вневыборочному F1
res_by_f1_73 <- res_73[order(-res_73$F1_test), ]

cat("\n=== Модели по AIC на train ===\n")
print(res_by_aic_73, row.names = FALSE)

cat("\n=== Модели по Accuracy на test ===\n")
print(res_by_acc_73, row.names = FALSE)

cat("\n=== Модели по F1 на test ===\n")
print(res_by_f1_73, row.names = FALSE)

############################################################
# Лучшие модели
############################################################

best_aic <- res_by_aic_73[1, ]
best_acc <- res_by_acc_73[1, ]
best_f1 <- res_by_f1_73[1, ]

cat("\n--- Лучшие модели ---\n")

cat(
  "По AIC:",
  best_aic$Copula,
  "+",
  best_aic$Margins,
  "| AIC =",
  best_aic$AIC_train,
  "\n"
)

cat(
  "По Accuracy:",
  best_acc$Copula,
  "+",
  best_acc$Margins,
  "| Accuracy =",
  best_acc$Accuracy_test,
  "\n"
)

cat(
  "По F1:",
  best_f1$Copula,
  "+",
  best_f1$Margins,
  "| F1 =",
  best_f1$F1_test,
  "\n"
)

cat(
  "Совпадают ли лучшие модели по AIC и Accuracy? ",
  best_aic$Copula == best_acc$Copula &&
    best_aic$Margins == best_acc$Margins,
  "\n"
)

cat(
  "Совпадают ли лучшие модели по AIC и F1? ",
  best_aic$Copula == best_f1$Copula &&
    best_aic$Margins == best_f1$Margins,
  "\n"
)



############################################################
# Часть 8. Модель бинарного выбора с ошибками Стьюдента
# 8.2. Симуляция процесса генерации данных
############################################################

set.seed(123)

# Объем выборки
n <- 5000

# Число степеней свободы:
# Владимир = 8 букв, Смирнов = 7 букв
df_t <- 15

############################################################
# Симулируем независимые переменные
############################################################

# Доход пользователя, тыс. рублей
income <- rlnorm(
  n = n,
  meanlog = log(80),
  sdlog = 0.5
)

# Количество часов использования сервиса в неделю
hours <- rpois(
  n = n,
  lambda = 8
)

# Стоимость подписки, рублей
price <- rnorm(
  n = n,
  mean = 700,
  sd = 150
)

# Ограничим цену снизу, чтобы не было отрицательных значений
price <- pmax(price, 100)

############################################################
# Задаем истинные параметры модели
############################################################

beta_0 <- -1.5
beta_income <- 0.010
beta_hours <- 0.120
beta_price <- -0.002

############################################################
# Симулируем случайные ошибки из распределения Стьюдента
############################################################

epsilon <- rt(
  n = n,
  df = df_t
)

############################################################
# Формируем латентную переменную
############################################################

subscribe_star <- beta_0 +
  beta_income * income +
  beta_hours * hours +
  beta_price * price +
  epsilon

############################################################
# Формируем бинарную зависимую переменную
############################################################

subscribe <- ifelse(subscribe_star > 0, 1, 0)

############################################################
# Собираем данные в датафрейм
############################################################

sim_data <- data.frame(
  subscribe = subscribe,
  income = income,
  hours = hours,
  price = price
)

############################################################
# Проверяем результат симуляции
############################################################

summary(sim_data)

table(sim_data$subscribe)
prop.table






















############################################################
# 8.3. Оценка бинарной модели с ошибками Стьюдента
############################################################

# Число степеней свободы
df_t <- 15

# Матрица регрессоров
X_t <- model.matrix(
  subscribe ~ income + hours + price,
  data = sim_data
)

# Зависимая переменная
y_t <- sim_data$subscribe

# Логарифм функции правдоподобия
lnL_t_binary <- function(beta, X, y, df_t) {
  
  xb <- as.vector(X %*% beta)
  
  # F_t(x beta)
  p <- pt(xb, df = df_t)
  
  # защита от log(0)
  eps <- 1e-10
  p <- pmin(pmax(p, eps), 1 - eps)
  
  lnL <- sum(y * log(p) + (1 - y) * log(1 - p))
  
  return(lnL)
}

# Начальные значения можно взять из пробит-модели
start_probit <- coef(
  glm(
    subscribe ~ income + hours + price,
    data = sim_data,
    family = binomial(link = "probit")
  )
)

# Оценивание ММП
opt_t_binary <- optim(
  par = start_probit,
  fn = lnL_t_binary,
  X = X_t,
  y = y_t,
  df_t = df_t,
  method = "BFGS",
  hessian = TRUE,
  control = list(
    fnscale = -1,
    maxit = 10000,
    reltol = 1e-10
  )
)

# Оценки коэффициентов
coef_t <- opt_t_binary$par

# Ковариационная матрица оценок
cov_t <- solve(-opt_t_binary$hessian)

# Стандартные ошибки
se_t <- sqrt(diag(cov_t))

# z-статистики
z_t <- coef_t / se_t

# p-value
p_value_t <- 2 * (1 - pnorm(abs(z_t)))

# Итоговая таблица
t_binary_table <- data.frame(
  Переменная = names(coef_t),
  Оценка = coef_t,
  Стандартная_ошибка = se_t,
  z_статистика = z_t,
  p_value = p_value_t
)

t_binary_table[, 2:5] <- round(t_binary_table[, 2:5], 4)

t_binary_table




############################################################
# 8.4. Функция для расчета предельного эффекта
############################################################

# Функция предельного эффекта для произвольной непрерывной переменной
me_t_binary <- function(beta, newdata, variable, df_t = 15) {
  
  # Матрица регрессоров для выбранного индивида
  X_new <- model.matrix(
    ~ income + hours + price,
    data = newdata
  )
  
  # Линейный индекс x beta
  xb <- as.numeric(X_new %*% beta)
  
  # Плотность распределения Стьюдента в точке x beta
  density_t <- dt(xb, df = df_t)
  
  # Предельный эффект
  me <- density_t * beta[variable]
  
  return(as.numeric(me))
}

############################################################
# Характеристики произвольного индивида
############################################################

individual_t <- data.frame(
  income = 90,
  hours = 8,
  price = 700
)

individual_t

############################################################
# Предельный эффект переменной hours
############################################################

me_hours_t <- me_t_binary(
  beta = coef_t,
  newdata = individual_t,
  variable = "hours",
  df_t = 15
)

me_hours_t

me_hours_t_table <- data.frame(
  Переменная = "hours",
  income = individual_t$income,
  hours = individual_t$hours,
  price = individual_t$price,
  Предельный_эффект = me_hours_t
)

me_hours_t_table$Предельный_эффект <- round(
  me_hours_t_table$Предельный_эффект,
  4
)

me_hours_t_table








############################################################
# 8.5. Гетероскедастичная t-модель бинарного выбора
# Вариант 2: LR-тест гомоскедастичности
############################################################

set.seed(123)

# Объем выборки
n <- 5000

# df = Владимир (8) + Смирнов (7)
df_t <- 15

############################################################
# 1. Симуляция данных с гетероскедастичной ошибкой
############################################################

# Основные переменные
income <- rlnorm(n, meanlog = log(80), sdlog = 0.5)
hours <- rpois(n, lambda = 8)
price <- rnorm(n, mean = 700, sd = 150)
price <- pmax(price, 100)

# Дополнительная переменная, которая НЕ входит в основное уравнение,
# но влияет на дисперсию ошибки
student <- rbinom(n, size = 1, prob = 0.35)

# Истинные параметры основного уравнения
beta_0 <- -1.5
beta_income <- 0.010
beta_hours <- 0.120
beta_price <- -0.002

# Уравнение масштаба ошибки:
# sigma_i = exp(gamma_1 * hours_i + gamma_2 * student_i)
# hours входит и в основное уравнение, и в уравнение дисперсии,
# student входит только в уравнение дисперсии
hours_s <- as.numeric(scale(hours))

gamma_hours <- 0.15
gamma_student <- 0.40

sigma_i <- exp(gamma_hours * hours_s + gamma_student * student)

# Гетероскедастичная ошибка
epsilon <- sigma_i * rt(n, df = df_t)

# Латентная переменная
subscribe_star <- beta_0 +
  beta_income * income +
  beta_hours * hours +
  beta_price * price +
  epsilon

# Наблюдаемая бинарная переменная
subscribe <- ifelse(subscribe_star > 0, 1, 0)

# Данные
sim_data_het <- data.frame(
  subscribe = subscribe,
  income = income,
  hours = hours,
  price = price,
  student = student,
  hours_s = hours_s
)

table(sim_data_het$subscribe)
prop.table(table(sim_data_het$subscribe))


############################################################
# 2. Функции логарифма правдоподобия
############################################################

# Ограниченная модель: гомоскедастичная t-модель
lnL_t_homosk <- function(par, X, y, df_t) {
  
  beta <- par
  
  xb <- as.vector(X %*% beta)
  
  p <- pt(xb, df = df_t)
  
  eps <- 1e-10
  p <- pmin(pmax(p, eps), 1 - eps)
  
  lnL <- sum(y * log(p) + (1 - y) * log(1 - p))
  
  return(lnL)
}

# Неограниченная модель: гетероскедастичная t-модель
lnL_t_hetero <- function(par, X, W, y, df_t) {
  
  k_beta <- ncol(X)
  
  beta <- par[1:k_beta]
  gamma <- par[(k_beta + 1):length(par)]
  
  xb <- as.vector(X %*% beta)
  wg <- as.vector(W %*% gamma)
  
  sigma <- exp(wg)
  
  p <- pt(xb / sigma, df = df_t)
  
  eps <- 1e-10
  p <- pmin(pmax(p, eps), 1 - eps)
  
  lnL <- sum(y * log(p) + (1 - y) * log(1 - p))
  
  return(lnL)
}


############################################################
# 3. Оценивание моделей
############################################################

# Основное уравнение
X <- model.matrix(
  subscribe ~ income + hours + price,
  data = sim_data_het
)

# Уравнение дисперсии: hours_s и student
# student не входит в основное уравнение
W <- model.matrix(
  ~ hours_s + student,
  data = sim_data_het
)[, -1]

y <- sim_data_het$subscribe

# Начальные значения beta из обычной t-модели, можно взять из пробита
start_beta <- coef(
  glm(
    subscribe ~ income + hours + price,
    data = sim_data_het,
    family = binomial(link = "probit")
  )
)

# Ограниченная модель
opt_R <- optim(
  par = start_beta,
  fn = lnL_t_homosk,
  X = X,
  y = y,
  df_t = df_t,
  method = "BFGS",
  hessian = TRUE,
  control = list(
    fnscale = -1,
    maxit = 10000,
    reltol = 1e-10
  )
)

lnL_R <- opt_R$value

# Неограниченная модель
start_gamma <- rep(0, ncol(W))
start_par_UR <- c(start_beta, start_gamma)

opt_UR <- optim(
  par = start_par_UR,
  fn = lnL_t_hetero,
  X = X,
  W = W,
  y = y,
  df_t = df_t,
  method = "BFGS",
  hessian = TRUE,
  control = list(
    fnscale = -1,
    maxit = 10000,
    reltol = 1e-10
  )
)

lnL_UR <- opt_UR$value


############################################################
# 4. LR-тест гомоскедастичности
############################################################

LR_hetero_t <- 2 * (lnL_UR - lnL_R)

# Проверяем два ограничения:
# gamma_hours = 0 и gamma_student = 0
df_LR <- ncol(W)

p_value_LR <- 1 - pchisq(LR_hetero_t, df = df_LR)

LR_t_hetero_table <- data.frame(
  Тест = "LR-тест гомоскедастичности t-модели",
  lnL_R = lnL_R,
  lnL_UR = lnL_UR,
  LR_статистика = LR_hetero_t,
  df = df_LR,
  p_value = p_value_LR
)

LR_t_hetero_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")] <-
  round(LR_t_hetero_table[, c("lnL_R", "lnL_UR", "LR_статистика", "p_value")], 4)

LR_t_hetero_table


############################################################
# 5. Таблица оценок гетероскедастичной модели
############################################################

par_UR <- opt_UR$par

k_beta <- ncol(X)

beta_hat <- par_UR[1:k_beta]
gamma_hat <- par_UR[(k_beta + 1):length(par_UR)]

names(beta_hat) <- colnames(X)
names(gamma_hat) <- colnames(W)

hetero_t_coef_table <- data.frame(
  Уравнение = c(
    rep("Основное уравнение", length(beta_hat)),
    rep("Уравнение дисперсии", length(gamma_hat))
  ),
  Переменная = c(names(beta_hat), names(gamma_hat)),
  Оценка = c(beta_hat, gamma_hat)
)

hetero_t_coef_table$Оценка <- round(hetero_t_coef_table$Оценка, 4)

hetero_t_coef_table


