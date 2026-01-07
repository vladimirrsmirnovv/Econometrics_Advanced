#install the AER package (once) - "applied econometrics with R"
install.packages("AER")
library(AER)

#load the data from the set
data(CASchools)

#Единица измерения - школа. У каждой школы свой индивидуальный идентификатор.
#есть категориальная переменная district, округ
#количество студентов и учителей - количественные переменные

#Рассказывает, что такое CASchools (датафрейм)
class(CASchools)
#Показывает переменные (?)
head(CASchools)

#compute STR (student to teacher ratio) and append it to CASchools
CASchools$STR = CASchools$students/CASchools$teachers
#compute TestScore and append it to CASchools
CASchools$score = (CASchools$read + CASchools$math)/2

#построение диаграммы рассеяния (?)
plot(score ~ STR,
     data = CASchools,
     main = "Scatterplot of Test Score and STR",
     xlab = "STR(X)",
     ylab = "Test Score (Y)")

#Посчитаем коэффициент корреляции
cor(CASchools$STR, CASchools$score)

#Оценим модель
#lm(Y ∼ X, data = D) — построение парной линейной регрессии для изучения влияния
#объясняющей переменной X на зависимую переменную Y в наборе D; STR - объясняющая переменная

linear_model = lm(score ~ STR, data = CASchools)
linear_model2 = lm(math ~ STR, data = CASchools)
linear_model3 = lm(read ~ STR, data = CASchools)

linear_model
linear_model2 
linear_model3

lm(score ~ STR, data = CASchools)

#add the regression line - истинную зависимость
abline (linear_model)

#score = 699 - 2.28*STR - модель, описывающая связь между числом учеников, приходящихся на 1 учителя
#Интерпртетация: увеличение числа учеников на одного, приходящихся на 1 учителя, приходящегося на 1 ученика,
#в среднем связано с падением среднего результата на двух тестах на 2.28 балла.
#В среднем 1 ученик, приходящийся на одного учителя, в среднем (не всегда так) связан с ...

#18.09.2023
# print the standard output of the estimated lm object to the console
linear_model
mod_summary = summary(linear_model)
mod_summary


#t value = -4.571 < 1.96
#p-value 2.78e-06 < 0.05


#Посчитаем всё вручную:

# compute R^2 manually
SSR <- sum(mod_summary$residuals^2)
TSS <- sum((CASchools$score - mean(CASchools$score))^2)
R2 <- 1 - SSR/TSS

# print the value to the console
R2
#> [1] 0.05124009

# compute SER manually
n <- nrow(CASchools)
SER <- sqrt(SSR / (n-2))

# print the value to the console
SER
#> [1] 18.58097

# plot the data
plot(score ~ STR, 
     data = CASchools,
     main = "Scatterplot of Test Score and STR", 
     xlab = "STR (X)",
     ylab = "Test Score (Y)",
     xlim = c(10, 30),
     ylim = c(600, 720))

# add the regression line
abline(linear_model) 


#Откуда мы знаем про процентили?

# compute sample averages of STR and score
avg_STR <- mean(CASchools$STR) 
avg_score <- mean(CASchools$score)

# compute sample standard deviations of STR and score
sd_STR <- sd(CASchools$STR) 
sd_score <- sd(CASchools$score)
# set up a vector of percentiles and compute the quantiles 
quantiles <- c(0.10, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
quant_STR <- quantile(CASchools$STR, quantiles)
quant_score <- quantile(CASchools$score, quantiles)
quant_STR
quant_score

# gather everything in a data.frame 
DistributionSummary <- data.frame(Average = c(avg_STR, avg_score), 
                                  StandardDeviation = c(sd_STR, sd_score), 
                                  quantile = rbind(quant_STR, quant_score))

# print the summary to the console
DistributionSummary


#Симуляция

# simulate data
N <- 100000
X <- runif(N, min = 0, max = 20)
u <- rnorm(N, sd = 10)

# population regression
Y <- -2 + 3.5 * X + u
population <- data.frame(X, Y)

# set sample size
n <- 100

# compute the variance of beta_hat_0
H_i <- 1 - mean(X) / mean(X^2) * X
var_b0 <- var(H_i * u) / (n * mean(H_i^2)^2 )

# compute the variance of hat_beta_1
var_b1 <- var( ( X - mean(X) ) * u ) / (n * var(X)^2)

# print variances to the console
var_b0
#> [1] 4.045066
var_b1
#> [1] 0.03018694
# set repetitions and sample size
n <- 100
reps <- 10000

# initialize the matrix of outcomes
fit <- matrix(ncol = 2, nrow = reps)

# loop sampling and estimation of the coefficients
for (i in 1:reps){
  
  sample <- population[sample(1:N, n), ]
  fit[i, ] <- lm(Y ~ X, data = sample)$coefficients
  
}

# compute variance estimates using outcomes
var(fit[, 1])
#> [1] 4.186832
var(fit[, 2])
#> [1] 0.03096199
# divide plotting area as 1-by-2 array
par(mfrow = c(1, 2))

# plot histograms of beta_0 estimates
hist(fit[, 1],
     cex.main = 0.8,
     main = bquote(The ~ Distribution  ~ of ~ 10000 ~ beta[0] ~ Estimates), 
     xlab = bquote(hat(beta)[0]), 
     freq = F)

# add true distribution to plot
curve(dnorm(x, 
            -2, 
            sqrt(var_b0)), 
      add = T, 
      col = "darkred")

# plot histograms of beta_hat_1 
hist(fit[, 2],
     cex.main = 0.8,
     main = bquote(The ~ Distribution  ~ of ~ 10000 ~ beta[1] ~ Estimates), 
     xlab = bquote(hat(beta)[1]), 
     freq = F)

# add true distribution to plot
curve(dnorm(x, 
            3.5, 
            sqrt(var_b1)), 
      add = T, 
      col = "darkred")



#???

confint(linear_model)
#Intercept = для B0
#STR = для B1

# create the dummy variable as defined above
CASchools$D = CASchools$STR < 20
#Compute the average score when D = 1 (Low STR)
mean.score.for.D.1 = mean(CASchools$score[CASchools$D == TRUE])

#compute the average score when D=0 (high STR)
mean.score.for.D.0 = mean(CASchools$score[CASchools$D == FALSE])

#1. Что является базовой категорией
# По сравнению с базовой категорией (школой, где приходится 20 учеников и более) 
#в школе, где приходится 20 учеников и менее, результаты тестов в среднем на 7.169 меньше.





















