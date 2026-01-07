#ПРИМЕР
#Давайте опять представим, что мы знаем истинную зависимость
# Is your company in tech? Let's say 30% of firms are
df <- tibble(tech = sample(c(0,1),500,replace=T,prob=c(.7,.3))) %>%
  #Tech firms on average spend $3mil more defending IP lawsuits
  mutate(IP.spend = 3*tech+runif(500,min=0,max=4)) %>%
  #Tech firms also have higher profits. But IP lawsuits lower profits
  mutate(log.profit = 2*tech - .3*IP.spend + rnorm(500,mean=2))

#ЧТО НАМ ГОВОРЯТ ДАННЫЕ О ЗАВИСИМОСТИ?
# Now let's check for how profit and IP.spend are correlated!
cor(df$log.profit,df$IP.spend)
## [1] 0.1368556
# Or with regression
lm<-lm(df$log.profit ~ df$IP.spend)
lm

#Контроль с регрессией: the same idea
m1 <- lm(log.profit ~ IP.spend, data = df)
m2 <- lm(log.profit ~ IP.spend + tech, data = df)
msummary(list(m1,m2), stars = TRUE, gof_omit = 'AIC|BIC|Lik|Adj.|F')

#ПРОВЕРИМ НА ДАННЫХ

# load the AER package
library(AER)
# load the data set
data(CASchools)   

# define variables
CASchools$STR <- CASchools$students/CASchools$teachers       
CASchools$score <- (CASchools$read + CASchools$math)/2

# compute correlations
cor(CASchools$STR, CASchools$score)
#> [1] -0.2263627
cor(CASchools$STR, CASchools$english)
#> [1] 0.1876424

# estimate both regression models
mod <- lm(score ~ STR, data = CASchools) 
mult.mod <- lm(score ~ STR + english, data = CASchools)

# print the results to the console
mod
mult.mod

#К нашему примеру
summary(mult.mod)$coef
#Чтобы получить меры качества модели
summary(mult.mod)

#посчитаем вручную
# define the components
n <- nrow(CASchools)                            # number of observations (rows)
k <- 2                                          # number of regressors

y_mean <- mean(CASchools$score)                 # mean of avg. test-scores

SSR <- sum(residuals(mult.mod)^2)               # sum of squared residuals
TSS <- sum((CASchools$score - y_mean )^2)       # total sum of squares
ESS <- sum((fitted(mult.mod) - y_mean)^2)       # explained sum of squares

# compute the measures

SER <- sqrt(1/(n-k-1) * SSR)                    # standard error of the regression
Rsq <- 1 - (SSR / TSS)                          # R^2
adj_Rsq <- 1 - (n-1)/(n-k-1) * SSR/TSS          # adj. R^2

# print the measures to the console
c("SER" = SER, "R2" = Rsq, "Adj.R2" = adj_Rsq)
#>        SER         R2     Adj.R2 
#> 14.4644831  0.4264315  0.4236805


#Проверим, как это работает в R
# define the fraction of English learners        
CASchools$FracEL <- CASchools$english / 100

# estimate the model
mult.mod <- lm(score ~ STR + english + FracEL, data = CASchools) 

# obtain a summary of the model
summary(mult.mod) 

#Совершенная мультиколлинеарность в R
# define the fraction of English learners        
CASchools$FracEL <- CASchools$english / 100

# estimate the model
mult.mod <- lm(score ~ STR + english + FracEL, data = CASchools) 

# obtain a summary of the model
summary(mult.mod)         

#Создадим новую переменную, оценим уравнение
# if STR smaller 12, NS = 0, else NS = 1
CASchools$NS <- ifelse(CASchools$STR < 12, 0, 1)

# estimate the model
mult.mod <- lm(score ~ computer + english + NS, data = CASchools)

# obtain a model summary
summary(mult.mod)                                                  

#Не получилось
# if STR smaller 12, NS = 0, else NS = 1
CASchools$NS <- ifelse(CASchools$STR < 12, 0, 1)

# estimate the model
mult.mod <- lm(score ~ computer + english + NS, data = CASchools)

# obtain a model summary
summary(mult.mod)                                                  

#ПОЧЕМУ ЭТО ПРОИСХОДИТ?
#У нас просто нет школ, таких, чтобы STR<12
table(CASchools$NS)

#На данных
# set seed for reproducibility
set.seed(1)

# generate artificial data on location
CASchools$direction <- sample(c("West", "North", "South", "East"), 
                              420, 
                              replace = T)

# estimate the model
mult.mod <- lm(score ~ STR + english + direction, data = CASchools)

# obtain a model summary
summary(mult.mod)                                                 

#Последний пример на совершенную мультиколлинеарность
# Percentage of english speakers 
CASchools$PctES <- 100 - CASchools$english

# estimate the model
mult.mod <- lm(score ~ STR + english + PctES, data = CASchools)

# obtain a model summary
summary(mult.mod)                                                 