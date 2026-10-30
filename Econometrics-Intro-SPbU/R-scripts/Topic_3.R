#ТЕМА 4

#ОБРАЗОВАНИЕ И ДОХОД

# load package and attach data
library(AER)
data("CPSSWEducation")
attach(CPSSWEducation)

# get an overview
summary(CPSSWEducation)

# estimate a simple regression model
labor_model <- lm(earnings ~ education)
labor_model
# plot observations and add the regression line
plot(education, 
     earnings,
     xlab="Earnings",
     ylab="Education",
     ylim = c(0, 150))

abline(labor_model, 
       col = "steelblue", 
       lwd = 2)

labor_model
confint(labor_model)

#gendermale, esimate = 2.2268, в среднем мужчины зарабатывают больше на 2.2268