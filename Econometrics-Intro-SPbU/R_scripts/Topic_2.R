#ТЕМА 4

#ПРИМЕР
# load the AER package
library(AER)
# load the the data set in the workspace
data(CASchools)
# compute STR and append it to CASchools
CASchools$STR <- CASchools$students/CASchools$teachers 

# compute TestScore and append it to CASchools
CASchools$score <- (CASchools$read + CASchools$math)/2

# estimate the model and assign the result to linear_model
linear_model <- lm(score ~ STR, data = CASchools)

# print the standard output of the estimated lm object to the console 
linear_model
mod_summary <- summary(linear_model)
mod_summary

#Посчитаем вручную:
# determine residual degrees of freedom
linear_model$df.residual
#> [1] 418
2 * pt(-4.751327, df = 418)
#> [1] 2.78331e-06
2 * pnorm(-4.751327)
#> [1] 2.02086e-06

#________________________________
#TestScorei=β0+β1Di+ui

confint(linear_model)
#Intercept = для B0
#STR = для B1

# Create the dummy variable as defined above
CASchools$D <- CASchools$STR < 20

# Compute the average score when D=1 (low  STR)
mean.score.for.D.1 <- mean(CASchools$score[CASchools$D == TRUE])

# Compute the average score when D=0 (high STR)
mean.score.for.D.0 <- mean(CASchools$score[CASchools$D == FALSE])

#1. Что является базовой категорией
# По сравнению с базовой категорией (школой, где приходится 20 учеников и более) 
#в школе, где приходится 20 учеников и менее, результаты тестов в среднем на 7.169 меньше.



plot( CASchools$score ~ CASchools$D,        # provide the data to be plotted
      pch = 19,                             # use filled circles as plot symbols
      cex = 0.5,                            # set size of plot symbols to 0.5
      col = "Steelblue",                    # set the symbols' color to "Steelblue"
      xlab = expression(D[i]),              # Set title and axis names
      ylab = "Test Score",
      main = "Dummy Regression")

# Add the average for each group
points( y = mean.score.for.D.0, x = 0,   col="red", pch = 19)
points( y = mean.score.for.D.1, x = 1,   col="red", pch = 19)


#оценим модель
# estimate the dummy regression model
dummy_model <- lm(score ~ D, data = CASchools)
summary(dummy_model)

# confidence intervals for coefficients in the dummy regression model
confint(dummy_model)


