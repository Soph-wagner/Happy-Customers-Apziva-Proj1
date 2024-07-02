# Author: Sophia Wagner
# Date: 6/07/2024
# Description: Final 2 data visualizations for analyzing dataset distribution
# Apziva Project 1 - Happy Customers

#make sure to set the correct working directory

### Load Packages !
library(tidyverse)
library(ggthemes)
library(waffle)
library(ggplot2)
library(patchwork)
library(dplyr)
library(ggthemes)

# set theme for ggplot2
ggplot2::theme_set(ggplot2::theme_minimal(base_size = 16))
# set figure parameters for knitr
knitr::opts_chunk$set(
  fig.width = 7, # 7" width
  fig.asp = 0.618, # the golden ratio
  fig.retina = 3, # dpi multiplier for displaying HTML output on retina
  fig.align = "center", # center align figures
  dpi = 300 # higher dpi, sharper image
)


### Loading in Data !!
survey <- read.csv("ACME-HappinessSurvey2020.csv")

glimpse(survey)

### OVERVIEW OF DATASET ###
# Y [0, 1] - 0:Unhappy Customer , 1:Happy Customer
# X1 - My Order Was Delivered On Time
# X2 - Contents Of My Order Was As I Expected
# X3 - I Ordered Everything I Wanted To Order
# X4 - I Paid A Good Price For My Order
# X5 - I Am Satisfied With My Courier
# X6 - The App Makes Ordering Easy For Me

# X1:X6 all [1,2,3,4,5]
# 1 - Very Dissatisfied
# 2 - Dissatisfied
# 3 - Neutral
# 4 - Satisfied
# 5 - Very Satisfied
########

question_code <- c("My Order Was Delivered On Time", "Contents Of My Order Was As I Expected",
                   "I Ordered Everything I Wanted To Order", "I Paid A Good Price For My Order", 
                   "I Am Satisfied With My Courier", "The App Makes Ordering Easy For Me")

#response_code <- c("Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied")
response_code <- c("Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied")

#color = c("red", "orange", "beige", "lightblue", "green")

sat_colr <- if_else(survey$Y == 1, "darkblue", "yellow") #darkblue is Happy, Yellow is Unhappy
y_code <- c("Happy", "Unhappy")
y_colr <- c("darkblue", "yellow")

#### GRAPH 1 ####
# attempting to make a COUNT waffle chart for X1
w1 <- survey |>
  count(X1) |>
  ggplot(aes(fill = X1, values = n)) +
  geom_waffle(n_rows = 10, size = 1, color = sat_colr) + 
  scale_fill_tableau() + 
  labs(fill = question_code[1]) + 
  theme_enhance_waffle() + 
  theme(legend.position = "top")

# COUNT waffle chart for X2
w2 <- survey |>
  count(X2) |>
  ggplot(aes(fill = X2, values = n)) +
  geom_waffle(n_rows = 10, size = 1, color = sat_colr) + 
  scale_fill_tableau() + 
  labs(fill = question_code[2]) + 
  theme_enhance_waffle() + 
  theme(legend.position = "top")

# COUNT waffle chart for X3
w3 <- survey |>
  count(X3) |>
  ggplot(aes(fill = X3, values = n)) +
  geom_waffle(n_rows = 10, size = 1, color = sat_colr) + 
  scale_fill_tableau() + 
  labs(fill = question_code[3]) + 
  theme_enhance_waffle() + 
  theme(legend.position = "top")

# COUNT waffle chart for X4
w4 <- survey |>
  count(X4) |>
  ggplot(aes(fill = X4, values = n)) +
  geom_waffle(n_rows = 10, size = 1, color = sat_colr) + 
  scale_fill_tableau() + 
  labs(fill = question_code[4]) + 
  theme_enhance_waffle() + 
  theme(legend.position = "top")

# COUNT waffle chart for X5
w5 <- survey |>
  count(X5) |>
  ggplot(aes(fill = X5, values = n)) +
  geom_waffle(n_rows = 10, size = 1, color = sat_colr) + 
  scale_fill_tableau() + 
  labs(fill = question_code[5]) + 
  theme_enhance_waffle() + 
  theme(legend.position = "top")

# COUNT waffle chart for X6
w6 <- survey |>
  count(X6) |>
  ggplot(aes(fill = X6, values = n)) +
  geom_waffle(n_rows = 10, size = 1, color = sat_colr) + 
  scale_fill_tableau() + 
  labs(fill = question_code[6]) + 
  theme_enhance_waffle() + 
  theme(legend.position = "top")

#Finally displaying all waffle charts
w1 + w2 + w3
w4 + w5 + w6


#### GRAPH 2 ####
# bar chart for X1 colored by Y category
p1 <- ggplot(survey, aes(x = X1, fill = sat_colr)) + 
  geom_bar(stat = "count", position = position_dodge(width = 1), 
           color = "white", width = 1) + 
  labs(title = question_code[1], 
       x = "Response", y = "Count", 
       fill = "Happy Rating") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8),
        #places the title inside the plot panel
        plot.title.position = "plot",
        #removes the minor grid lines on the x axis !
        panel.grid.minor.x = element_blank(), 
        plot.margin = unit(c(1, 1, 2, 1), "cm")) + 
  scale_fill_manual(values = y_colr, labels = c(y_code))

# bar chart for X2 colored by Y category
p2 <- ggplot(survey, aes(x = X2, fill = sat_colr)) + 
  geom_bar(stat = "count", position = position_dodge(width = 1), 
           color = "white", width = 1) + 
  labs(title = question_code[2], 
       x = "Response", y = "Count", 
       fill = "Happy Rating") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8),
        #places the title inside the plot panel
        plot.title.position = "plot",
        #removes the minor grid lines on the x axis !
        panel.grid.minor.x = element_blank(), 
        plot.margin = unit(c(1, 1, 2, 1), "cm")) + 
  scale_fill_manual(values = y_colr, labels = c(y_code))

# bar chart for X3 colored by Y category
p3 <- ggplot(survey, aes(x = X3, fill = sat_colr)) + 
  geom_bar(stat = "count", position = position_dodge(width = 1), 
           color = "white", width = 1) + 
  labs(title = question_code[3], 
       x = "Response", y = "Count", 
       fill = "Happy Rating") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8),
        #places the title inside the plot panel
        plot.title.position = "plot",
        #removes the minor grid lines on the x axis !
        panel.grid.minor.x = element_blank(), 
        plot.margin = unit(c(1, 1, 2, 1), "cm")) + 
  scale_fill_manual(values = y_colr, labels = c(y_code))

# bar chart for X4 colored by Y category
p4 <- ggplot(survey, aes(x = X4, fill = sat_colr)) + 
  geom_bar(stat = "count", position = position_dodge(width = 1), 
           color = "white", width = 1) + 
  labs(title = question_code[4], 
       x = "Response", y = "Count", 
       fill = "Happy Rating") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8),
        #places the title inside the plot panel
        plot.title.position = "plot",
        #removes the minor grid lines on the x axis !
        panel.grid.minor.x = element_blank(), 
        plot.margin = unit(c(1, 1, 2, 1), "cm")) + 
  scale_fill_manual(values = y_colr, labels = c(y_code))

# bar chart for X5 colored by Y category
p5 <- ggplot(survey, aes(x = X5, fill = sat_colr)) + 
  geom_bar(stat = "count", position = position_dodge(width = 1), 
           color = "white", width = 1) + 
  labs(title = question_code[5], 
       x = "Response", y = "Count", 
       fill = "Happy Rating") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8),
        #places the title inside the plot panel
        plot.title.position = "plot",
        #removes the minor grid lines on the x axis !
        panel.grid.minor.x = element_blank(), 
        plot.margin = unit(c(1, 1, 2, 1), "cm")) + 
  scale_fill_manual(values = y_colr, labels = c(y_code))

# bar chart for X6 colored by Y category
p6 <- ggplot(survey, aes(x = X6, fill = sat_colr)) + 
  geom_bar(stat = "count", position = position_dodge(width = 1), 
           color = "white", width = 1) + 
  labs(title = question_code[6], 
       x = "Response", y = "Count", 
       fill = "Happy Rating") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8),
        #places the title inside the plot panel
        plot.title.position = "plot",
        #removes the minor grid lines on the x axis !
        panel.grid.minor.x = element_blank(), 
        plot.margin = unit(c(1, 1, 2, 1), "cm")) + 
  scale_fill_manual(values = y_colr, labels = c(y_code))

# Finally displaying all bar plots
p1 + p2 + p3 + p4 + p5+ p6






#### GRAPH 2 ####
# creating a colored horizontal chart for Likert data
survey |>
  select(contains("X")) |> 
  glimpse()
# ^ checking to select only X1:X6 columns

survey |>
  select(contains("X")) |> 
  pivot_longer(
    cols = everything(),
    names_to =  "question",
    values_to = "response",
    values_transform = as.character) |>
  ggplot(aes(y = question)) + 
  geom_bar(aes(fill = fct_rev(response))) + 
  #geom_bar(aes(fill = response)) + 
  scale_fill_viridis_d(na.value = "gray", labels = c(response_code)) +
  scale_y_discrete(labels = c(question_code), expand = c(0.1, 0)) +
  labs( 
    title = "Overall Quality of Services",
    fill = "Response",
    y = "Question",
    x = "Count"
    )

##################################
#Important observation: 
# - The model is saved in the same directory as the script

#Important DATA VIS observations:
# - it could be worth while to the company to keep in X2 as a question for future surverys
#  HOWEVER, I would strongly encourage reworking the wording of the question to be more like
#  "Contents of my order met or exceeded my expectations"
# - this is becaue comparing the Likert scale graph and the colored bar graph a larger number of customers
#   are giving a 1-3 rating for X2 but most of those customers are being labeled as "happy" in the colored bar graph
# - A possible example to consider for the benefit of rewriting the question is the scenario where a customer orders 1 item
#  and they recieve 2 pieces of the item. When answering the question "Contents of my order were as I expected" the customer
#  could likely answer 'No' because they only ordered 1 item but recieved 2. However, the customer could be a Happy customer!
#  therefore, rewording the question like the suggestion above could lead to a better understanding/representation of the customer's experience
####################################