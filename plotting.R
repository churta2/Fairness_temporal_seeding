library(ggplot2)
library(dplyr)
require("ggrepel")

data <- read.csv("Low_high.csv")


colnames(data) <- c("alpha","Tau", "gain", "LB", "UB")

data$alpha=as.factor(data$alpha)


g<-ggplot(data, aes(x=Tau, y=gain, colour=alpha)) + 
  geom_line(size=0.7) +
  geom_errorbar(aes(ymin=LB, ymax=UB), width=0) +
  geom_point(size=2)+
  xlab(expression(paste(tau)))+
  ylab("Social Welfare gain")+
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))


g

g<-g + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
               panel.background = element_blank(), axis.line = element_line(colour = "black"))+ labs(color=expression(paste(alpha))) 
g
##############################33

data <- read.csv("Low_high.csv")


colnames(data) <- c("alpha","Tau", "Low", "High")

data$alpha=as.factor(data$alpha)



g<-ggplot(data, aes(x=Low, y=High, colour=alpha, shape = alpha)) + 
  geom_line(size=0.7) +
  geom_point(size=2)+
  geom_text_repel(aes(label =Tau,
                  size = 3.5)) +
  xlab("Social Welfare gain Low Endowment")+
  ylab("Social Welfare gain High Endowment")+
  scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))


g

g<-g + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
             panel.background = element_blank(), axis.line = element_line(colour = "black"))+ labs(color=expression(paste(alpha))) 
g


ggsave("low_high_final.pdf", plot = g)






