# ME705 - Inferência Bayesiana
# Trabalho Final - Parte Prática
# Probabilidade de sobrevivência de mulheres no Titanic
# Autores: Julia Folgueral (RA 277178) e Luiz Fernando de Oliveira Pereira (RA 267356)

library(tidyverse)
library(bayesrules)
library(ggplot2)
library(cowplot)


# Dados

url <- "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic <- read.csv(url)

mulheres <- titanic %>% filter(Sex == "female")

n <- nrow(mulheres)   
s <- sum(mulheres$Survived)  


# Modelo: Yi ~ iid Bernoulli(p)
# Priori: p ~ Beta(alpha, beta)
# Posteriori: p | y ~ Beta(alpha + s, beta + n - s)


# Log-posteriori 

log_post <- function(p, s, n, alpha, beta) {
  if (p <= 0 || p >= 1) return(-Inf)
  (s + alpha - 1) * log(p) + (n - s + beta - 1) * log(1 - p)
}


# Metropolis-Hastings 

MH_beta <- function(T = 50000, burn = 10000, p0 = 0.7, sigma_prop = 0.05,
                    s, n, alpha, beta) {
  chain <- numeric(T)
  chain[1] <- p0
  aceitos <- 0
  
  for (t in 2:T) {
    proposta <- rnorm(1, mean = chain[t - 1], sd = sigma_prop)
    
    # Rejeitar propostas fora do suporte
    while (proposta <= 0 || proposta >= 1) {
      proposta <- rnorm(1, mean = chain[t - 1], sd = sigma_prop)
    }
    
    log_r <- log_post(proposta, s, n, alpha, beta) -
      log_post(chain[t - 1], s, n, alpha, beta)
    
    if (log(runif(1)) < log_r) {
      chain[t] <- proposta
      aceitos <- aceitos + 1
    } else {
      chain[t] <- chain[t - 1]
    }
  }
  
  list(
    amostras = chain[(burn + 1):T],
    cadeia = chain,
    taxa_aceitacao = aceitos / (T - 1)
  )
}


# Resultados com priori Beta(1,1) 

res1 <- MH_beta(s = s, n = n, alpha = 1, beta = 1)

cat("Taxa de aceitação:", res1$taxa_aceitacao, "\n")
cat("Média posterior:", mean(res1$amostras), "\n")
cat("Mediana posterior:", median(res1$amostras), "\n")
cat("ICr 95%:", quantile(res1$amostras, probs = c(0.025, 0.975)), "\n")


# Trace plot 

png("traceplot.png", width = 8, height = 5, units = "in", res = 300)
par(mar = c(5, 5, 4, 2), cex.axis = 1.5, cex.lab = 1.7, cex.main = 1.8)
plot(res1$cadeia, type = "l",
     xlab = "Iteração", ylab = expression(p), main = "Trace Plot")
dev.off()


# Análise de sensibilidade

amostras_11  <- MH_beta(s = s, n = n, alpha = 1,  beta = 1)
amostras_55  <- MH_beta(s = s, n = n, alpha = 5,  beta = 5)
amostras_205 <- MH_beta(s = s, n = n, alpha = 20, beta = 5)

# Gráfico comparativo usando bayesrules::plot_beta_binomial
tema <- theme(
  text = element_text(size = 16),
  axis.title = element_text(size = 15),
  axis.text = element_text(size = 13),
  plot.title = element_text(size = 15)
)

legenda <- get_legend(
  plot_beta_binomial(alpha = 1, beta = 1, y = s, n = n) +
    theme(legend.position = "bottom", legend.text = element_text(size = 14))
)

g1 <- plot_beta_binomial(alpha = 1,  beta = 1, y = s, n = n) +
  coord_cartesian(xlim = c(0.3, 0.9)) + ggtitle("Beta(1,1)") +
  tema + theme(legend.position = "none")

g2 <- plot_beta_binomial(alpha = 5,  beta = 5, y = s, n = n) +
  coord_cartesian(xlim = c(0.3, 0.9)) + ggtitle("Beta(5,5)") +
  tema + theme(legend.position = "none")

g3 <- plot_beta_binomial(alpha = 20, beta = 5, y = s, n = n) +
  coord_cartesian(xlim = c(0.3, 0.9)) + ggtitle("Beta(20,5)") +
  tema + theme(legend.position = "none")

fig_final <- plot_grid(
  plot_grid(g1, g2, ncol = 2),
  plot_grid(NULL, g3, NULL, ncol = 3, rel_widths = c(0.5, 1, 0.5)),
  legenda,
  ncol = 1, rel_heights = c(1, 1, 0.12)
)

ggsave("sensibilidade.png", plot = fig_final, width = 12, height = 10, dpi = 600)


# Predição posterior para 20 novas passageiras 
# Y' | p ~ Binomial(20, p), com p amostrado da posteriori

set.seed(1)
y_pred <- rbinom(n = length(amostras_11$amostras), size = 20, prob = amostras_11$amostras)

cat("Média preditiva:", mean(y_pred), "\n")
cat("Intervalo preditivo 95%:", quantile(y_pred, probs = c(0.025, 0.975)), "\n")

png("dist_pred.png", width = 10, height = 7, units = "in", res = 300)
par(mar = c(5, 5, 4, 2), cex.axis = 1.6, cex.lab = 1.8, cex.main = 1.8)
hist(y_pred,
     breaks = seq(-0.5, 20.5, by = 1),
     main = "Distribuição Preditiva",
     xlab = "Número de sobreviventes em 20 mulheres",
     ylab = "Frequência")
dev.off()