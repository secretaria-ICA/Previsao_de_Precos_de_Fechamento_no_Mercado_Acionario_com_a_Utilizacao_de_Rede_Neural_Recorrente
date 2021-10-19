
# Previsão de Preços de Fechamento no Mercado Acionário com a Utilização de Rede Neural Recorrente

#### Aluno: [Pedro Marques](https://github.com/pedromq)
#### Orientador: [Leonardo Forero Mendonza](https://github.com/leofome8)

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".


- [Link para o código](https://github.com/pedromq/TCC/Notebook_TCC_BI_MASTER.ipynb). 

---

### Resumo

Este trabalho tem o objetivo de desenvolver um modelo baseado em redes neurais recorrentes para a previsão do preço de fechamento de ações no mercado brasileiro. Foram realizados diversos testes com informações públicas do mercado acionário, como preço diário de fechamento, de abertura, máximo, mínimo e volumes diários negociados; além disso foi testada a incorporação de indicadores de análise técnica.

A análise técnica de ações é uma das duas principais metodologias tradicionais que procuram basear as operações no mercado acionário, sendo a outra a análise fundamentalista. Na análise técnica, através da manipulação matemática dos preços dos ativos no tempo, procura-se identificar momentos em que estes ativos devem ser comprados ou vendidos. A estas manipulações matemáticas dos preços dá-se o nome de indicadores.

### 1. Introdução



A aquisição dos dados foi realizada com o pacote [investpy](https://github.com/alvarobartt/investpy), que coleta dados no portal financeiro [Investing](https://investing.com), e retorna dados diários em formato dataframe pandas com os seguintes atributos:
* Open (Preço de abertura);
* High (Preço máximo);
* Low (Preço mínimo);
* Close (Preço de fechamento);
* Volume (Volume negociado);
* Currency (Moeda negociada).

Para fins práticos de determinação do modelo, a análise foi focada em ações da Petrobras (PETR4), porém procurou-se extrapolar o mesmo modelo para ações de três outras empresas, Vale (VALE3), Banco Itaú (ITUB4) e Bradesco (BBDC4), que juntamente com a primeira representam as quatro ações mais negociadas na Bolsa de Valores de São Paulo, conforme a carteira teórica do IBOVESPA Set-Dez/2021.

A fim de gerar uma massa de dados que fosse significativa foram adquiridos dados diários dos últimos 5 anos, tendo como data limite 08/Out/2021. Desse intervalo foram separados 15 dias como conjunto de teste para comparação com as previsões do modelo.

Na sequência foram inseridos dois atributos a partir de indicadores de análise técnica. São eles:
* rsi (Índice de Força Relativa);
* williams (Williams %R).

Foi criado um pequeno pacote Tech_Analysis para o cálculo dos indicadores acima a partir dos dados adquiridos. Abaixo as expressões dos indicadores utilizados:

---

rsi = 100 - {100 / [1 + (U<sub>n</sub>/D<sub>n</sub>)]}

Onde:

U<sub>n</sub> = Média dos preços de fechamento, em dias de alta, considerando janela móvel de n dias (adotado n = 14);

D<sub>n</sub> = Média dos preços de fechamento, em dias de baixa, considerando janela móvel de n dias.

---

williams = 100 * [(H<sub>n</sub> - C<sub>t</sub> / (H<sub>n</sub> - L<sub>n</sub>)]

Onde:

C<sub>t</sub> = Preço de fechamento mais recente considerando janela móvel de n dias (adotado n = 14);

L<sub>n</sub> = Menor preço mínimo considerando janela móvel de n dias;

H<sub>n</sub> = Maior preço máximo considerando janela móvel de n dias.

---

A coluna do atributo Currency foi descartada por não agregar informação. Também foram descartadas as 14 linhas iniciais do conjunto de dados pois apresentavam valores NaN nas colunas dos indicadores rsi e williams (devido ao fato de serem calculados em janelas móveis).

Foram propostas 5 alternativas como entrada da RNN, a fim de procurar determinar quais dados e quantas dimensões apresentam melhores resultados, em especial identificar se os indicadores de análise técnica podem ajudar a melhorar o desempenho do modelo. As alternativas utilizadas para entrada de dados na RNN foram:

* Alternativa A - 1 dimensão - Preço de fechamento;
* Alternativa B - 2 dimensões - Preço de fechamento e volume negociado;
* Alternativa C1 - 3 dimensões - Preço de fechamento, volume negociado e índice de força relativa;
* Alternativa C2 - 3 dimensões - Preço de fechamento, volume negociado e Williams %R;
* Alternativa D - 4 dimensões - Preço de fechamento, preço de abertura, preço máximo e preço mínimo.

Então, seguindo as alternativas descritas, os dados foram normalizados com Z-score, que apresentou resultados melhores que min-max, e organizados em entradas (X_train) com suas respectivas saídas de treino (y_train).

### 2. Modelagem

O modelo adotado, descrito na tabela abaixo, foi resultado de diversos testes realizados variando:
* o número de neurônios, camadas e tipo (LSTM / GRU) da RNN (4 camadas apresentaram resultados pouco significativamente melhores que 3 camadas);
* tamanho da janela móvel de entrada dos dados;
* otimizadores (NAdam apresentando melhores resultados em convergência que SGD, Adam e Adagrad);
* loss function (MAPE apresentando melhores resultados que MSE).

Foi testada ainda a conversão de séries não-estacionárias de entrada (como os preços) para séries estacionárias, seja por diferenciação, seja por adoção da variação percentual diária, mas não foram percebidos melhores resultados.

Como pretendeu-se gerar uma previsão de 15 dias, conforme seção anterior, foram testados dois formatos de saída da RNN: prevendo 15 dias em lote e prevendo 1 dia por vez com um laço iterativo para alcançar um horizonte de previsão de 15 dias. Por fim a previsão em lote não apresentou ganhos em termo de minimização do erro, assim foi adotada a previsão unitária com iteração.

   | Parâmetro | Valor |
   |---|---|
   | Janela Móvel | 12 dias |
   | Tipo RNN | LSTM |   
   | Camada Recorrente | 3 camadas * 100 neurônios |
   | Camada Dense | 1 camada * 1 a 4 neurônios (de acordo com número de dimensões da série de entrada |
   | Otimizador | NAdam |
   | Loss Function | MAPE |
   | Dropout | 0.3 |   
   | Épocas | 2000 (com callbacks de EarlyStop e CheckPoint) |
   | Validation Split | 0.1 |   
   | Tamanho do Batch | 32 |

### 3. Resultados

Abaixo consolidados os resultados obtidos considerando a média de 5 rodadas de treinamento. Foram utilizados sempre os melhores modelos, de acordo com o critério da otimização (minimização) da loss function, de cada alternativa descrita na seção 1. Introdução.

Resultados Médios para PETR4

   | Alternativa | MAPE (%) | RMSE | MAE |
   |---|---|---|---|
   | A | 7.67 | 2.41 | 2.10 |
   | B | 11.08 | 3.26 | 3.04 |   
   | C1 | 8.68 | 2.78 | 2.50 |
   | C2 | 15.35 | 4.48 | 4.28 |
   | D | 7.81 | 2.59 | 2.18 |

---

Resultados Médios para VALE3

   | Alternativa | MAPE (%) | RMSE | MAE |
   |---|---|---|---|
   | A | 30.23 | 23.88 | 23.49 |
   | B | 37.00 | 28.94 | 28.77 |   
   | C1 | 31.54 | 24.78 | 24.51 |
   | C2 | 29.41 | 23.12 | 22.85 |
   | D | 27.83 | 22.01 | 21.62 |

---

Resultados Médios para ITUB4

   | Alternativa | MAPE (%) | RMSE | MAE |
   |---|---|---|---|
   | A | 10.22 | 2.91 | 2.58 |
   | B | 8.43 | 2.48 | 2.04 |   
   | C1 | 7.19 | 1.92 | 1.67 |
   | C2 | 7.07 | 1.96 | 1.69 |
   | D | 8.33 | 2.30 | 2.05 |

---

Resultados Médios para BBDC4

   | Alternativa | MAPE (%) | RMSE | MAE |
   |---|---|---|---|
   | A | 12.51 | 2.60 | 2.56 |
   | B | 7.18 | 1.64 | 1.54 |   
   | C1 | 7.25 | 1.56 | 1.46 |
   | C2 | 10.52 | 2.19 | 2.14 |
   | D | 11.84 | 2.45 | 2.41 |

### 4. Conclusões

Primeiramente observou-se que o acréscimo de dimensões na entrada da RNN não gera necessariamente melhores resultados. Comparando as alternativas A e D, que tem respectivamente 1 dimensão (Close) e 4 dimensões (Close/Open/High/Low), as diferenças de resultados foram insignificantes (PETR4 e BBDC4) ou muito pequenas (VALE3 e ITUB4).

Comparando as alternativas A e B, percebe-se que a inclusão da dimensão Volume na entrada da rede melhora os resultados nos casos de ITUB4 e BBDC4 e piora nos casos PETR4 e VALE3. Uma hipótese que justifique essa diferença pode estar na natureza dos negócios. Enquanto ITUB4 e BBDC4 são instituições do setor financeiro, PETR4 e VALE3 são empresas de petróleo e mineração respectivamente, muito afetadas por fatores exógenos, como preço de commodities, taxas de câmbio e contexto político interno e externo, e que são bruscamente afetadas por variações de preço/volume em momentos de stress do mercado financeiro. Como proposta para um futuro trabalho sugere-se a expansão das análises com segmentação por setores da economia e indexação dos preços por taxas de câmbio e cotações de commodities, afim de verificar a repetibilidade do fenômeno e teste da hipótese.

Não foi possível concluir se a inclusão de indicadores de análise técnica incorpora benefícios na entrada da rede. Comparativamente as alternativas B e C1/C2 apresentam resultados diversos. Enquanto em ITUB4 o acréscimo de dimensão com indicadores provoca melhora pouco significativa nos resultados, em VALE3 a melhora é considerável. Em PETR4 e BBDC4 a alternativa que incorpora Williams %R acaba aumentado as métricas. 

Por fim observou-se a dificuldade de generalizar o modelo considerando um mesmo ativo e também dentro do universo abrangente de ações do mercado brasileiro. Citando apenas um exemplo, PETR4 apresentou melhor resultado médio com a entrada de dados de 1 dimensão (Alternativa A - MAPE médio 7.67% e desvio padrão 0.71%), mas a alternativa de 2 dimensões "venceu" a maior parte das rodadas (Alternativa B - MAPE médio 11.08% e desvio padrão 7.06%). Isso ocorre porque o grau de dispersão dos resultados da Alternativa B é muito maior que na Alternativa A, vide desvio-padrão, afetando sua repetibilidade, assim os resultados apresentados por esta são de forma alternada muito bons ou muito ruins. O efeito do overfitting foi observado em todas as alternativas, exceto na alternativa D.

A título de sugestão de outros trabalhos futuros, proponho que sejam acoplados adaptações de mecanismos de atenção em redes recorrentes ou utilização de imagens de gráficos de ações como entrada de dados. Como inspiração sugiro [A Dual-Stage Attention-Based Recurrent Neural Network
for Time Series Prediction
](https://arxiv.org/pdf/1704.02971.pdf) e [Visual Time Series Forecasting: An Image-driven Approach](https://arxiv.org/pdf/2107.01273.pdf). Neste último não são utilizadas LSTM e dados sequenciais, sendo a proposta utilizar redes convolucionais em um processo autoencoder, gerando diretamente imagens de previsão a partir de imagens de gráficos de entrada.

---

Matrícula: 192.671.083

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
