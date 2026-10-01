# Sistema Quant com IA

Projeto educacional para construir, em etapas pequenas e validadas, um pipeline
de trading quantitativo assistido por IA. Este projeto nao deve ser usado com
dinheiro real sem validacao extensa adicional e supervisao humana.

## Estado atual

Etapa concluida: **Etapa 0 - Ativo de controle**.
Etapa concluida: **Etapa 1 - Dados reais e motor de backtest (sem IA)**.

## Arquitetura planejada

- `data/`: dados reais e sinteticos
- `models/`: modelos preditivos e calibracao
- `backtest/`: motor de backtest e custos
- `risk/`: sizing, stops e metricas de risco
- `execution/`: simulacao de paper trading
- `tests/`: testes automatizados

## Reprodutibilidade

A seed oficial do projeto fica em `config.py`:

```python
SEED = 42
```

Todo resultado oficial deve usar essa seed. Testes com outras seeds devem ser
tratados como uma etapa separada de robustez, nao misturados aos resultados
principais.

## Etapa 0 - Ativo de controle

Foi implementado um ativo sintetico via random walk geometrico puro, sem padrao
real embutido. Ele sera usado em todos os testes futuros em paralelo aos ativos
reais.

Configuracao atual:

- Preco inicial: `100.0`
- Periodos: `1260` dias uteis, aproximadamente 5 anos
- Volatilidade diaria simulada: `1%`
- Seed: `42`

Regra permanente: se qualquer modelo futuro encontrar performance melhor que
aleatoria de forma estatisticamente significativa nesse ativo, o resultado deve
ser tratado como bug ate prova em contrario.

Validacoes estatisticas atuais da seed oficial:

- Teste t formal de uma amostra para ausencia de drift medio em retornos
  simples.
- Teste de Ljung-Box para lags 1 a 20, com correcao de Bonferroni para
  multiplas comparacoes.

## Etapa 1 - Dados reais e motor de backtest (sem IA)

Objetivo: construir o motor de backtest e validar que ele nao introduz vies,
antes de qualquer modelo preditivo entrar em cena.

Escopo:

- `data/real.py`: download e cache local (parquet) de OHLCV via `yfinance`
  para um conjunto pequeno e fixo de ativos (`REAL_ASSET_TICKERS` em
  `config.py`, atualmente `SPY` e `AAPL`).
- `backtest/engine.py`: motor de simulacao vetorizado. O sinal do dia `t` so
  pode capturar o retorno do dia `t+1` (shift explicito), o que previne
  lookahead bias estruturalmente, nao por convencao.
- `backtest/baselines.py`: baselines obrigatorios -- buy-and-hold e entrada
  aleatoria (i.i.d., sem persistencia) -- que qualquer modelo futuro precisa
  superar com significancia estatistica para ser considerado.
- Custos de transacao (`TRANSACTION_FEE_BPS` + `SLIPPAGE_BPS`) aplicados
  apenas quando a posicao muda.

Achado relevante desta etapa: o baseline de entrada aleatoria, por ter ~50%
de turnover diario (sem persistencia), sofre um drag de custo real e
esperado quando custos de transacao sao aplicados -- isso nao e um vies do
motor, e sim o comportamento correto de uma estrategia de alto turnover. Os
testes separam explicitamente a checagem de "ausencia de edge espurio"
(sem custo) da checagem de "custo de transacao calculado corretamente" (com
custo), para nao confundir as duas coisas.

Gate estatistico da etapa: buy-and-hold e entrada aleatoria rodados no ativo
sintetico (sem custo) devem produzir retorno medio estatisticamente
indistinguivel de zero. Se esse teste falhar no futuro, o motor tem um bug
de lookahead ou de calculo, independente do que os ativos reais mostrarem.

Fontes de dados candidatas para etapas futuras (macro, volume, fundamentals,
sentimento) foram deliberadamente deixadas de fora do escopo e registradas
em `docs/backlog-fontes-dados.md`, cada uma com a hipotese que precisaria
ser validada antes de entrar no pipeline.

**Correcao retroativa**: a especificacao oficial da Etapa 1
(`prompt-antigravity-quant-ia-v2`) exige "UMA estrategia simples sem IA (ex:
cruzamento de medias moveis)" e tratamento explicito de dados faltantes.
Nenhum dos dois tinha sido implementado na primeira passada desta etapa --
foram adicionados junto com o fechamento da Etapa 2 (ver abaixo), já que a
mesma estrategia de cruzamento e reaproveitada como terceiro baseline
obrigatorio.

## Etapa 2 - Baselines obrigatorios antes da IA

Tres baselines, rodados em todos os ativos (reais + sintetico), que
qualquer modelo de IA futuro precisa bater de forma consistente para se
justificar:

1. **Aleatorio** (`random_entry_persistent_signal`) -- entrada/saida
   aleatoria com persistencia (holding period configuravel,
   `RANDOM_ENTRY_AVG_HOLDING_DAYS` em `config.py`), sem nenhuma habilidade
   preditiva por construcao.
2. **Buy-and-hold** (`buy_and_hold_signal`) -- sempre 100% investido.
3. **Media movel** (`moving_average_crossover_signal`) -- cruzamento de duas
   SMAs (`SMA_FAST_WINDOW`/`SMA_SLOW_WINDOW` em `config.py`), posicionado
   quando a rapida esta acima da lenta. Janelas sao valores de referencia
   comuns, nao otimizados -- ajuste de parametro exigiria a infraestrutura
   de walk-forward reservada para a Etapa 3.

O baseline aleatorio i.i.d. (`random_entry_signal`) da Etapa 1 foi mantido
como teste de estresse extra, nao como requisito oficial da Etapa 2.

**Gate estatistico**: no ativo sintetico e com custo zero, nenhum dos 3
baselines obrigatorios deve mostrar retorno medio estatisticamente
diferente de zero -- isso e validado automaticamente nos testes (incluindo
para o cruzamento de medias moveis, que e a checagem mais rigorosa por
reagir de fato ao preco, ao contrario do buy-and-hold).

## Correções pós-Etapa 2 (revisão de metodologia)

Após fechar a Etapa 2, uma revisão crítica identificou e corrigiu quatro
lacunas antes de avançar para a Etapa 3:

1. **`backtest/statistics.py`** — erro-padrão Newey-West (HAC), robusto a
   autocorrelação nos retornos (o t-test ingênuo assume i.i.d., o que só é
   garantido para o ativo sintético via Ljung-Box; estratégias com
   persistência de posição podem ter retornos autocorrelacionados).
   Substituiu o t-test ingênuo em todos os testes do motor
   (`tests/test_engine.py`) e nos dois scripts de execução. O teste de
   drift bruto do ativo sintético (`tests/test_synthetic.py`, Etapa 0) foi
   deixado como está, por testar retornos já comprovadamente i.i.d.
2. **Correção de Bonferroni para múltiplas comparações** — os scripts de
   execução testam ~20 combinações (ativo × baseline) simultaneamente; sem
   correção, a chance de pelo menos um falso positivo por acaso é alta.
   `significant_bonferroni` nas tabelas de saída usa o limiar corrigido
   (`alpha / num_tests`), bem mais rigoroso que o `|t| > 1.96` usado antes.
   Os gates individuais em `pytest` continuam sem correção de Bonferroni de
   propósito -- são afirmações isoladas pré-registradas, não uma tabela
   exploratória.
3. **Teste de sizing fracionário** — o motor sempre aceitou floats
   arbitrários, mas nunca tinha sido testado com posições fracionárias (só
   0.0/1.0). Adicionados testes validando retorno proporcional e custo
   escalando com a magnitude da mudança de posição, não só com sua
   ocorrência -- pré-requisito direto para a Etapa 4 (Kelly fracionário).
4. **`backtest/reporting.py`** — cada execução de `run_stage1_backtest.py`
   e `run_stage2_baselines.py` agora salva a tabela de resultados em
   `results/<etapa>_<timestamp>.csv`, para comparação entre execuções sem
   depender de copiar saída do console manualmente.

Recomendação (não aplicada automaticamente: não tenho visibilidade do
`.gitignore` real do seu projeto): adicionar `results/` e `data/cache/` ao
`.gitignore`, já que são artefatos gerados localmente, não código-fonte.

## Como rodar os testes

Dependencias desta etapa:

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m pytest
```

## Como rodar os backtests

```powershell
python run_stage1_backtest.py
```

Roda os baselines da Etapa 1 (buy-and-hold + 2 variantes de entrada
aleatoria) nos ativos reais e no sintetico.

```powershell
python run_stage2_baselines.py
```

Roda os 3 baselines obrigatorios da Etapa 2 (aleatorio, buy-and-hold, media
movel) nos mesmos ativos, com o gate estatistico explicito na saida.

Ambos baixam (ou leem do cache local em `data/cache/`) os ativos reais
configurados.

## Limitacoes conhecidas

- Ainda nao ha sinal preditivo de IA, gestao de risco (Kelly) ou paper
  trading. Esses itens pertencem as proximas etapas.
- O motor de backtest ainda nao suporta position sizing dinamico -- apenas
  posicoes fixas (0 ou 1), suficiente para validar a infraestrutura antes de
  introduzir complexidade.
- As janelas da media movel (20/50) e o holding period do baseline aleatorio
  (20 dias) sao valores de referencia, nao ajustados nos dados.

## Etapa 3 - Modelo de IA com probabilidade calibrada

Escopo inicial (por decisão explícita): apenas `SPY` + ativo sintético
(`STAGE3_TICKERS` em `config.py`). Expandir para os demais ativos reais é
trabalho futuro, só depois de validar aqui.

**Alvo**: probabilidade de retorno positivo em `PREDICTION_HORIZON_DAYS=20`
dias úteis à frente — não classificação binária "sobe/desce".

**Features** (`models/features.py`), deliberadamente poucas e só de preço,
por escolha explícita seguindo a filosofia "poucas features" do projeto
contra overfitting:
- `momentum_5`, `momentum_20`, `momentum_60` — retorno simples sobre as
  janelas de 5/20/60 dias.
- `volatility_20` — desvio padrão dos retornos diários, janela de 20 dias.
- `sma_distance_50` — distância relativa entre o preço e sua própria SMA de
  50 dias.

Todas causais (usam só `rolling`/`shift` com janela passada), sem lookahead
-- validado em teste com um "spike futuro" que não pode alterar valores de
dias anteriores.

**Modelos** (`models/calibration.py`): XGBoost (modelo principal) comparado
contra regressão logística (baseline interno, exigido pela especificação).
XGBoost é calibrado com `CalibratedClassifierCV` (isotonic), ajustado numa
fatia cronológica *posterior* à usada para treinar a árvore -- nunca a
mesma linha usada pra treinar e pra calibrar.

**Walk-forward purgado** (`models/walk_forward.py`): janela de treino
expansiva, blocos de teste de `WALK_FORWARD_TEST_WINDOW_DAYS=126` dias.
Como o rótulo olha `PREDICTION_HORIZON_DAYS` dias à frente, as
`WALK_FORWARD_PURGE_DAYS` observações imediatamente antes de cada bloco de
teste são removidas do treino -- sem isso, o rótulo vazaria informação do
período de teste para dentro do treino mesmo com os índices "certos" (ver
achado abaixo, onde esse tipo de vazamento por sobreposição quase gerou um
falso positivo).

**Holdout final**: tudo a partir de `STAGE3_HOLDOUT_START_DATE=2024-01-01`
nunca entra no walk-forward, reservado para uma única avaliação final
depois que o modelo estiver pronto -- não automatizada neste script de
propósito, para evitar consultar o holdout repetidamente.

**Gate estatístico obrigatório**: o modelo precisa ter acurácia fora da
amostra estatisticamente equivalente a 50% no ativo sintético. Se não for,
o script para e avisa em vez de seguir para os ativos reais.

**Achado relevante durante o desenvolvimento**: a primeira versão do gate
usava um t-test ingênuo e acusou o XGBoost de ter "encontrado padrão" no
sintético (56% de acurácia, t=2,38 -- pareceria um bug real). Investigando
antes de aceitar, a causa era o mesmo problema já corrigido nos baselines:
observações vizinhas dentro do mesmo bloco de teste têm rótulos com
horizontes fortemente sobrepostos (o rótulo de um dia usa quase os mesmos
20 dias futuros que o do dia seguinte), autocorrelacionando os
acertos/erros. Corrigido trocando para Newey-West (t caiu para 1,35, gate
passa). Regressão coberta por teste dedicado
(`tests/test_stage3_gate.py`).

**SHAP**: aplicado no modelo final (refit em todo o período de
desenvolvimento, excluindo o holdout), gerando importância média por
feature e um gráfico salvo em `results/shap_summary_<ticker>.png`.

**Fora de escopo, de propósito**: sizing proporcional à confiança do modelo
(Kelly) -- isso é Etapa 4. O sinal usado para comparar contra os baselines
aqui é binário (`prob > 0.5`), mesma lógica dos baselines da Etapa 2, para
comparação justa.

## Como rodar o modelo da Etapa 3

```powershell
python run_stage3_model.py
```

Roda o ativo sintético primeiro (gate obrigatório) e só então o(s) ativo(s)
reais configurados, imprimindo calibração, gate estatístico, comparação
contra os baselines, e análise SHAP.

## Limitações conhecidas

- Ainda não há gestão de risco (Kelly) ou paper trading. Esses itens
  pertencem às próximas etapas.
- O motor de backtest suporta position sizing fracionário (testado), mas
  nenhum componente desta etapa ainda o utiliza -- o sinal do modelo é
  binário de propósito, para comparação justa com os baselines da Etapa 2.
- As janelas de feature, o horizonte de previsão e os parâmetros do
  walk-forward são valores de referência, não ajustados/otimizados nos
  dados (ajuste de hiperparâmetro é um risco de overfitting que não foi
  endereçado nesta etapa).
- Avaliação formal do holdout final não é automatizada -- é uma decisão
  deliberada, para só ser executada quando o time estiver confiante de que
  o modelo está pronto.
- Escopo de ativos reais limitado a SPY nesta primeira passada.

## Auditoria pós-Etapa 3

Depois de fechar a primeira versão da Etapa 3, uma auditoria dedicada
encontrou e corrigiu quatro problemas antes de confiar nos resultados:

1. **Gate agregado, não seed-a-seed** (`STAGE3_SYNTHETIC_GATE_SEEDS` em
   `config.py`) — o gate estatístico antes rodava contra uma única seed do
   ativo sintético, arriscando passar por sorte numa única realização.
   Investigação em duas rodadas:
   - **1ª tentativa**: rodar contra 5 seeds, exigindo que cada uma passasse
     individualmente (com Bonferroni entre as 10 comparações). A seed=303
     falhou mesmo após a correção (regressão logística: 32% de acurácia,
     p=0,00011).
   - **Investigação mais profunda, antes de aceitar ou "consertar" o
     modelo**: inspecionei os coeficientes aprendidos e descobri que a
     hipótese inicial (modelo seguindo tendência) estava errada -- o
     modelo tinha aprendido reversão à média no treino
     (`momentum_60`: coeficiente -1,44), que simplesmente não se sustentou
     no bloco de teste seguinte. Simulei 20 seeds independentes extras
     para checar se isso era raro: **2 de 20 (10%) tiveram um fold tão
     extremo quanto o da seed 303** -- não é uma anomalia dela
     especificamente, é variância normal de folds pequenos (126 dias de
     teste, ~6 blocos independentes efetivos por fold, dado a
     autocorrelação por sobreposição de horizonte).
   - **Causa raiz real**: o desenho do gate era estatisticamente
     ineficiente -- exigir que 5 amostras pequenas (poucas observações
     efetivas cada) passem *individualmente* é como exigir que cada
     estudo pequeno de uma meta-análise seja significativo sozinho, em vez
     de agregar a evidência. Corrigido: as predições fora da amostra de
     todas as seeds agora são **agrupadas** antes de UM único teste de
     significância (Bonferroni entre 2 modelos, não 10 combinações). Com
     1890 observações agregadas, o gate passa (logistic t=-1,48;
     xgboost t=-0,065) -- resultado estatisticamente sólido, não obtido
     escondendo ou ajustando a seed problemática.
   - Diagnóstico por seed continua impresso na saída do script (apenas
     informativo, não decide o gate sozinho).
2. **Calibração adaptativa** (`models/calibration.py`) — `isotonic` pode
   "decorar" ruído com poucos pontos; folds de walk-forward iniciais têm
   janelas de calibração pequenas. Agora a escolha entre `isotonic` e
   `sigmoid` é automática, baseada no tamanho da fatia de calibração
   (`CALIBRATION_METHOD_MIN_SAMPLES_FOR_ISOTONIC`), com o método escolhido
   reportado por fold na saída do script.
3. **Teste de integração do purge** (`tests/test_purge_integration.py`) —
   além dos testes de índice já existentes, um novo teste planta um choque
   de preço permanente exatamente no limite treino/teste e prova, pelo
   valor real do rótulo (não só aritmética de posição), que linhas
   contaminadas (rótulo que "vê" o choque) são removidas do treino quando
   o purge está ativo.
4. **Trava de configuração** (`_validate_stage3_config` em
   `run_stage3_model.py`) — impede em tempo de execução que
   `WALK_FORWARD_PURGE_DAYS` fique menor que `PREDICTION_HORIZON_DAYS`, o
   que reintroduziria vazamento de rótulo silenciosamente numa edição
   futura do `config.py`.

Além desses quatro, a auditoria encontrou e corrigiu um **quinto problema
não pedido explicitamente, mas descoberto no processo**: o backtest do
sinal do modelo incluía anos de posição "flat" (fora do mercado) antes da
primeira predição fora da amostra existir, enquanto os baselines rodavam
sobre o período de desenvolvimento inteiro — uma comparação injusta que
penalizava o modelo artificialmente. Corrigido restringindo todos os
sinais (modelo E baselines) à mesma janela, começando na primeira predição
disponível (`_comparison_window` em `run_stage3_model.py`, coberto por
teste de regressão).

70/70 testes passam após as correções (inclui teste de regressão que
reproduz o padrão exato: seeds individuais com acurácias
[0,51; 0,53; 0,53; 0,32; 0,41] têm uma falha espúria individual, mas o
teste agregado sobre todas não rejeita a hipótese nula).

## Auditoria a partir da primeira rodada real (SPY)

Depois de rodar `run_stage3_model.py` pela primeira vez com dados reais do
SPY (11 folds de walk-forward, cobrindo 2018-2023 de desenvolvimento +
holdout de 2024 em diante), a leitura detalhada da saída revelou mais
quatro problemas:

1. **Mesmo bug da janela de comparação, na outra ponta.** A correção
   anterior só limitava o *início* da janela de comparação (primeira
   predição disponível). A saída real mostrou 1454 dias na janela impressa
   contra apenas 1386 predições agregadas -- 68 dias no final sem predição
   nenhuma, onde o sinal do modelo caía para flat enquanto os baselines
   continuavam ativos. Causa: os blocos de teste do walk-forward preenchem
   o dataset em fatias de tamanho fixo, deixando um resto no final sem
   cobertura. Corrigido: `_comparison_window` agora limita os dois lados
   (`predictions.index.min()` E `.max()`), coberto por teste de regressão
   dedicado.
2. **Regressão logística nunca era testada como estratégia.** A saída real
   mostrou Brier score da regressão logística (0,2242) melhor que o do
   XGBoost calibrado (0,2450, quase idêntico ao baseline "sempre 50%") --
   mas só o sinal do XGBoost entrava no backtest comparativo. Corrigido:
   agora os dois modelos aparecem na tabela final, lado a lado com os
   baselines.
3. **`total_costs_paid` tinha sumido da tabela da Etapa 3** (presente nas
   Etapas 1 e 2) -- inconsistência simples, restaurada.
4. **`max_calibration_gap` pode ser dominado por um bin de 1 observação.**
   A saída real mostrou gap de calibração de 1,0000 para o XGBoost no SPY
   -- alarmante à primeira vista, mas causado por um único ponto isolado
   num bin quase vazio, não por miscalibração real generalizada.
   Adicionado `expected_calibration_error` (ECE, ponderado pelo tamanho de
   cada bin) em `models/evaluation.py` como métrica complementar mais
   robusta -- testado explicitamente para não ser dominado por bins
   pequenos, ao contrário do gap máximo.

## Auditoria geral do projeto (revisão de todos os módulos)

Uma varredura completa por todos os módulos (não só a saída do último
run) encontrou mais dois problemas reais, além de resolver um item que
antes só estava listado como "em aberto":

1. **Correlação entre features, agora medida e reportada, não só
   suposta.** `momentum_20` e `sma_distance_50` têm correlação de 0,87-0,88
   (dependendo do ativo) -- confirmando quantitativamente a suspeita
   registrada antes. `models/features.check_feature_correlation` agora
   roda automaticamente antes de cada análise SHAP e imprime um aviso
   explícito quando pares excedem `|correlação| >= 0.7`, em vez de deixar
   isso implícito. Não removi nenhuma feature unilateralmente -- isso é
   uma decisão de modelagem real que precisa ser tomada conscientemente,
   não uma correção automática.
2. **O modelo usado para SHAP era treinado com só ~80% dos dados**, não
   "todo o período de desenvolvimento" como o texto da saída afirmava --
   `train_models` reserva uma fatia para calibração, e o SHAP reusava
   `xgboost_raw` (a parte de ajuste, não o total). Corrigido com
   `train_xgboost_for_interpretation`, que treina em 100% do dataset --
   sem necessidade de reservar fatia de calibração, já que a saída de
   probabilidade desse modelo nunca é usada para nada que exija
   calibração, só para interpretar quais features pesam mais.

76/76 testes passam.

## Auditoria cruzada (revisão de um relatório de outro agente)

Depois da auditoria própria, o usuário submeteu o projeto a uma auditoria
independente feita por outro agente (Antigravity), que reportou 9 achados
e alegou tê-los corrigido. Em vez de aceitar o relatório, cada achado foi
verificado empiricamente contra o código real entregue -- resultado:
**3 dos 9 "bugs corrigidos" não eram bugs reais**, e uma das correções
não funcionava.

| # | Achado | Veredito | Como foi verificado |
|---|---|---|---|
| 1 | `backtest/validation.py` dead code perigoso | ✅ Real | Arquivo confirmado existente; contém `walk_forward_splits` baseado em datas, sem `purge_size` -- se importado por engano, reintroduz vazamento de rótulo silencioso. Origem do arquivo não determinada (pode ser de sessão anterior com outra ferramenta). Deprecado com aviso explícito; recomendação é remover de vez. |
| 2 | `total_costs_paid` "circularidade" | ❌ **A correção piorava a métrica** | Teste numérico direto (5 dias, custo de 1%, posição alternando): fórmula "corrigida" (baseada em equity bruto hipotético) reportou $4000,00 de custo; a conta real só caiu $3940,40 (de $100.000,00 para $96.059,60). Reversão aplicada, com o exemplo numérico documentado no código para não ser "corrigido" de novo por engano. |
| 3 | `iloc[1:]` sem comentário | ✅ Legítimo | Comentário adicionado, sem mudança de comportamento -- mantido |
| 4 | Log de forward-fill enganoso | ❌ **Não era bug** | Leitura cuidadosa do código original mostra que a variável usada no log já era calculada no ponto certo (antes do `dropna`) -- o "conserto" só introduziu uma variável intermediária com valor idêntico |
| 5 | `resolved_end_date` não usado no `load_all` | ✅ Legítimo | Confirmado via diff -- mantido |
| 6 | Falta `StandardScaler` na regressão logística | ✅ **Achado real, bem corrigido** | `Pipeline` com `StandardScaler` fitado só no treino de cada fold -- mantido |
| 7 | Bug SHAP/matplotlib (figura errada fechada) | ❌ Não era bug no código original | Testado empiricamente: `plot_size=None` já fazia o SHAP reaproveitar a figura certa, sem vazamento. A correção aplicada (`plt.gcf()` após `summary_plot`) também funciona -- mantida, sem prejuízo, mesmo não corrigindo um bug real |
| 8 | Empate no rótulo não documentado | ✅ Legítimo | Comentário adicionado -- mantido |
| 9 | `fillna` mascarando NaN interno | ❌ **A correção não funcionava (código morto)** | Teste com NaN plantado de propósito no meio da série: motor não levantava erro, convertia silenciosamente para "0% de retorno" -- a checagem rodava *depois* do `fillna(0.0)`, que já apaga qualquer NaN antes da validação rodar. Corrigido movendo a checagem para antes do `fillna`, validado com o mesmo teste que expôs o problema. |

**Lacuna de cobertura de teste identificada por esse episódio**: o caminho
de geração do SHAP (`_run_shap_analysis`) nunca teve teste automatizado --
por isso uma correção quebrada (a versão do relatório escrito, que usava
`shap.summary_plot(..., ax=ax)`, um parâmetro que não existe na versão
instalada do shap e gera `TypeError`) poderia ter passado despercebida até
alguém rodar o script de verdade. Adicionado `tests/test_shap_smoke.py`,
confirmado que ele de fato pega esse tipo de regressão (testado
reintroduzindo o bug deliberadamente e vendo o teste falhar).

79/79 testes passam após essa rodada.

**Lição de processo**: mesmo relatórios de auditoria detalhados e bem
formatados (com diffs, tabelas de status, "76/76 passando") podem conter
diagnósticos incorretos e correções que não funcionam ou que pioram o que
já estava certo. A mesma cautela que o projeto aplica a resultados "bons
demais" em backtests se aplica aqui: verificar empiricamente antes de
aceitar, mesmo quando a fonte parece rigorosa.

## Itens em aberto (não resolvidos, decisão pendente)

- Poucos folds de walk-forward por seed (~3 no sintético, ~11 no SPY)
  permanecem uma limitação do tamanho da série histórica, não do código.
- Regularização da regressão logística (`C` padrão do scikit-learn,
  sem ajuste) não foi revisada -- os coeficientes observados na
  investigação da seed=303 (`momentum_60`: -1,44) sugerem que o modelo
  pode estar aprendendo relações fortes com pouco dado por fold.
- Resultado real no SPY (2018-2023): nenhuma estratégia (modelo, aleatório,
  média móvel) bateu buy-and-hold de forma estatisticamente significativa
  -- documentado com honestidade, não é motivo para forçar mais
  complexidade no modelo.
- `momentum_20`/`sma_distance_50` altamente correlacionadas -- agora
  detectado e reportado automaticamente, mas nenhuma feature foi removida;
  decisão de simplificar (ou não) o conjunto de features fica em aberto.

## Avaliação final de holdout (rodar UMA VEZ, com cautela)

`run_stage3_holdout_evaluation.py` -- script separado e deliberado (não
roda automaticamente com `run_stage3_model.py`), para a avaliação final e
única do modelo contra o período `STAGE3_HOLDOUT_START_DATE` (2024-01-01)
em diante, nunca tocado pelo walk-forward.

**Construção sem vazamento** (coberta por `tests/test_holdout_evaluation.py`,
mesmo tipo de teste de choque de preço plantado usado em
`test_purge_integration.py`):
- Dataset de **treino**: construído a partir de `dev_close` (série
  truncada antes do holdout) -- rótulos que olhariam além dessa fronteira
  já saem `NaN` estruturalmente, o mesmo mecanismo já usado nos folds de
  walk-forward, aplicado aqui à fronteira dev/holdout.
- Dataset de **holdout**: construído a partir da série completa (para as
  features poderem usar o warmup da cauda do período de dev normalmente) e
  então filtrado para datas `>= STAGE3_HOLDOUT_START_DATE`. Nunca usado
  para treino.
- Modelo final treinado uma única vez em 100% do dataset de treino (sem
  folds -- só existe um holdout, avaliado uma vez).

```powershell
python run_stage3_holdout_evaluation.py
```

**Aviso explícito no próprio script**: rodar isso repetidamente e ajustar
o modelo com base no resultado anula o propósito do holdout -- ele vira só
mais um conjunto de validação. Só rode quando estiver confiante que o
modelo está pronto.

81/81 testes passam.

## Auditoria cruzada com relatório externo (Antigravity)

Um agente separado (Antigravity, rodando localmente na mesma pasta) gerou
um relatório de auditoria com 9 "bugs corrigidos". Cada um foi verificado
empiricamente (não só lido) antes de aceitar:

- **3 dos 9 não eram bugs reais**: a "correção" do log em `data/real.py`
  recalculava o mesmo valor sem mudar nada; o "bug" de figura do
  matplotlib/SHAP não existia no código original (`plot_size=None` já
  reaproveitava a figura certa); a fórmula de `total_costs_paid` que o
  relatório chamou de "circular" na verdade estava correta -- a
  "correção" proposta (`gross_equity`) foi **revertida** após prova
  numérica: num exemplo de 5 dias, a fórmula nova reportava $4000 de
  custo quando a conta real só caiu $3940,40. Documentado com o exemplo
  exato no docstring de `total_costs_paid`.
- **1 correção não funcionava de verdade**: a validação de NaN interno em
  `backtest/engine.py` tinha sido colocada *depois* do `fillna(0.0)`, que
  já apagava qualquer NaN antes da checagem rodar -- código morto,
  confirmado plantando um NaN de propósito e vendo que nenhum erro era
  levantado. Corrigido: a checagem agora roda antes do fillna.
- **2 achados eram reais e bem corrigidos**: falta de `StandardScaler` na
  regressão logística (features com escalas diferentes distorcem
  regularização L2) e `resolved_end_date` não sendo passado consistentemente
  para `load_all`.
- **`backtest/validation.py`**: dead code real, sem nenhum histórico de
  commit (`git log` vazio) -- indício de que foi criado pelo próprio
  Antigravity durante a sessão de auditoria, não uma sobra de sessão
  anterior. **Removido** (não só deprecado): um docstring de aviso não
  impede alguém de importar a função errada por engano.

## Avaliação de holdout: trava contra reexecução acidental

Descoberto durante a mesma auditoria: `run_stage3_holdout_evaluation.py`
(também gerado externamente, mas nunca executado contra dados reais --
verificado checando `results/` antes de qualquer alteração) tinha o aviso
de "rode só uma vez" apenas como `print()`, sem nada no código impedindo
reexecução de fato. Corrigido: `main()` agora verifica se já existe um
resultado de holdout salvo (`etapa3_*_HOLDOUT_FINAL_*.csv`) e recusa
rodar de novo sem a flag explícita `--force`.

```powershell
python run_stage3_holdout_evaluation.py            # roda normalmente
python run_stage3_holdout_evaluation.py --force    # reexecuta apos revisar
```

## Consolidação de código duplicado

`_evaluate_calibration` e a lógica de restrição de janela de comparação
já tinham **divergido** entre `run_stage3_model.py` e
`run_stage3_holdout_evaluation.py` (só no texto dos avisos, por enquanto,
mas confirma o risco). Extraídas para `models/evaluation.py`
(`print_calibration_report`) e o novo `models/comparison.py`
(`comparison_window`, `predictions_to_backtest_signal`), usadas pelos
dois scripts.

84/84 testes passam após esta rodada.

## Etapa 4 - Gestão de risco com sizing dinâmico

Escopo: todos os 4 ativos reais (`STAGE3_TICKERS` expandido de `["SPY"]`
para `["SPY", "AAPL", "PETR4.SA", "VALE3.SA"]`), reaproveitando o mesmo
walk-forward purgado da Etapa 3 (`_run_walk_forward`, importado
diretamente de `run_stage3_model.py` em vez de duplicado).

**Kelly fracionário** (`risk/kelly.py`): `f* = p - (1-p)/b`, com `p` a
probabilidade calibrada do XGBoost e `b` a razão ganho/perda estimada de
forma causal (retornos realizados de `PREDICTION_HORIZON_DAYS` dias,
janela móvel de `KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS`). Multiplicado por
`KELLY_FRACTION=0.5` (meio-Kelly, a recomendação prática mais citada —
Kelly completo maximiza crescimento assintótico mas é extremamente
sensível a erro de estimativa).

**Achado importante durante o desenvolvimento**: Kelly com `p=0.5` (sem
edge nenhuma) **não é garantido dar posição zero**. Testei em várias seeds
do ativo sintético: a razão ganho/perda `b` às vezes passa de 1.0 só por
ruído amostral (mesmo sem edge real), produzindo posições pequenas mas
não-nulas (até ~10% em algumas seeds) puramente pela assimetria natural do
log-normal. Por isso o gate desta etapa testa o **retorno resultante da
estratégia** (Newey-West + Bonferroni, agregado entre as 5 seeds — mesma
metodologia da Etapa 3), não "posição média = 0".

**Stop-loss e circuit breaker** (`risk/managed_backtest.py`): motor
sequencial (não vetorizado, ao contrário do motor da Etapa 1) porque essas
regras dependem do caminho percorrido — se uma posição é fechada hoje
depende do preço de entrada e do drawdown acumulado, que dependem de todas
as decisões anteriores.
- Stop-loss (`STOP_LOSS_PCT=0.10`): fecha a posição individual se cair
  10% do preço de entrada. Permite reentrada no dia seguinte se o modelo
  ainda pedir exposição — comportamento correto de stop-loss (distinto de
  circuit breaker), validado em teste.
- Circuit breaker (`CIRCUIT_BREAKER_DRAWDOWN_PCT=0.20`): uma vez que o
  drawdown da carteira passa de 20%, força flat pelo **resto do backtest**,
  sem religar. Validado que não reativa mesmo com recuperação forte de
  preço depois.

**VaR e volatilidade** (`risk/metrics.py`): VaR histórico (percentil
empírico, não paramétrico — mesma preferência por métodos robustos já
usada no Newey-West) e volatilidade anualizada.

**Comparação**: Kelly+risco vs sinal binário simples (Etapa 3) vs
buy-and-hold, todos rodando pelo mesmo motor de risco (para comparação
justa de drawdown/VaR). Em teste com dados simulados: Kelly reduziu
drawdown de -20% para -5% e volatilidade de 12% para 3%, ao custo de
menor retorno — trade-off esperado de sizing conservador, não uma falha.

**Achado corrigido durante os testes**: a primeira versão do pipeline
usava a série de preço completa (incluindo o período de holdout de 2024
em diante) para construir o walk-forward, silenciosamente consumindo
dados que a Etapa 3 reserva deliberadamente. Corrigido truncando para o
período de desenvolvimento antes de qualquer coisa (`respect_holdout` em
`_get_oos_predictions`), com teste de regressão dedicado.

**Fora de escopo, de propósito**: reativação do circuit breaker (regra de
"religar"), Kelly multi-ativo (correlação entre posições), shorting e
leverage (Kelly aqui é limitado a [0, 1], mesma convenção de todo o
projeto).

111/111 testes passam.

## Como rodar a Etapa 4

```powershell
python run_stage4_risk_management.py
```

Roda o gate no sintético primeiro (5 seeds, agregado); só segue para os
ativos reais se passar.

## Auditoria pós-Etapa 4

Três problemas identificados e corrigidos antes de avançar:

1. **Preço de entrada não acompanhava sizing variável.** Com Kelly, a
   posição muda de tamanho todo dia (não é mais 0/1 como nas etapas
   anteriores). O stop-loss fixava o preço de entrada só na primeira vez
   que a posição saía de zero -- se ela crescia depois (ex: 5% → 15%), a
   referência de stop continuava sendo o preço da entrada pequena
   original. Corrigido: preço de entrada agora é uma média ponderada,
   atualizada quando a posição *cresce* (comprando mais a um preço novo);
   quando a posição só *encolhe* (sem passar por zero), o preço de entrada
   não muda -- vender uma parte não altera o custo médio do que ainda está
   em carteira. Coberto por dois testes (`test_managed_backtest.py`).
2. **Circuit breaker "achatava" o resto do gráfico sem avisar.** Depois de
   disparar, a estratégia fica flat até o fim do período -- e as métricas
   agregadas (Sharpe, retorno total, volatilidade) misturavam o
   desempenho real (antes do desligamento) com a cauda plana depois,
   podendo parecer "desempenho fraco e consistente" quando na verdade é
   "uma perda concentrada seguida de inatividade". Agora
   `RiskManagedBacktestResult` expõe `active_trading_days`/`total_days`, e
   o script principal imprime um aviso explícito com o percentual de
   tempo realmente ativo sempre que o circuit breaker dispara.
3. **Kelly instável com amostra efetiva pequena.** A janela de estimativa
   de ganho/perda (252 dias, retornos de 20 dias sobrepostos) tem só
   ~12-13 blocos genuinamente independentes -- o mesmo problema de
   autocorrelação por sobreposição já visto na Etapa 3, agora alimentando
   o tamanho da aposta em vez de um teste estatístico. Adicionado
   shrinkage em `estimate_win_loss_ratio`: a razão ganho/perda estimada é
   puxada em direção a 1.0 (neutro) proporcionalmente ao tamanho da
   amostra efetiva -- pouca amostra, puxa forte para 1.0; muita amostra,
   converge para a estimativa bruta. Efeito observado no gate do
   sintético: disparos de stop-loss caíram para 0 em todas as 5 seeds
   (antes, 2 das 5 tinham 1 disparo cada).

118/118 testes passam.

## Recomendação de processo (não é código)

Nenhum commit git foi feito desde etapas bem antigas do projeto -- `git
status` mostra praticamente tudo como "untracked". Isso já causou atrito
real: o arquivo `backtest/validation.py` sem histórico (impossível saber
com certeza a origem) e o script de holdout sem trava de segurança
(descoberto só por inspeção manual). Recomendação: commitar agora
(`git add -A && git commit -m "Etapa 4 completa"`) e a partir daqui
commitar ao final de cada etapa concluída, como o prompt original do
projeto sempre pediu.

## Investigação do primeiro resultado real (SPY, AAPL, PETR4.SA, VALE3.SA)

A primeira rodada real revelou um padrão que exigiu investigação antes de
aceitar qualquer conclusão:

**PETR4.SA ficou ativa só 2,7-3,3% do período** (37-46 de 1386 dias) --
circuit breaker disparou quase imediatamente para as três estratégias,
inclusive buy-and-hold puro. Como buy-and-hold também disparou, não é bug
do modelo: é um evento de mercado real e severo logo no início da janela
de teste (~maio/2018), coincidindo com a greve dos caminhoneiros no
Brasil, que derrubou a Petrobras com força devido à disputa sobre preço de
combustível. **O circuit breaker fazendo exatamente o que foi desenhado
para fazer não é motivo para afrouxar o limite.**

Isso expôs uma limitação real: um limite de `-20%` único para todos os
ativos deixa a comparação quase sem sentido para ativos com volatilidade
estrutural bem maior (PETR4.SA, VALE3.SA vs SPY, AAPL) -- a tabela vira
"o que aconteceu nas primeiras semanas" em vez de uma avaliação dos 5,5
anos inteiros. Adicionado `ASSET_CIRCUIT_BREAKER_OVERRIDES` (mesmo padrão
de `ASSET_COST_OVERRIDES`), permitindo calibrar por ativo -- mas
**deliberadamente vazio por padrão**. Não escolhi um valor "melhor" para
PETR4.SA depois de ver que -20% cortava cedo, porque isso seria a mesma
armadilha de p-hacking já rejeitada desde a Etapa 0 (seed shopping),
só que em parâmetro de risco em vez de seed. Se decidir customizar,
documente a justificativa (volatilidade estrutural do ativo) antes de
rodar de novo, não depois de ver o número.

**Segundo achado**: `binario_risk_managed` e `buy_and_hold_risk_managed`
deram retorno idêntico em 2 dos 4 ativos (AAPL, PETR4.SA). Investigação em
duas rodadas:
- **1ª tentativa**: diagnóstico medindo média/desvio-padrão do sinal
  binário sobre os 1386 dias inteiros da janela de comparação. Resultado
  real: desvio de 0,28 (AAPL) e 0,42 (PETR4.SA) -- **o sinal claramente
  varia**, refutando a hipótese inicial de "sinal constante". Isso não
  bateu com os retornos idênticos observados.
- **Causa real, achada ao reconsiderar**: o circuit breaker corta a
  atividade muito cedo nesses dois ativos (169 de 1386 dias em AAPL, só
  37 na PETR4.SA). O diagnóstico media a série inteira, mas só os
  primeiros ~37-169 dias importam de verdade para o retorno -- e o sinal
  pode variar bastante *depois* desse ponto, num período que nenhuma das
  duas estratégias chega a alcançar (já estão flat pelo circuit breaker).
  Corrigido: diagnóstico agora mede só a janela ativa (`_active_window_
  signal_stats`, extraída como função pura e testada). Reproduzido com
  dados simulados: janela ativa de 12 dias, sinal constante em 1.0 nela
  -- confirma exatamente por que os retornos batem, sem ser coincidência.

122/122 testes passam.

## Investigação: por que o Kelly foi pior que buy-and-hold na PETR4.SA

Depois de confirmadas as duas investigações acima, sobrou uma terceira
pergunta: por que `kelly_risk_managed` teve retorno pior (-13,32%) que
`binario_risk_managed`/`buy_and_hold_risk_managed` (-3,92%) na PETR4.SA,
mesmo o Kelly usando posições bem menores?

**Confirmado com teste controlado** (queda constante de -1,5%/dia): uma
posição de exposição total (1.0) dispara o circuit breaker de carteira
quando o **ativo** cai ~20%. Uma posição diluída (0.2, típica do Kelly)
só dispara quando o ativo cai **~68%** -- porque o limite mede drawdown
da carteira, não do ativo, e uma posição pequena precisa de uma queda
muito maior do ativo para produzir a mesma perda percentual na carteira.
Isso deixa o Kelly exposto (mesmo que pouco) a uma queda bem mais
profunda e ainda em curso, exatamente o que parece ter acontecido na
PETR4.SA (Kelly ficou ativo 9 dias a mais que binário/buy-and-hold).

**Isso não tem um "conserto óbvio"** -- é um trade-off real, não bug de
código isolado. Implementado como opção, não mudança automática:
`run_risk_managed_backtest` ganhou o parâmetro opcional
`asset_circuit_breaker_drawdown_pct` (`None` por padrão), que dispara o
circuit breaker também quando o **ativo** (não só a carteira) cai além do
limite. Testado no cenário controlado: com a opção ligada, uma posição
diluída saiu em 51 dias em vez de ficar exposta o backtest inteiro,
melhorando o retorno de -8,09% para -3,08% no cenário de teste.

`ASSET_LEVEL_CIRCUIT_BREAKER_PCT` em `config.py` fica `None`
(desligado) por padrão -- **decisão deliberada, não escolhida para
"consertar" o número da PETR4.SA**. Ativar isso é uma troca real: uma
estratégia que dilui exposição *porque* espera volatilidade continuada
pode ser cortada antes dessa diluição compensar. Se você quiser ativar,
defina um valor e documente o motivo antes de rodar de novo -- mesma
disciplina já aplicada ao `ASSET_CIRCUIT_BREAKER_OVERRIDES`.

125/125 testes passam.

## Etapa 5 - Backtest integrado

Adiciona Sortino ratio e segmentacao por regime (bull vs 3 janelas de
crise conhecidas -- Q4 2018, COVID Q1 2020, 2022 -- definidas ANTES de
rodar, nao ajustadas depois do resultado) em cima do mesmo pipeline
ja validado na Etapa 4. `backtest/regime.py` fatiadas os retornos JA
REALIZADOS de um backtest continuo (nao re-roda a simulacao sequencial
numa janela descontinua -- quebraria o rastreio de drawdown/entrada).

Rodar: `python run_stage5_integrated_backtest.py`

## Etapa 6 - Paper trading diario

Mudanca estrutural real: tudo ate a Etapa 5 roda uma vez e acaba. Paper
trading precisa rodar todo dia, guardando estado entre execucoes
separadas do processo Python.

**Refactor que viabilizou isso sem duplicar codigo**: `risk/managed_
backtest.py` foi reestruturado para expor `RiskState` (dataclass
serializavel) + `step_risk_managed_backtest` (avanca a simulacao em
UM dia). O motor em lote (`run_risk_managed_backtest`, usado nas Etapas
4-5) virou um loop fino em cima dessa mesma funcao -- garantido por
teste (`test_step_by_step_matches_batch_result`) que as duas formas de
rodar produzem resultado identico. Sem isso, backtest e paper trading
arriscariam divergir silenciosamente, o mesmo problema ja encontrado
uma vez entre `run_stage3_model.py` e `run_stage3_holdout_evaluation.py`.

**Modelo congelado**: no primeiro dia que roda para um ativo, treina o
modelo com TODOS os dados reais disponiveis ate aquele ponto (nao ha mais
"futuro" a proteger -- hoje E a fronteira prospectiva) e salva via
`joblib` em `paper_trading/models/`. Nunca retreina sozinho depois disso
-- isso e Etapa 7, deliberadamente separada, para que retreino so
aconteca por decisao consciente.

**Estado**: um CSV por ativo em `paper_trading/state/`, append-only. O
proprio CSV E o estado -- a ultima linha tem tudo necessario pra
reconstruir e continuar no dia seguinte. Idempotente: rodar duas vezes no
mesmo dia nao duplica nada (checado por data).

**Escopo**: os 4 ativos reais (decisao explicita do usuario, "para
parecer bem real").

Rodar manualmente: `python run_stage6_paper_trading_daily.py`

**Execucao automatica -- duas opcoes:**

1. **GitHub Actions (recomendado)**: `.github/workflows/paper_trading_daily.yml`
   roda o script sozinho, todo dia util as 19h (horario de Brasilia), nos
   servidores do GitHub -- nao depende de nenhum computador do usuario
   estar ligado ou logado. O workflow commita as atualizacoes de estado
   (`paper_trading/state/*.csv`, `paper_trading/models/*.joblib`) de volta
   no repositorio automaticamente. Configurado apos um problema real
   identificado: rodar isso num computador com acesso incerto (ex: maquina
   de laboratorio) arrisca buracos na sequencia diaria, o que invalida a
   comparacao backtest-vs-real (a metrica principal desta etapa).
2. **Windows Task Scheduler** (`schedule_paper_trading.ps1`): alternativa
   local, so funciona enquanto aquele computador especifico estiver ligado
   e logado todo dia -- mantida no projeto como opcao, mas o GitHub
   Actions e a forma primaria agora.

Ambas as opcoes escrevem no MESMO formato de estado (`paper_trading/state/`)
-- nao rode as duas ao mesmo tempo para o mesmo ativo, ou o log pode
receber duas linhas para o mesmo dia (o script e idempotente por *processo*,
nao protegido contra duas maquinas diferentes rodando no mesmo horario).

**Metrica principal desta etapa** (ainda nao automatizada): comparar o
retorno realizado no log do paper trading contra o que um backtest
normal (Etapas 4/5) preveria para as mesmas datas, usando o mesmo modelo
congelado. Divergencia grande importa mais que o retorno em si -- costuma
indicar bug no backtest que passou despercebido, nao "o mercado mudou".
So faz sentido comparar depois de dias/semanas de dados acumulados.

138/138 testes passam.

## Melhorias do paper trading: resumo automatico, dado parado, consistencia, dashboard

Depois de conferir manualmente os primeiros ~8 dias reais de paper
trading (matematicamente corretos), quatro melhorias foram adicionadas:

**Achado critico durante a construcao da checagem de consistencia**: o
teste que compara o motor em lote contra o script diario (que deveriam
ser matematicamente identicos, ja que os dois chamam a mesma
`step_risk_managed_backtest`) **divergiu** logo na primeira tentativa --
0,54% de diferenca relativa, bem acima da tolerancia. Investigado ate a
causa raiz: `_today_target_position` usava `build_dataset` (que exige
rotulo) para extrair a linha de "hoje", mas `build_forward_return_label`
sempre marca as ultimas `horizon_days` linhas como NaN (falta preco
futuro) -- e o dropna descartava a linha de hoje mesmo com
`horizon_days=1`, silenciosamente pegando ONTEM em vez de HOJE. **Todo
dia desde o lancamento da Etapa 6 calculou a posicao com um dia de
atraso nos dados.** Corrigido usando `build_features` diretamente (sem
depender de rotulo, que so faz sentido para treino, nunca para
inferencia do dia atual). Confirmado por teste que reproduz o cenario:
com a correcao, motor em lote e script diario batem exatamente. Isso e
o proprio objetivo da Etapa 6 funcionando como projetado -- a
comparacao backtest-vs-real achou um bug real na primeira vez que foi
construida.

**`paper_trading/report.py`**: resumo por ativo (dias, retorno
acumulado, equity, avisos) e deteccao de dado parado --
`STALENESS_WARNING_DAYS=5` cobre um fim de semana + folga de um feriado;
acima disso, avisa explicitamente em vez de tratar "sem dado novo"
sempre da mesma forma (que esconderia uma falha real do `yfinance` por
tras de um silencio identico ao de um fim de semana comum).

**`paper_trading/consistency_check.py`**: a checagem central da Etapa 6
-- roda o motor em lote sobre o mesmo periodo do log real, usando o
MESMO modelo congelado, e compara equity dia a dia. Ja provou o proprio
valor (achou o bug acima).

**`paper_trading/dashboard.py` + `run_stage6_report.py`**: dashboard
HTML autocontido (grafico de curva de capital embutido em base64, sem
precisar de servidor nem gerar arquivo separado por ativo) com tabela
de status, resumo e resultado da checagem de consistencia por ativo.
Gerado automaticamente pelo workflow do GitHub Actions apos cada
atualizacao diaria, commitado junto (`paper_trading/dashboard.html`) --
baixa esse arquivo e abre no navegador para ver o progresso, sem
precisar copiar CSV nenhum.

146/146 testes passam.

## Etapa 7 - Deteccao de drift e retreino

Tres perguntas separadas, porque tem causas e remedios diferentes:
1. **Divergencia de implementacao** (`consistency_check.py`): live bate com o motor em lote
   usando o mesmo modelo? Divergencia = BUG, nao drift.
2. **Drift de features** (`paper_trading/drift.py`): as entradas de hoje parecem diferentes
   do treino? Teste contra uma distribuicao nula EMPIRICA (janelas moveis do periodo de
   referencia, que preserva a autocorrelacao -- o p-valor ingenuo alarmaria muito mais que
   5%; coberto por teste de taxa de falso alarme). Reporta no maximo "ATENCAO".
3. **Drift de desempenho**: Brier versus um previsor ingenuo (taxa-base) em resultados
   MADUROS, IC por bootstrap de blocos. Veredito so com >= `DRIFT_MIN_EFFECTIVE_SAMPLES`
   amostras efetivas (dias maduros / 20, ja que rotulos de 20 dias se sobrepoem; 6 ~ 6
   meses). Antes disso o veredito e **INSUFICIENTE, por construcao** -- nao e defeito.
   Com menos de 2 blocos independentes o IC e "indisponivel" (achado real: antes colapsava
   num ponto e parecia preciso).

Limiares fixados em `config.py` ANTES de ver qualquer resultado de drift.

**Retreino** (`paper_trading/retrain.py`) nunca e automatico e nunca sobrescreve nada:
- *Challenger*: treinado so com precos ate um corte; avaliado contra o modelo vigente na MESMA
  janela posterior, fora da amostra para os dois (sem vazamento, testado).
- *Promocao*: passo manual (`--promote TICKER`) que registra uma NOVA versao
  (`paper_trading/models/registry.csv`) valida a partir do dia seguinte -- o historico nunca e
  re-atribuido a outro modelo (tentar retroagir e recusado). Exige veredito
  CHALLENGER_MELHOR (IC acima de zero); `--force-promote` existe mas e explicito e fica
  marcado no relatorio. Provar que retreinar ajudou leva meses de dado -- propriedade do
  problema, nao algo a "afrouxar".
- O script diario e a checagem de consistencia agora usam a versao vigente em cada data.

```
python run_stage7_drift_check.py                  # relatorio (paper_trading/drift_report.md)
python run_stage7_drift_check.py --retrain-eval   # + comparacao challenger
python run_stage7_drift_check.py --promote SPY    # promove (so com evidencia)
```
Workflow semanal (`drift_check_weekly.yml`, segundas): so relatorio. Cadencia fixa de proposito:
checar drift todo dia e um teste repetido (multiplas comparacoes).

162/162 testes passam.

## Próximos passos

A próxima etapa e a **Etapa 7 - Deteccao de drift e retreino periodico**,
mas so deve comecar apos confirmacao explicita -- e apos paper trading
ter rodado tempo suficiente para ter dados reais de divergencia pra
comparar.


## Auditoria pos-Etapa 7 (baselines justos, ablacao, precos canonicos)

- **Precos logados diferiam dos re-baixados em ate 2,7%** (yfinance reajusta o historico por
  dividendos/revisoes). Consequencias: replay inexato na checagem de consistencia e dividendos
  contabilizados como queda de preco. Corrigido com `paper_trading/prices.py`: serie
  canonica encadeada e so de acrescimo (razao de preco dentro de um mesmo download), usada por
  decisao, log, consistencia e drift. Migracao a partir do log existente preserva os closes
  logados exatamente. Teste de integracao com reajuste retroativo aleatorio todo dia.
- **Bug na migracao** (achado lendo o fluxo): o primeiro run criaria a serie sem as barras
  posteriores ao ultimo dia logado, deixando um buraco. Corrigido e coberto por teste.
- **Baselines justos + ablacao** (`run_baseline_ablation.py`): ver `leitura_baselines.md`.
  Conclusao central: sem evidencia de informacao no modelo; o ganho do Kelly e dimensionamento.
- Cache das previsoes walk-forward (`models/prediction_cache.py`, chave = hash dos precos +
  config relevante: nunca serve resultado velho). CI de testes (`tests.yml`). Resumo diario
  (`daily_summary.txt`) + notificacao Telegram opcional (so com secrets; sem eles e ignorada).
- Passo do Telegram saiu com YAML invalido na primeira versao (linha na coluna 0 dentro de um
  bloco `run: |`); pego por validacao local antes do envio.
- README dividido: `README.md` (como rodar) + este arquivo (historico). `docs/CRITERIOS.md`:
  criterios de sucesso/parada pre-registrados.
- 175 testes.
