# Criterios de sucesso e de parada (pre-registro)

Fixados ANTES de existirem dias suficientes de paper trading limpo (o primeiro dia
gerado pelo codigo corrigido e 2026-10-01). Servem para impedir a armadilha de sempre:
decidir o que conta como "funcionou" depois de ver o resultado. **Editar este arquivo
depois que os dados chegarem e mudar a regua -- o historico do git mostra quando e por que.**
(Rascunho proposto por Claude; Guilherme deve revisar e ajustar AGORA, nao depois.)

## Quando avaliar
Primeira avaliacao formal: **>= 120 dias uteis limpos** (~6 meses, ~6 amostras efetivas
de janelas de 20 dias). Antes disso, relatorios dizem INSUFICIENTE e nada e decidido.

## Sucesso (todos precisam valer)
1. **Implementacao**: checagem de consistencia OK em todos os dias limpos.
2. **O modelo tem informacao** (H1): vantagem de Brier sobre o previsor ingenuo com IC95%
   (bootstrap de blocos) **inteiramente acima de zero**. "Nao pior" nao basta.
3. **O modelo agrega valor sobre o dimensionamento**: retorno diario do
   `kelly_modelo` menos `ablacao_sem_modelo` e menos `exposicao_igualada` positivo, com
   Newey-West p < 0,05 (Bonferroni x2) -- `run_baseline_ablation.py` com dados ao vivo.
4. (aprovado em 2026-10-14) A estrategia final nao pode ter Sharpe inferior ao do
   baseline `vol_target` (sem modelo) no mesmo periodo limpo -- senao a complexidade extra
   nao se justifica. Surgiu da analise de baselines (ver `leitura_baselines.md`).
5. Resultado nao depende de um unico ativo: vale em pelo menos 2 dos 4, lembrando que
   SPY~AAPL e PETR4~VALE3 se movem juntos (sao ~2 amostras independentes, nao 4).

## Hipoteses adicionais H2 e H3 (pre-registradas em 2026-10-14)

Registradas ANTES de existir qualquer dia de paper trading delas e antes de calcular H3 em
qualquer dado historico. Parametros em `config.py`/`strategies/rules.py`, congelados aqui.
Rodam diariamente ao lado do modelo, mesmo preco canonico e mesmo motor, logs proprios
(`paper_trading/state/<ATIVO>__<regra>.csv`), **sem stop-loss nem circuit breaker**.

- **H2 vol-target**: exposicao = 10% a.a. / vol realizada (20 retornos diarios, anualizada
  com raiz de 252), limitada a [0, 1]. Sem modelo.
- **H3 tendencia + vol-target**: igual a H2 enquanto o retorno dos ultimos 252 dias for
  positivo; caixa caso contrario. Uma regra so, nao duas.
- Referencia: `buyhold` (exposicao 1,0), rodando no mesmo periodo e com os mesmos custos.

**Teste de cada uma** (na primeira avaliacao, >= 120 dias uteis do log dela):
Sharpe da regra maior que o do `buyhold` do mesmo ativo pelo teste de Memmel
(`backtest/deflated_sharpe.py`), p < 0,05 / 3 (Bonferroni: H1 modelo, H2, H3), em pelo
menos 2 dos 4 ativos sendo pelo menos 1 de cada par (SPY/AAPL e PETR4/VALE3).
O Deflated Sharpe de cada uma e reportado junto, com o numero de tentativas declarado;
DSR < 0,95 rebaixa qualquer "sucesso" para "inconclusivo".

**Se falharem**: a regra e descartada como esta. Sem variar a janela (20/252), o alvo (10%)
ou o limite -- isso seria uma nova hipotese, com novo pre-registro e novos dados.

## Hipotese H4: ranking momentum + baixa volatilidade na B3 (pre-registrada em 2026-10-14)

Registrada ANTES de calcular qualquer numero desta regra (o commit deste texto precede o
codigo). Muda a pergunta: em vez de "esta acao sobe?", **"quais acoes vao render mais que as
outras?"**. Parametros sao os padroes da literatura (Jegadeesh-Titman 12-1; low-vol de
Ang et al. / Baker et al.), nao escolhidos olhando dados deste projeto.

- **Universo (congelado)**: 42 acoes da B3 em `config.H4_UNIVERSE` -- as da lista
  pre-definida com historico no yfinance. Vies de sobrevivencia declarado: 8 nomes da lista
  inicial nao tem historico (deslistados/renomeados) e foram excluidos; o benchmark usa o
  MESMO universo, o que neutraliza parte do vies na comparacao, nao no retorno absoluto.
- **Elegivel** no rebalanceamento: >= 273 pregoes de historico.
- **Sinal** (fechamento do ultimo pregao do mes, t):
  momentum 12-1 = close[t-21] / close[t-252] - 1; vol = desvio-padrao dos ultimos 63
  retornos diarios. Score = percentil(momentum) + percentil(-vol).
- **Carteira**: as 10 maiores notas, pesos iguais, rebalanceamento mensal, executada no
  fechamento de t (sinal usa so dados ate t). Custo: 10 bps sobre o valor negociado.
- **Benchmark justo**: todas as elegiveis com peso igual, mesmo rebalanceamento e custo.

**Teste 1 -- desenvolvimento (2012-01 a 2023-12)**, primeira e unica vez:
diferenca mensal H4 - benchmark com media > 0 e Newey-West **p < 0,05 / 4** (Bonferroni:
H1, H2, H3, H4), **e** Sharpe de excesso (CDI) do H4 maior que o do benchmark.
**Controle sintetico**: a mesma regra em 42 random walks independentes NAO pode passar.
**Teste 2 -- holdout (2024-01 em diante)**, so se o Teste 1 passar, uma unica vez:
diferenca mensal media > 0 com p < 0,05 (unilateral). E o holdout desta regra; nao afeta
o holdout reservado ao pipeline do modelo (H1), que continua intocado.
**Teste 3 -- paper trading** ao vivo, so se os dois passarem.

**Se falhar em qualquer etapa**: descartada como esta. Sem trocar 10 por 5 acoes, 12 por 6
meses, 63 por 20 dias, pesos ou universo -- cada variacao seria nova hipotese, e o periodo
2012-2023 nao pode ser reusado para ela.

## Hipotese H5: valor + qualidade com balancos da CVM (pre-registrada em 2026-10-14)

Registrada ANTES de calcular qualquer sinal ou retorno (so a estrutura dos arquivos da CVM
foi inspecionada). Motivo: H4 so usava preco; H5 traz informacao que o preco nao tem.

- **Universo**: o da H4 menos PETR3 (uma classe por empresa) = 41 acoes.
- **Dados (CVM, dados abertos)**: DFP anual consolidado (individual se nao houver):
  lucro liquido do exercicio e patrimonio liquido. Acoes emitidas: FRE (capital
  integralizado, total). Cada dado so vale a partir do pregao **seguinte** ao `DT_RECEB`.
  Acoes ajustadas por desdobramentos/grupamentos/bonificacoes (yfinance) para a base do
  preco; units divididas pelo numero de acoes por unit (SANB11 2, TAEE11 3, KLBN11 5,
  ENGI11 5). Valor de mercado = preco x acoes totais da empresa (aproximacao declarada para
  quem tem ON e PN com precos diferentes).
- **Sinal**: E/P = lucro anual / valor de mercado; ROE = lucro / PL. PL <= 0 ou dado
  ausente -> inelegivel. Score = percentil(E/P) + percentil(ROE).
- **Carteira**: igual H4 -- top 10, peso igual, mensal, 10 bps. Benchmark: todas as
  elegiveis (mesmo filtro de dados), peso igual.
- **Teste 1 (2012-01 a 2023-12)**: diferenca mensal media > 0 com Newey-West
  **p < 0,05 / 5** (H1-H5) e Sharpe de excesso (CDI) maior que o do benchmark.
  **Controle sintetico**: precos trocados por random walks (mesma vol, mesmos balancos
  reais) -- nao pode passar alem do acaso.
- **Teste 2 (holdout 2024+)** uma vez, so se o Teste 1 passar; depois paper trading.
- **Vieses declarados**: sobrevivencia (como H4); a CVM publica a versao mais recente de
  cada documento, entao reapresentacoes posteriores vazam um pouco de futuro.
- **Se falhar**: descartada sem variar indicadores, pesos, N ou universo.

## Parada
- Veredito **DEGRADADO** em qualquer relatorio semanal com amostra suficiente: o sinal do
  modelo sai do paper trading (fica so o overlay de risco ou nada).
- Criterio 2 ou 3 falha na primeira avaliacao: o modelo e descartado como fonte de sinal.
  **Sem** "mais uma rodada de ajuste de features/hiperparametros": o periodo 2015-2023 ja
  foi consultado muitas vezes; qualquer ajuste nele e overfitting por construcao.

## Proibido
- Ajustar features, hiperparametros ou limiares deste arquivo / do `config.py` olhando
  resultados de paper trading ou do holdout.
- Gastar o holdout (2024+) antes de congelar o pipeline final; e uma vez so.
- Promover retreino sem o gate de evidencia (`--force-promote` conta como decisao
  consciente e deve ser justificada por escrito aqui).
