# Leitura dos resultados de `run_baseline_ablation.py`

Numeros completos em `analise_baselines.md` (gerado pelo script). Periodo 2018-2023,
fora da amostra, holdout intocado.

## O que os dados dizem
1. **Sem evidencia de que o modelo agrega informacao.** Modelo menos ablacao (mesmo Kelly e
   stops, probabilidade trocada pela taxa-base causal): SPY +2,0% a.a. (p=0,13), AAPL +0,08%,
   PETR4 -0,4%, VALE3 +0,03%. Nenhum significativo; em 3 de 4 e praticamente zero.
2. **Os stops/timing nao superam uma diluicao constante.** Modelo menos exposicao igualada
   (mesma exposicao media, sem modelo e sem stops): +0,4%, -1,1%, -1,1%, -2,6% a.a.
   Nenhum significativo, mas o sinal e negativo em 3 de 4.
3. **Menos risco, nao mais retorno por risco.** Sharpe do Kelly 0,57 / 0,65 / -0,24 / 0,12
   contra buy-and-hold puro 0,61 / 0,99 / 0,82 / 0,59 (SPY/AAPL/PETR4/VALE3): pior em 4 de 4.
   O retorno absoluto fica bem menor (SPY +18% contra +79%) porque a exposicao media e ~15-22%.
4. **O benchmark simples mais forte e o vol-target** (exposicao = 10% / vol realizada, sem
   modelo, 1 parametro fixado antes de rodar): Sharpe 0,74 / 1,33 / 0,92 / 0,53, maior que o
   buy-and-hold em 3 de 4 e com drawdown menor. Sem teste de significancia ainda.
5. **A comparacao antiga era enganosa.** "Kelly +18,9% contra buy-and-hold -2,8%" usava um
   buy-and-hold COM circuit breaker, que ficou fora do mercado depois da queda. O buy-and-hold
   puro fez +79% na mesma janela.

## Ressalvas (leia antes de concluir qualquer coisa)
- **PETR4**: a estrategia ficou ativa so 2,8% do periodo (o circuit breaker de ativo disparou
  em 2018 e nao ha regra de religar). Os resultados do modelo nessa linha sao quase vazios, e
  as colunas "c/ juros caixa" ali sao artefato (~100% caixa a 10% ASSUMIDOS; Sharpe 7-25 nao
  significa nada). O breaker sem religamento e a decisao de design mais cara do projeto.
- Um unico periodo, em grande parte de alta (AAPL +335%); 4 ativos correlacionados
  (SPY~AAPL, PETR4~VALE3 = ~2 amostras independentes); varias comparacoes feitas.
- Os numeros deste periodo ja foram vistos muitas vezes: servem para descartar hipoteses,
  nao para escolher parametros. Qualquer ideia nova (ex.: vol-target) e **hipotese a
  pre-registrar e testar adiante**, nao uma "descoberta" destes dados.

## Implicacao
O paper trading e o unico teste realmente fora da amostra. Os criterios em `CRITERIOS.md`
foram escritos para que, se o modelo nao tiver informacao, isso apareca como "parar" e nao
como "mais uma rodada de ajuste".
