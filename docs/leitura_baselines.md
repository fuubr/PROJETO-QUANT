# Leitura dos resultados de `run_baseline_ablation.py`

Numeros completos em `analise_baselines.md` (gerado pelo script, com a fonte da taxa de caixa
em cada secao). Periodo 2018-2023, fora da amostra, holdout intocado. Caixa = taxa livre de
risco **real** (BCB CDI para .SA, FRED T-bill 3m para os demais); `sharpe_excesso_rf` e o
Sharpe correto (sobre o excedente ao caixa).

## O que os dados dizem
1. **Sem evidencia de que o modelo agrega informacao.** Modelo menos ablacao (mesmo Kelly e
   stops, probabilidade trocada pela taxa-base causal): SPY +2,0% a.a. (p=0,12), AAPL +0,08%,
   PETR4 -0,4%, VALE3 +0,12%. Nenhum significativo; em 3 de 4 e praticamente zero.
2. **Os stops/timing nao superam uma diluicao constante.** Modelo menos exposicao igualada:
   +0,4%, -1,1%, -1,1%, -2,5% a.a. Nenhum significativo, mas o sinal e negativo em 3 de 4.
3. **Menos risco, nao mais retorno por risco.** Sharpe de excesso do Kelly 0,52 / 0,62 / -0,25 /
   0,00 contra buy-and-hold puro 0,53 / 0,93 / 0,67 / 0,41 (SPY/AAPL/PETR4/VALE3): empate em SPY,
   pior nos outros 3. Retorno absoluto bem menor (SPY +18% contra +79%): exposicao media ~15-22%.
4. **Nos dois ativos brasileiros a estrategia rendeu menos que o caixa puro.** Retorno incluindo
   juros do caixa: PETR4 +40,9% e VALE3 +45,8%, contra +50,7% do CDI sozinho na mesma janela.
   Em SPY/AAPL ficou acima do caixa (+28% e +33% contra +10,4%).
5. **O benchmark simples mais forte e o vol-target** (exposicao = 10% / vol realizada, sem
   modelo, 1 parametro fixado antes de rodar): Sharpe de excesso 0,62 / 1,27 / 0,74 / 0,33,
   maior que o buy-and-hold em 3 de 4. Sem teste de significancia ainda.
6. **A comparacao antiga era enganosa.** "Kelly +18,9% contra buy-and-hold -2,8%" usava um
   buy-and-hold COM circuit breaker, que ficou fora do mercado depois da queda. O buy-and-hold
   puro fez +79% na mesma janela.

## Correcao desta versao
A versao anterior usava taxas de caixa ASSUMIDAS (10% BR, 2,5% US) e um "Sharpe com juros" que nao
era sobre o excedente. As taxas reais na janela rendem ~7,4% a.a. (BR) e ~1,8% a.a. (US): o
caixa estava superestimado, e valores como "Sharpe 7-25" na PETR4 eram artefato. Corrigido.

## Ressalvas (leia antes de concluir qualquer coisa)
- **PETR4**: ativa so 2,8% do periodo (o circuit breaker de ativo disparou em 2018 e nao ha
  regra de religar). O resultado do modelo nessa linha e quase vazio; o que sobra e caixa.
  O breaker sem religamento e a decisao de design mais cara do projeto.
- Um unico periodo, em grande parte de alta (AAPL +335%); 4 ativos correlacionados
  (SPY~AAPL, PETR4~VALE3 = ~2 amostras independentes); varias comparacoes feitas.
- Os numeros deste periodo ja foram vistos muitas vezes: servem para descartar hipoteses,
  nao para escolher parametros. Qualquer ideia nova (ex.: vol-target) e **hipotese a
  pre-registrar e testar adiante**, nao uma "descoberta" destes dados.

## Implicacao
O paper trading e o unico teste realmente fora da amostra. Os criterios em `CRITERIOS.md`
foram escritos para que, se o modelo nao tiver informacao, isso apareca como "parar" e nao
como "mais uma rodada de ajuste".
