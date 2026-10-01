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
2. **O modelo tem informacao**: vantagem de Brier sobre o previsor ingenuo com IC95%
   (bootstrap de blocos) **inteiramente acima de zero**. "Nao pior" nao basta.
3. **O modelo agrega valor sobre o dimensionamento**: retorno diario do
   `kelly_modelo` menos `ablacao_sem_modelo` e menos `exposicao_igualada` positivo, com
   Newey-West p < 0,05 (Bonferroni x2) -- `run_baseline_ablation.py` com dados ao vivo.
4. (proposta nova, requer aprovacao) A estrategia final nao pode ter Sharpe inferior ao do
   baseline `vol_target` (sem modelo) no mesmo periodo limpo -- senao a complexidade extra
   nao se justifica. Surgiu da analise de baselines (ver `leitura_baselines.md`).
5. Resultado nao depende de um unico ativo: vale em pelo menos 2 dos 4, lembrando que
   SPY~AAPL e PETR4~VALE3 se movem juntos (sao ~2 amostras independentes, nao 4).

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
