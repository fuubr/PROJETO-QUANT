# Backlog de fontes de dados futuras

Este documento existe para registrar candidatos a fontes de dados adicionais
sem incorporar nenhum deles precipitadamente na Etapa 1. A regra do projeto
é: nenhuma fonte de dado entra sem uma hipótese específica de por que ela
deveria ter poder preditivo, e sem validação isolada contra o ativo
sintético como controle.

Adicionar dados sem hipótese é o análogo, no espaço de features, do que
"seed shopping" é no espaço de aleatoriedade: aumenta a chance de achar
correlação espúria que parece sinal.

Cada item abaixo deve virar uma nota `Template-Decisao` no Obsidian quando
for efetivamente avaliado, registrando: hipótese, resultado do teste contra
o sintético, e decisão (aceito/rejeitado/adiado).

## Candidatos

### Dados macroeconômicos (ex: FRED — taxa de juros, CPI, desemprego)
Hipótese candidata: regimes macro (juros subindo vs caindo, por exemplo)
podem mudar o comportamento de risco/retorno de ativos de forma
suficientemente persistente para servir de feature de regime, não de sinal
direcional direto.

### Volume e liquidez (order book, volume relativo)
Hipótese candidata: padrões de volume anômalo podem preceder movimentos de
preço por refletirem informação de fluxo antes que o preço se ajuste
completamente.

### Dados fundamentalistas (earnings, múltiplos)
Hipótese candidata: relevante principalmente para holding periods mais
longos; baixa relevância esperada para estratégias de curto prazo. Validar
o horizonte de holding antes de considerar.

### Sentimento (notícias, redes sociais)
Hipótese candidata: mais arriscado que os anteriores — alto risco de
overfitting e de vazamento de informação futura (lookahead) se a coleta não
for cuidadosamente alinhada ao timestamp real de disponibilidade da
informação. Exigir validação extra de ausência de lookahead antes mesmo de
testar poder preditivo.

## Critério de entrada (repetir para cada fonte candidata)

1. Qual é a hipótese específica? (não "pode ajudar" — o mecanismo causal
   esperado)
2. A feature derivada dessa fonte tem timestamp real de disponibilidade
   compatível com o momento de decisão do backtest? (evitar lookahead)
3. Testada isoladamente contra o ativo sintético: produz falso sinal no
   controle? Se sim, a feature (ou sua implementação) está com problema.
4. Registrar decisão em `Template-Decisao` antes de mesclar ao pipeline
   principal.
