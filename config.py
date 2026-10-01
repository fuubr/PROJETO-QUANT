"""Central project configuration."""

SEED = 42
SYNTHETIC_START_PRICE = 100.0
SYNTHETIC_PERIODS = 252 * 5
SYNTHETIC_DAILY_VOL = 0.01
SYNTHETIC_START_DATE = "2015-01-01"

# --- Etapa 1: dados reais e motor de backtest ---

# Ativos reais cobrindo mercado americano e brasileiro (B3), para validar
# que o pipeline funciona em qualquer um dos dois. Tickers B3 usam o sufixo
# ".SA" exigido pelo yfinance. Cada ativo roda em sua propria moeda -- nao
# ha conversao cambial nem combinacao de portfolio entre mercados nesta
# etapa, cada serie eh independente.
REAL_ASSET_TICKERS = ["SPY", "AAPL", "PETR4.SA", "VALE3.SA"]

# Janela de backtest. Inicio fixo, fim dinamico (ate a data atual), resolvido
# em tempo de execucao por data.real.resolve_end_date(). None aqui significa
# "hoje". Atencao: por nao reservar um periodo held-out, esta configuracao so
# e apropriada enquanto nenhuma etapa estiver ajustando parametros nos dados
# (a partir do momento em que houver otimizacao de regra/modelo, uma janela
# final deve ser reservada como fora da amostra).
BACKTEST_START_DATE = "2015-01-01"
BACKTEST_END_DATE = None

INITIAL_CAPITAL = 100_000.0

# Custos de transacao default, usados quando um ativo nao tem override em
# ASSET_COST_OVERRIDES. Valor de referencia generico -- ajuste para a
# corretora/mercado real antes de qualquer decisao com dinheiro real.
TRANSACTION_FEE_BPS = 0.0
SLIPPAGE_BPS = 5.0

# Overrides de custo por ticker, permitindo usar o mesmo motor em qualquer
# mercado/corretora sem mudar codigo. Nao presentes na chave = usa o default
# acima. Valores da B3 aqui sao apenas um ponto de partida (emolumentos +
# spread tipicamente maior que large caps americanas) -- validar contra a
# corretora real antes de qualquer decisao com dinheiro real.
ASSET_COST_OVERRIDES: dict[str, tuple[float, float]] = {
    "PETR4.SA": (0.0, 10.0),
    "VALE3.SA": (0.0, 10.0),
}

# Probabilidade de estar posicionado nos baselines de entrada aleatoria
# (ver backtest/baselines.py). Fixa em 50% para nao favorecer nenhum lado.
RANDOM_ENTRY_PROBABILITY = 0.5

# Duracao media de posicao (em dias) do baseline de entrada aleatoria com
# persistencia -- um piso mais realista de turnover que o i.i.d. puro.
RANDOM_ENTRY_AVG_HOLDING_DAYS = 20

# --- Etapa 1/2: estrategia de cruzamento de medias moveis ---
# "UMA estrategia simples sem IA" exigida na Etapa 1, reutilizada como
# terceiro baseline obrigatorio na Etapa 2. Janelas sao valores de
# referencia comuns, NAO otimizados/ajustados nos dados -- fazer isso
# exigiria a infraestrutura de walk-forward reservada para a Etapa 3, para
# nao contaminar a validacao com overfitting.
SMA_FAST_WINDOW = 20
SMA_SLOW_WINDOW = 50

# --- Etapa 3: modelo de IA com probabilidade calibrada ---

# Ativos usados nesta etapa. Expandido para todos os 4 ativos reais do
# projeto (Etapa 1/2) apos o gate estatistico no sintetico ter passado
# de forma agregada e robusta com SPY sozinho.
STAGE3_TICKERS = ["SPY", "AAPL", "PETR4.SA", "VALE3.SA"]

# Horizonte de previsao: probabilidade de retorno positivo em N dias uteis
# a frente. Define o rotulo (label) do modelo, nao uma feature.
PREDICTION_HORIZON_DAYS = 20

# Janelas usadas nas features de preco (todas causais/trailing, sem
# lookahead). Mantidas poucas de proposito -- mitigacao de overfitting
# citada na filosofia original do projeto.
FEATURE_MOMENTUM_WINDOWS = (5, 20, 60)
FEATURE_VOLATILITY_WINDOW = 20
FEATURE_SMA_DISTANCE_WINDOW = 50

# Walk-forward: janela de treino expansiva (todo dado disponivel ate a data
# de corte), teste em blocos subsequentes. purge_days remove do treino
# qualquer observacao cujo rotulo (que olha PREDICTION_HORIZON_DAYS a
# frente) alcancaria o periodo de teste -- sem isso, o rotulo vazaria
# informacao do futuro para dentro do treino mesmo com os indices "certos".
WALK_FORWARD_TEST_WINDOW_DAYS = 126  # ~6 meses uteis por bloco de teste
WALK_FORWARD_MIN_TRAIN_DAYS = 750  # ~3 anos uteis minimos antes do 1o teste
WALK_FORWARD_PURGE_DAYS = PREDICTION_HORIZON_DAYS

# Conjunto held-out final: nunca usado no walk-forward, so avaliado uma vez
# ao final. Ajuste esta data com cautela -- ela so e valida enquanto
# permanecer verdadeiramente "nao vista" durante todo o desenvolvimento do
# modelo.
STAGE3_HOLDOUT_START_DATE = "2024-01-01"

# Fracao do final de cada janela de treino reservada para calibrar a
# probabilidade do XGBoost (CalibratedClassifierCV em modo "prefit",
# cronologico -- nunca embaralhado).
CALIBRATION_FRACTION = 0.2

# Numero de bins usados no reliability diagram.
CALIBRATION_DIAGRAM_BINS = 10

# Numero minimo de observacoes na fatia de calibracao para usar isotonic
# (nao-parametrica, flexivel, mas propensa a "decorar" ruido com poucos
# pontos). Abaixo disso, usa sigmoid (Platt scaling), que tem so 2
# parametros e generaliza melhor com pouco dado -- comum em folds de
# walk-forward iniciais, que tendem a ter janelas de calibracao menores.
CALIBRATION_METHOD_MIN_SAMPLES_FOR_ISOTONIC = 200

# Seeds usadas no gate estatistico do ativo sintetico. Rodar contra uma unica
# seed arrisca o gate passar por sorte numa unica realizacao do random walk;
# todas precisam passar antes de qualquer ativo real ser avaliado.
STAGE3_SYNTHETIC_GATE_SEEDS = (42, 101, 202, 303, 404)

# --- Etapa 4: gestao de risco com sizing dinamico ---

# Fracao do Kelly completo usada no sizing (Kelly fracionario). Kelly
# completo maximiza crescimento geometrico assintotico, mas e extremamente
# sensivel a erro de estimativa de p e b -- um pequeno erro de estimativa
# pode levar a apostas grandes demais. Meio-Kelly (0.5) e a recomendacao
# pratica mais citada (ex: Ed Thorp): metade do crescimento esperado, com
# volatilidade e risco de ruina muito menores. Nao ajustado nos dados --
# escolha de seguranca, nao um hiperparametro otimizado.
KELLY_FRACTION = 0.5

# Janela (em dias, retornos realizados no periodo de desenvolvimento) usada
# para estimar a razao ganho/perda (b no Kelly: media do ganho medio sobre
# a perda media) que alimenta a formula de Kelly. Reestimado antes de cada
# ponto de decisao usando so dados ate aquele ponto (causal).
KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS = 252

# Limites minimos de observacoes de ganho/perda para confiar na estimativa
# de win/loss ratio. Abaixo disso, a posicao sugerida por Kelly e zerada
# (sem dado suficiente para apostar com confianca) em vez de usar uma
# estimativa ruidosa.
KELLY_MIN_WIN_OBSERVATIONS = 20
KELLY_MIN_LOSS_OBSERVATIONS = 20

# Stop-loss por posicao: fecha a posicao (forca flat) se o preco cair mais
# que este percentual desde a entrada, independente do que o modelo disser
# no dia. Protege contra o modelo "segurar" uma posicao perdedora contra
# uma queda continua ate o proximo sinal.
STOP_LOSS_PCT = 0.10

# Circuit breaker: desliga a estrategia (forca flat permanentemente pelo
# resto do backtest) se o drawdown da carteira ultrapassar este percentual.
# Nao ha regra de "religar" -- uma vez acionado, a estrategia fica de fora
# ate o fim do periodo simulado. Simples e conservador de proposito; regras
# de reativacao sao um risco de design mais complexo, fora do escopo desta
# etapa.
CIRCUIT_BREAKER_DRAWDOWN_PCT = 0.20

# Overrides por ativo, mesmo padrao de ASSET_COST_OVERRIDES -- permite
# calibrar o limite do circuit breaker para ativos com volatilidade
# estrutural bem diferente (ex: PETR4.SA/VALE3.SA vs SPY/AAPL), sem mudar
# o valor global. DELIBERADAMENTE VAZIO por padrao: um evento real de
# mercado (ex: PETR4.SA em maio/2018, greve dos caminhoneiros) disparando
# o breaker forte e o sistema funcionando como projetado, nao um motivo
# automatico para afrouxar o limite. Se voce decidir customizar isso para
# algum ativo, faca antes de olhar o proximo resultado, com base na
# volatilidade estrutural do ativo -- nunca depois, para "melhorar" um
# numero que voce ja viu.
ASSET_CIRCUIT_BREAKER_OVERRIDES: dict[str, float] = {}

# Circuit breaker a nivel de ATIVO (nao carteira), opcional -- None por
# padrao (desligado). Achado real durante a Etapa 4: com exposicao diluida
# (Kelly), o circuit breaker de carteira precisa que o ATIVO caia muito
# mais que ASSET_CIRCUIT_BREAKER_DRAWDOWN_PCT antes de disparar (testado:
# posicao de 0.2 precisou o ativo cair ~68% para o breaker de carteira
# disparar em 20%, contra ~20% de queda do ativo para uma posicao 1.0).
# Ativar isso fecha essa brecha, mas troca um risco por outro: uma
# estrategia que dilui exposicao *porque* espera volatilidade continuada
# pode ser cortada antes dessa diluicao compensar. Decisao deliberada, nao
# padrao -- None mantem o comportamento ja validado da Etapa 4.
ASSET_LEVEL_CIRCUIT_BREAKER_PCT: float | None = None

# Override por ativo, mesmo padrao de ASSET_CIRCUIT_BREAKER_OVERRIDES --
# permite testar/ativar o circuit breaker de nivel de ativo em um ativo
# especifico (ex: PETR4.SA) sem mudar o comportamento ja validado dos
# outros. Vazio por padrao. Se preencher, documente o motivo da escolha do
# valor antes de rodar -- nunca depois de ja ter visto o resultado que
# você está tentando "melhorar".
#
# PETR4.SA: -35%, decidido ANTES de rodar o teste, com base na volatilidade
# historica real do papel (~25% ao ano nos ultimos 12 meses, ~45% de longo
# prazo -- fonte: maisretorno.com, consultado em 2026-07). Um limite mais
# apertado (ex: -20%, igual ao de carteira) cortaria com frequencia mesmo
# em oscilacoes normais para esse ativo, dada sua volatilidade estrutural
# alta comparada a SPY/AAPL.
ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES: dict[str, float] = {
    "PETR4.SA": 0.35,
}

# Nivel de confianca usado no calculo de VaR historico (percentil empirico
# dos retornos, nao parametrico -- consistente com a preferencia do projeto
# por metodos robustos a nao-normalidade, ja usada no Newey-West).
VAR_CONFIDENCE_LEVEL = 0.95

# --- Etapa 5: periodos de crise/bull para segmentacao ---
#
# Definidos por eventos macro conhecidos e documentados publicamente,
# ANTES de rodar qualquer backtest segmentado -- nao ajustados depois de
# ver o resultado. "Bull" = tudo no periodo de desenvolvimento que nao cai
# dentro de um desses intervalos.
CRISIS_PERIODS: list[tuple[str, str]] = [
    ("2018-10-01", "2018-12-31"),  # selloff global Q4 2018 (a greve dos
    # caminhoneiros na PETR4.SA, maio/2018, ja e capturada pelo circuit
    # breaker de ativo especifico dela; esta janela cobre o selloff global
    # mais amplo do 4o trimestre)
    ("2020-02-01", "2020-04-30"),  # crash da COVID-19
    ("2022-01-01", "2022-12-31"),  # mercado em baixa por juros/inflacao
]

# --- Etapa 6: paper trading diario ---
#
# Todos os 4 ativos reais, para parecer realista (decisao explicita do
# usuario). O modelo usado aqui e CONGELADO no primeiro dia de paper
# trading (treinado com todos os dados disponiveis ate aquele ponto) e
# NAO e retreinado nas execucoes seguintes -- retreino periodico e
# explicitamente Etapa 7, nao esta etapa. Misturar as duas contaminaria a
# unica validacao prospectiva de verdade que o projeto tem.
PAPER_TRADING_TICKERS = STAGE3_TICKERS

PAPER_TRADING_INITIAL_CAPITAL = INITIAL_CAPITAL

# Numero de dias corridos sem dado novo que ainda e normal (cobre um fim
# de semana comum + folga para um feriado isolado). Acima disso, o script
# diario e o resumo (paper_trading/report.py) avisam explicitamente --
# passar disso sem ninguem notar pode esconder um problema real no feed
# de dados (yfinance fora do ar, ticker delistado, etc.), nao so um
# feriado.
STALENESS_WARNING_DAYS = 5

# --- Etapa 7: deteccao de drift e retreino ---
#
# Todos os limiares abaixo foram fixados ANTES de olhar qualquer resultado
# de drift real -- ajusta-los depois de ver um alerta (ou a falta dele)
# seria a mesma armadilha de p-hacking rejeitada desde a Etapa 0.

# Drift de features: p-valor empirico (contra a distribuicao de janelas
# moveis do periodo de referencia, que preserva a autocorrelacao) abaixo
# disso gera "ATENCAO" -- nunca "drift confirmado" sozinho.
FEATURE_DRIFT_ALERT_P = 0.05
# Menos dias ao vivo que isso: nao ha o que comparar, o veredito e INSUFICIENTE.
DRIFT_MIN_LIVE_DAYS_FEATURES = 10
# Amostras EFETIVAS (dias maduros / horizonte, ja que rotulos de 20 dias se
# sobrepoem) minimas para qualquer veredito de desempenho. 6 ~ 120 dias
# uteis ~ 6 meses. Abaixo disso o veredito e INSUFICIENTE, por construcao.
DRIFT_MIN_EFFECTIVE_SAMPLES = 6
DRIFT_BOOTSTRAP_RESAMPLES = 2000
DRIFT_CI_ALPHA = 0.05  # IC bilateral de 95%
DRIFT_BOOTSTRAP_SEED = 42

# Retreino: o periodo ao vivo ja "maduro" (com rotulo de 20 dias resolvido)
# depois do fim do treino do modelo vigente e dividido em duas partes -- a
# primeira alimenta o treino do challenger (dado novo que o modelo vigente
# nao viu), a segunda e a janela de avaliacao, fora da amostra para os DOIS.
RETRAIN_EVAL_FRACTION = 0.5

# Ultima data de log gerada pelo codigo COM o bug do dia atrasado (corrigido
# em 2026-10-01). A checagem de consistencia so compara a partir da primeira
# linha posterior a esta -- sem isso, os dias antigos acusariam DIVERGE para
# sempre e esconderiam qualquer bug novo de verdade. Dados historicos nao sao
# apagados nem reescritos; apenas deixam de ser usados como referencia.
PAPER_TRADING_BUGGY_THROUGH = "2026-09-30"

# --- Analise de baselines justos e ablacao (run_baseline_ablation.py) ---
# Volatilidade-alvo (anual) do baseline "vol-target": exposicao = alvo / vol
# realizada de 20 dias, limitada a [0, 1]. Fixada antes de ver resultados.
VOL_TARGET_ANNUAL = 0.10
# Taxa anual ASSUMIDA para o caixa (parte nao investida), apenas para a
# coluna "com juros no caixa" -- valores medios de longo prazo aproximados,
# NAO dados de mercado; a coluna principal continua com caixa a 0%.
CASH_ANNUAL_RATE_ASSUMED = {"US": 0.025, "BR": 0.10}
