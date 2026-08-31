# Registra uma tarefa agendada do Windows para rodar o paper trading
# (Etapa 6) uma vez por dia, automaticamente.
#
# Rode este script UMA VEZ (como Administrador, se pedir) para configurar.
# Depois disso, o Windows executa o paper trading sozinho todo dia -- nao
# precisa mais rodar manualmente.
#
# Horario escolhido (19:00): depois do fechamento tanto da bolsa americana
# (16h ET = ~17h-18h horario de Brasilia dependendo do horario de verao)
# quanto da B3 (18h horario de Brasilia) -- garante que o preco de
# fechamento do dia ja esta disponivel para os 4 ativos (SPY, AAPL,
# PETR4.SA, VALE3.SA) antes do script rodar.

$pythonExe = "C:\Users\guilherme.cardoso\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$scriptPath = Join-Path $PSScriptRoot "run_stage6_paper_trading_daily.py"
$projectDir = $PSScriptRoot
$logPath = Join-Path $PSScriptRoot "paper_trading\daily_run_log.txt"

$action = New-ScheduledTaskAction `
    -Execute $pythonExe `
    -Argument "`"$scriptPath`" *>> `"$logPath`"" `
    -WorkingDirectory $projectDir

$trigger = New-ScheduledTaskTrigger -Daily -At "19:00"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask `
    -TaskName "QuantIA_PaperTrading_Diario" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Roda um dia de paper trading (Etapa 6) para todos os ativos configurados." `
    -Force

Write-Host "Tarefa 'QuantIA_PaperTrading_Diario' registrada com sucesso."
Write-Host "Vai rodar todo dia as 19:00, mesmo com o computador travado (nao se ele estiver desligado)."
Write-Host "Log de cada execucao: $logPath"
Write-Host ""
Write-Host "Para testar AGORA sem esperar ate as 19:00:"
Write-Host "  Start-ScheduledTask -TaskName 'QuantIA_PaperTrading_Diario'"
Write-Host ""
Write-Host "Para desativar (parar de rodar automaticamente):"
Write-Host "  Unregister-ScheduledTask -TaskName 'QuantIA_PaperTrading_Diario' -Confirm:`$false"
