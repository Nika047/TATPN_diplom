# Ежедневная работа с GPU-компьютером

| | |
|---|---|
| Адрес домашнего ПК (Radmin VPN) | `26.244.190.10` |
| Пользователь | `kirill` |
| Проект на домашнем ПК | `C:\Users\kirill\Desktop\TATPN_diplom` |

## 1. На рабочем ПК (Windows 10)

1. Запустить Radmin VPN, убедиться, что домашний ПК "в сети".
2. В PowerShell проверить порт (нужно `TcpTestSucceeded : True`):

   ```powershell
   Test-NetConnection 26.244.190.10 -Port 22
   ```

3. Подключиться (пароль учётной записи, не PIN):

   ```powershell
   ssh kirill@26.244.190.10
   ```

## 2. На домашнем ПК (внутри ssh-сессии)

4. Войти в PowerShell, перейти в проект, подтянуть код:

   ```
   powershell
   cd C:\Users\kirill\Desktop\TATPN_diplom
   .venv\Scripts\activate
   git pull
   ```

5. Посмотреть, что идёт и как нагружена видеокарта:

   ```powershell
   Get-Process python -ErrorAction SilentlyContinue
   nvidia-smi
   ```

6. Посмотреть журнал последнего запуска:

   ```powershell
   $log = Get-ChildItem logs\run_logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
   Get-Content $log.FullName -Tail 25
   ```

7. Запустить этап (`N` - номер этапа):

   - **этапы 0, 1, 2, 6** - прямо в сессии:

     ```powershell
     python run_stageN.py
     ```

   - **этапы 3, 4, 5** - в фоне, чтобы обрыв ssh не остановил обучение:

     ```powershell
     $P = (Get-Location).Path
     Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"cd /d $P && .venv\Scripts\python.exe run_stageN.py > stageN.out 2>&1`"" }
     ```

     Нужно `ReturnValue = 0`. После этого ssh можно закрыть.
     При первом запуске проверить: закрыть ssh, зайти снова, `Get-Process python` - процесс должен быть жив.

8. Прочитать результат:

   ```powershell
   Get-Content logs\stage5_tatpn\summary.txt
   ```

   Папки сводок: `stage1_baselines`, `stage2_cnn`, `stage3_cnn_mamba`, `stage4_target`, `stage5_tatpn`.

## 3. Снова на рабочем ПК

9. Забрать сводку (новое окно PowerShell, не внутри ssh):

   ```powershell
   scp kirill@26.244.190.10:C:/Users/kirill/Desktop/TATPN_diplom/logs/stage5_tatpn/summary.txt .
   ```

## Порядок этапов

Следующий этап - только после того, как предыдущий прошёл проверки в своей сводке.
`run_stage0.py` -> `1` -> `2` -> `3` -> `4` -> `5` -> `6`.

## Полный цикл одной командой

`run_all.py` запускает этапы 0-6 подряд, на всех эпохах, и замеряет реальное
время каждого. Если этап упал - цикл останавливается; если у этапа не прошли
проверки - идёт дальше и помечает это в итоге.

В фоне (в PowerShell, в папке проекта):

```powershell
$P = (Get-Location).Path
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"cd /d $P && .venv\Scripts\python.exe -u run_all.py > run_all.out 2>&1`"" }
```

Следить и смотреть итог:

```powershell
Get-Content run_all.out -Tail 30
Get-Content logs\run_all\summary.txt
```

Продолжить с места остановки: `--from N` (например, `run_all.py --from 3`).

## Если не подключается

- Домашний ПК в сне или перезагрузился - включить, при необходимости войти в учётную запись на нём.
- Radmin VPN не запущен на одном из ПК.
- ПК "в сети", но порт закрыт - на домашнем ПК `Start-Service sshd`, проверить правило брандмауэра для порта 22.
- Рабочая сеть блокирует VPN.

## Остановить запуск

```powershell
Stop-Process -Name python -Force
```

Останавливает все процессы python на домашнем ПК.
